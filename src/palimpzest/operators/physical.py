from __future__ import annotations

from palimpzest.constants import *
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.elements import Any
from palimpzest.solver import Solver
from palimpzest.tools.profiler import Profiler

from typing import Any, Callable, Dict, Tuple, Union

import concurrent
import os
import sys

# DEFINITIONS
IteratorFn = Callable[[], DataRecord]


class PhysicalOp:
    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    synthesizedFns = {}
    solver = Solver(verbose=LOG_LLM_OUTPUT)

    def __init__(self, outputSchema: Schema) -> None:
        self.outputSchema = outputSchema
        self.datadir = DataDirectory()

        # if profiling is set to True, collect execution statistics and history of transformations
        if Profiler.profiling_on():
            self.profiler = Profiler()

    def dumpPhysicalTree(self) -> Tuple[PhysicalOp, Union[PhysicalOp, None]]:
        raise NotImplementedError("Abstract method")

    def getProfilingData(self) -> Dict[str, Any]:
        raise NotImplementedError("Abstract method")

    def estimateCost(self) -> Dict[str, Any]:
        """Returns dict of time, cost, and quality metrics."""
        raise NotImplementedError("Abstract method")

class MarshalAndScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, datasetIdentifier: str, op_id: str=None):
        super().__init__(outputSchema=outputSchema)
        self.datasetIdentifier = datasetIdentifier
        self.op_id = op_id

    def __str__(self):
        return "MarshalAndScanDataOp(" + str(self.outputSchema) + ", " + self.datasetIdentifier + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)

    def getProfilingData(self):
        if Profiler.profiling_on():
            return self.profiler.get_data()
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        cardinality = self.datadir.getCardinality(self.datasetIdentifier) + 1
        size = self.datadir.getSize(self.datasetIdentifier)
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        datasetType = self.datadir.getRegisteredDatasetType(self.datasetIdentifier)
        timePerElement = (
            LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
            if datasetType in ["dir", "file"]
            else MEMORY_SCAN_TIME_PER_KB * perElementSizeInKb
        )
        usdPerElement = 0

        # TODO: similar to notes in other physical operators' estimateCost() function
        #       we will likely want to augment PZ to use sampling and/or real-time updates
        #       to estimates like these, rather then employing constants.
        #
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality)) # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT   # fraction of the element which is provided in context
            * BYTES_TO_TOKENS           # convert bytes to tokens
        )

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

    def __iter__(self) -> IteratorFn:
        @self.profiler.iter_profiler(name="base_scan", op_id=self.op_id)
        def iteratorFn():
            for nextCandidate in self.datadir.getRegisteredDataset(self.datasetIdentifier):
                yield nextCandidate

        return iteratorFn()

class CacheScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, cacheIdentifier: str, op_id: str=None):
        super().__init__(outputSchema=outputSchema)
        self.cacheIdentifier = cacheIdentifier
        self.op_id = op_id

    def __str__(self):
        return "CacheScanDataOp(" + str(self.outputSchema) + ", " + self.cacheIdentifier + ")"
    
    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)

    def getProfilingData(self):
        if Profiler.profiling_on():
            return self.profiler.get_data()
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        # TODO: at the moment, getCachedResult() looks up a pickled file that stores
        #       the cached data specified by self.cacheIdentifier, opens the file,
        #       and then returns an iterator over records in the pickled file.
        #
        #       I'm guessing that in the future we may want to load the cached data into
        #       the DataDirectory._cache object on __init__ (or in the background) so
        #       that this operation doesn't require a read from disk. If that happens, be
        #       sure to switch LOCAL_SCAN_TIME_PER_KB --> MEMORY_SCAN_TIME_PER_KB; and store
        #       metadata about the cardinality and size of cached data upfront so that we
        #       can access it in constant time.
        #
        #       At a minimum, we could use this function call to load the data into DataManager._cache
        #       since we have to iterate over it anyways; which would cache the data before the __iter__
        #       method below gets called.
        cached_data_info = [(1, sys.getsizeof(data)) for data in self.datadir.getCachedResult(self.cacheIdentifier)]
        cardinality = sum(list(map(lambda tup: tup[0], cached_data_info))) + 1
        size = sum(list(map(lambda tup: tup[1], cached_data_info)))
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        timePerElement = LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
        usdPerElement = 0

        # TODO: similar to notes in other physical operators' estimateCost() function
        #       we will likely want to augment PZ to use sampling and/or real-time updates
        #       to estimates like these, rather then employing constants.
        #
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality)) # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT   # fraction of the element which is provided in context
            * BYTES_TO_TOKENS           # convert bytes to tokens
        )

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

    def __iter__(self) -> IteratorFn:
        @self.profiler.iter_profiler(name="cache_scan", op_id=self.op_id)
        def iteratorFn():
            # NOTE: see comment in `estimateCost()` 
            for nextCandidate in self.datadir.getCachedResult(self.cacheIdentifier):
                yield nextCandidate
        return iteratorFn()


class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT, op_id: str=None, desc: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.op_id = op_id
        self.desc = desc
        self.targetCacheId = targetCacheId

        if outputSchema == ImageFile and source.outputSchema == File:
            self.model = Model.GPT_4V

        taskDescriptor = ("InduceFromCandidateOp", (self.model, prompt_strategy, op_id, desc), outputSchema, source.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputSchema) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data(model_name=self.model.value)
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        inputEstimates = self.source.estimateCost()

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # estimate number of output tokens as constant multiple of input tokens (for now)
        # 
        # TODO: we could get better est. if we could update plans in real-time (or use sampling)
        est_num_output_tokens = OUTPUT_TOKENS_MULTIPLE * est_num_input_tokens

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        # 
        # TODO: once again, real-time updates and/or sampling could improve est.
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        # TODO: the time is a linear function of the number of output tokens,
        #       if we look at the distribution of output tokens as we generate
        #       responses we can get better estimates in real-time. This of
        #       course would require modifying our design of PZ to enable it
        #       to be more of a bandit which can switch query plans as it observes
        #       query results.
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        #
        # TODO: once again, real-time updates and/or sampling could improve this est.
        if self.prompt_strategy == PromptStrategy.DSPY_COT:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # TODO: can selectivity be >1.0? Imagine an induce operation which extracts the authors from a research paper.
        selectivity = 1.0
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # NOTE: the following estimate assumes that nested generators effectively execute
        #       a single record at a time in sequence. I.e., there is no waterfall / time
        #       overlap for execution in two different stages of the chain of generators.
        #       The example below illustrates how this leads the total execution time to
        #       be equal to the 
        #
        #       >>> def f():
        #       ...   for idx in range(3):
        #       ...     time.sleep(2)
        #       ...     yield idx
        #
        #       >>> def g():
        #       ... for idx in f():
        #       ...     time.sleep(3)
        #       ...     yield idx
        #
        #       >>> def test():
        #       ...   start_time = time.time()
        #       ...   lst = [elt for elt in g()]
        #       ...   end_time = time.time()
        #       ...   print(f"duration: {end_time - start_time}")
        #
        #       >>> test()
        #       duration: 15.014
        #
        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * inputEstimates["cardinality"] + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # TODO: simple first hack -- use model's MMLU score / 100.0 to get a rough
        #       estimate of the quality in the range [0, 1]
        # 
        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["MMLU"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self) -> IteratorFn:
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="induce", op_id=self.op_id)
        def iteratorFn():    
            for nextCandidate in self.source:
                resultRecord = self._attemptMapping(nextCandidate)
                if resultRecord is not None:
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, resultRecord)
                    yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = ("InduceFromCandidateOp", (self.model, self.prompt_strategy, self.op_id, self.desc), self.outputSchema, candidate.schema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelInduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT, op_id: str=None, desc: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.op_id = op_id
        self.desc = desc
        self.targetCacheId = targetCacheId
        self.max_workers = 20

        taskDescriptor = ("ParallelInduceFromCandidateOp", (model, prompt_strategy, op_id, desc), outputSchema, source.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "ParallelInduceFromCandidateOp(" + str(self.outputSchema) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data(model_name=self.model.value)
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        """
        See InduceFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        inputEstimates = self.source.estimateCost()

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # estimate number of output tokens as constant multiple of input tokens (for now)
        est_num_output_tokens = OUTPUT_TOKENS_MULTIPLE * est_num_input_tokens

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        selectivity = 1.0
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * (inputEstimates["cardinality"] / self.max_workers) + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["MMLU"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="p_induce", op_id=self.op_id)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks using self.max_workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._attemptMapping, inputs))

                for resultRecord in results:
                    if resultRecord is not None:
                        if shouldCache:
                            self.datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = ("ParallelInduceFromCandidateOp", (self.model, self.prompt_strategy, self.op_id, self.desc), self.outputSchema, candidate.schema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class FilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_BOOL, op_id: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.filter = filter
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.op_id = op_id
        self.targetCacheId = targetCacheId

        taskDescriptor = ("FilterCandidateOp", (filter, model, prompt_strategy, op_id), source.outputSchema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "FilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data(model_name=self.model.value)
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        """
        See InduceFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        inputEstimates = self.source.estimateCost()

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # TODO: use sampling / real-time feedback to better estimate selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * inputEstimates["cardinality"] + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="filter", op_id=self.op_id)
        def iteratorFn():
            for nextCandidate in self.source:
                if self._passesFilter(nextCandidate):
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, nextCandidate)
                    yield nextCandidate

                # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                # the profiler will check the nextCandidate._passed_filter field to see if it needs to be dropped
                elif self.profiler.profiling_on():
                    yield nextCandidate

            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("FilterCandidateOp", (self.filter, self.model, self.prompt_strategy, self.op_id), candidate.schema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelFilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_BOOL, op_id: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.filter = filter
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.op_id = op_id
        self.targetCacheId = targetCacheId
        self.max_workers = 20

        taskDescriptor = ("ParallelFilterCandidateOp", (filter, model, prompt_strategy, op_id), source.outputSchema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "ParallelFilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data(model_name=self.model.value)
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        inputEstimates = self.source.estimateCost()

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # TODO: use sampling / real-time feedback to better estimate selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * (inputEstimates["cardinality"] / self.max_workers) + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="p_filter", op_id=self.op_id)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source: 
                inputs.append(nextCandidate)

            # Grab items from the list of inputs in chunks using self.max_workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._passesFilter, inputs))

                for idx, filterResult in enumerate(results):
                    if filterResult:
                        resultRecord = inputs[idx]
                        if shouldCache:
                            self.datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord

                    # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                    # the profiler will check the nextCandidate._passed_filter field to see if it needs to be dropped
                    elif self.profiler.profiling_on():
                        yield nextCandidate

            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("ParallelFilterCandidateOp", (self.filter, self.model, self.prompt_strategy, self.op_id), candidate.schema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)

        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ApplyCountAggregateOp(PhysicalOp):
    def __init__(self, source: PhysicalOp, aggFunction: AggregateFunction, op_id: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=Number)
        self.source = source
        self.aggFunction = aggFunction
        self.op_id = op_id
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "ApplyCountAggregateOp(" + str(self.outputSchema) + ", " + "Function: " + str(self.aggFunction) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data()
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        inputEstimates = self.source.estimateCost()

        outputEstimates = {**inputEstimates}
        outputEstimates['cardinality'] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates['timePerElement'] = 0
        outputEstimates['usdPerElement'] = 0
        outputEstimates['estOutputTokensPerElement'] = 0

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="count", op_id=self.op_id)
        def iteratorFn():
            counter = 0
            for _ in self.source:
                counter += 1

            dr = DataRecord(Number)
            dr.value = counter
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyAverageAggregateOp(PhysicalOp):
    def __init__(self, source: PhysicalOp, aggFunction: AggregateFunction, op_id: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=Number)
        self.source = source
        self.aggFunction = aggFunction
        self.op_id = op_id
        self.targetCacheId = targetCacheId

        if not source.outputSchema == Number:
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

    def __str__(self):
        return "ApplyAverageAggregateOp(" + str(self.outputSchema) + ", " + "Function: " + str(self.aggFunction) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data()
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        inputEstimates = self.source.estimateCost()

        outputEstimates = {**inputEstimates}
        outputEstimates["cardinality"] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="average", op_id=self.op_id)
        def iteratorFn():
            sum = 0
            counter = 0
            for nextCandidate in self.source:
                try:
                    sum += int(nextCandidate.value)
                    counter += 1
                except:
                    pass

            dr = DataRecord(Number)
            dr.value = sum / float(counter)
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class LimitScanOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, limit: int, op_id: str=None, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.limit = limit
        self.op_id = op_id
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "LimitScanOp(" + str(self.outputSchema) + ", " + "Limit: " + str(self.limit) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self):
        if Profiler.profiling_on():
            source_data = self.source.getProfilingData()
            operator_data = self.profiler.get_data()
            operator_data["source"] = source_data
            return operator_data
        else:
            raise Exception("Profiling was not turned on; please set PZ_PROFILING=TRUE in your shell.")

    def estimateCost(self):
        inputEstimates = self.source.estimateCost()

        outputEstimates = {**inputEstimates}
        outputEstimates["cardinality"] = min(self.limit, inputEstimates["cardinality"])

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profiler.iter_profiler(name="limit", op_id=self.op_id)
        def iteratorFn():
            counter = 0
            for nextCandidate in self.source: 
                if counter >= self.limit:
                    break
                if shouldCache:
                    datadir.appendCache(self.targetCacheId, nextCandidate)
                yield nextCandidate
                counter += 1

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()

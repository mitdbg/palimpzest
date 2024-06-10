from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator
from .physical import PhysicalOperator

from palimpzest.constants import *
from palimpzest.corelib import Schema
from palimpzest.elements import *
from palimpzest.profiler import RecordOpStats, OperatorCostEstimates

from typing import Any, Dict, Optional

import concurrent
import multiprocessing


class FilterOp(PhysicalOperator):
    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        filter: Filter,
        targetCacheId: str = None,
        shouldProfile=False,
        max_workers=1,
    ):
        assert inputSchema == outputSchema, "Input and output schemas must match for FilterOp"
        super().__init__(inputSchema=inputSchema, outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.filter = filter
        self.targetCacheId = targetCacheId
        self.max_workers = max_workers

    def copy(self):
        return self.__class__(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            filter=self.filter,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            max_workers=self.max_workers,
        )

    def physical_op_id(self, plan_position: Optional[int] = None):
        op_dict = {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "filter": str(self.filter),
        }

        return self._compute_op_id_from_dict(op_dict, plan_position)

    def __iter__(self):
        # TODO GV Why is this logic in the __iter__ method and not in execution?
        #      MR: this logic will be moved to plan.execute(); top of my TODO list tomorrow
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="filter", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.source:
                resultRecord = self.__call__(nextCandidate)
                if resultRecord._passed_filter:
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, resultRecord)
                    yield resultRecord

                # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                # the profiler will check the resultRecord._passed_filter field to see if it needs to be dropped
                elif self.shouldProfile:
                    yield resultRecord

            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()


# TODO: delete once __call__ methods are implemented in NonLLLMFilter and LLMFilter
class ParallelFilterCandidateOp(FilterOp):

    def __init__(self, streaming=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = multiprocessing.cpu_count()
        self.streaming = streaming

    def copy(self):
        return super().copy(streaming=self.streaming)

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_filter", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            if self.streaming:
                chunksize = self.max_workers
            else:
                chunksize = len(inputs)

            # Grab items from the list of inputs in chunks using self.max_workers
            for i in range(0, len(inputs), chunksize):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    results = list(
                        executor.map(self._passesFilter, inputs[i : i + chunksize])
                    )

                    for resultRecord in results:
                        if resultRecord._passed_filter:
                            if shouldCache:
                                self.datadir.appendCache(
                                    self.targetCacheId, resultRecord
                                )
                            yield resultRecord

                        # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                        # the profiler will check the resultRecord._passed_filter field to see if it needs to be dropped
                        elif self.shouldProfile:
                            yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class NonLLMFilter(FilterOp):

    def __eq__(self, other: NonLLMFilter):
        return (
            isinstance(other, self.__class__)
            and self.filter == other.filter
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return f"{self.op_name()}({str(self.outputSchema)}, Filter: {str(self.filter)})"

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate 1 ms single-threaded execution for filter function
        time_per_record = 0.001 / self.max_workers

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    # TODO: modify to support parallel execution and RecordOpStats collection
    def __call__(self, candidate: DataRecord):
        # start_time = time.time()
        result = self.filter.filterFn(candidate)
        # fn_call_duration_secs = time.time() - start_time
        # if profiling, set record's stats for the given op_id
        # if shouldProfile:
        # candidate._stats[td.op_id] = FilterNonLLMStats(
        # fn_call_duration_secs=fn_call_duration_secs,
        # filter=str(td.filter.filterFn),
        # )
        # set _passed_filter attribute and return record
        setattr(candidate, "_passed_filter", result)
        print(f"ran filter function on {candidate}")

        return candidate


class LLMFilter(FilterOp):

    def __init__(self, model: Model, prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_BOOL, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy

    def __eq__(self, other: LLMFilter):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.filter == other.filter
            and self.prompt_strategy == other.prompt_strategy
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return f"{self.op_name()}({str(self.outputSchema)}, Filter: {str(self.filter)}, Model: {self.model.value}, Prompt Strategy: {str(self.prompt_strategy.value)})"

    def copy(self):
        return self.__class__(
            model=self.model,
            prompt_strategy=self.prompt_strategy,
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            filter=self.filter,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            max_workers=self.max_workers,
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS

        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"]
            * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    # TODO: modify to support parallel execution and RecordOpStats collection
    def __call__(self, candidate: DataRecord):
        # compute record schema and type
        doc_schema = str(self.inputSchema)
        doc_type = self.inputSchema.className()

        # TODO: is this needed anymore?
        # do not filter candidate if it doesn't match inputSchema
        if not candidate.schema == self.inputSchema:
            return False

        # create generator
        generator = None
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            generator = DSPyGenerator(
                self.model.value,
                self.prompt_strategy,
                doc_schema,
                doc_type,
                self._verbose,
            )
        # TODO
        elif self.prompt_strategy == PromptStrategy.ZERO_SHOT:
            raise Exception("not implemented yet")
        # TODO
        elif self.prompt_strategy == PromptStrategy.FEW_SHOT:
            raise Exception("not implemented yet")
        # TODO
        elif self.prompt_strategy == PromptStrategy.CODE_GEN_BOOL:
            raise Exception("not implemented yet")

        # invoke LLM to generate filter decision (True or False)
        text_content = candidate._asJSON(include_bytes=False)
        try:
            response, _, gen_stats = generator.generate(
                context=text_content,
                question=self.filter.filterCondition,
            )

            # if profiling, set record's stats for the given op_id
            # if shouldProfile:
            # candidate._stats[td.op_id] = FilterLLMStats(
            # gen_stats=gen_stats, filter=td.filter.filterCondition
            # )

            # set _passed_filter attribute and return record
            setattr(candidate, "_passed_filter", "true" in response.lower())
        except Exception as e:
            # If there is an exception consider the record as not passing the filter
            print(f"Error invoking LLM for filter: {e}")
            setattr(candidate, "_passed_filter", False)

        return candidate

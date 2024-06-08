from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator
from .physical import PhysicalOperator, MAX_ID_CHARS

from palimpzest.constants import *
from palimpzest.corelib import Schema
from palimpzest.elements import *
from palimpzest.profiler import Profiler

from typing import Any, Dict

import concurrent
import hashlib
import json


class FilterOp(PhysicalOperator):

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        filter: Filter,
        targetCacheId: str = None,
        shouldProfile=False,
        max_workers=1,
        *args,
        **kwargs,
    ):
        assert inputSchema == outputSchema, "Input and output schemas must match for FilterOp"
        super().__init__(inputSchema=inputSchema, outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.filter = filter
        self.targetCacheId = targetCacheId
        self.max_workers = max_workers

        # # NOTE: need to construct profiler after all fields used by self.opId() are set
        # self.profiler = Profiler(op_id=self.opId())
        # self.profile = self.profiler.iter_profiler

        # # construct TaskDescriptor
        # taskDescriptor = self._makeTaskDescriptor()

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.filter == other.filter
            and self.prompt_strategy == other.prompt_strategy
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        model_str = self.model.value if self.model is not None else str(None)
        return f"{self.__class__.__name__}({str(self.outputSchema)}, Filter: {str(self.filter)}, Model: {model_str}, Prompt Strategy: {str(self.prompt_strategy.value)})"

    def copy(self, *args, **kwargs):
        return self.__class__(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            model=self.model,
            filter=self.filter,
            prompt_strategy=self.prompt_strategy,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            max_workers=self.max_workers,
            *args,
            **kwargs,
        )

    def opId(self):
        d = {
            "operator": self.__class__.__name__,
            "outputSchema": str(self.outputSchema),
            "filter": str(self.filter),
            "model": self.model.value if self.model is not None else None,
            "prompt_strategy": self.prompt_strategy.value,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        """
        See ConvertFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        # fetch cost estimates from source operation
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)

        filter_str = (
            self.filter.filterCondition
            if self.filter.filterCondition is not None
            else str(self.filter.filterFn)
        )
        op_filter = f"(filter == '{str(filter_str)}') & (op_name == 'filter' | op_name == 'p_filter')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # get estimate data for this physical op's model
            model_name = None if self.model is None else self.model.value
            time_per_record = cost_est_data[op_filter][model_name]["time_per_record"]
            usd_per_record = cost_est_data[op_filter][model_name]["cost_per_record"]
            est_num_input_tokens = cost_est_data[op_filter][model_name][
                "est_num_input_tokens"
            ]
            est_num_output_tokens = cost_est_data[op_filter][model_name][
                "est_num_output_tokens"
            ]
            selectivity = cost_est_data[op_filter][model_name]["selectivity"]
            quality = cost_est_data[op_filter][model_name]["quality"]

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates["cardinality"] * selectivity

            # apply quality for this filter to overall quality est.
            quality = (
                inputEstimates["quality"]
                if self.model is None
                else inputEstimates["quality"] * quality
            )

            thisCostEst = {
                "time_per_record": time_per_record,
                "usd_per_record": usd_per_record,
                "est_num_output_tokens": est_num_output_tokens,
                "selectivity": selectivity,
                "quality": quality,
            }

            costEst = {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates["cumulativeTimePerElement"]
                + time_per_record,
                "cumulativeUSDPerElement": inputEstimates["cumulativeUSDPerElement"]
                + usd_per_record,
                "totalTime": cardinality * time_per_record
                + inputEstimates["totalTime"],
                "totalUSD": cardinality * usd_per_record + inputEstimates["totalUSD"],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
            }

            return costEst, {
                "cumulative": costEst,
                "thisPlan": thisCostEst,
                "subPlan": subPlanCostEst,
            }

        # otherwise, if this filter is a function call (not an LLM call) estimate accordingly
        if self.filter.filterFn is not None:
            # estimate output cardinality using a constant assumption of the filter selectivity
            selectivity = EST_FILTER_SELECTIVITY
            cardinality = selectivity * inputEstimates["cardinality"]

            # estimate 1 ms execution for filter function
            time_per_record = 0.001
            # (divide non-parallel est. by 10x for parallelism speed-up)
            if self.max_workers > 1:
                time_per_record /= 10

            # assume filter fn has perfect quality
            quality = inputEstimates["quality"]

            thisCostEst = {
                "time_per_record": time_per_record,
                "usd_per_record": 0.0,
                "est_num_output_tokens": inputEstimates["estOutputTokensPerElement"],
                "selectivity": selectivity,
                "quality": quality,
            }

            costEst = {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": 0.0,
                "cumulativeTimePerElement": inputEstimates["cumulativeTimePerElement"]
                + time_per_record,
                "cumulativeUSDPerElement": inputEstimates["cumulativeUSDPerElement"],
                "totalTime": cardinality * time_per_record
                + inputEstimates["totalTime"],
                "totalUSD": inputEstimates["totalUSD"],
                # next operator processes input based on contents, not T/F output by this operator
                "estOutputTokensPerElement": inputEstimates[
                    "estOutputTokensPerElement"
                ],
                "quality": quality,
            }

            return costEst, {
                "cumulative": costEst,
                "thisPlan": thisCostEst,
                "subPlan": subPlanCostEst,
            }

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
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"]
            * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"]
            * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = (
            model_conversion_time_per_record
            + inputEstimates["cumulativeTimePerElement"]
        )
        cumulativeUSDPerElement = (
            model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]
        )

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = (
            model_conversion_time_per_record
            * (inputEstimates["cardinality"] / self.max_workers)
            + inputEstimates["totalTime"]
        )
        totalUSD = (
            model_conversion_usd_per_record * inputEstimates["cardinality"]
            + inputEstimates["totalUSD"]
        )

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates[
            "quality"
        ]

        costEst = {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            # next operator processes input based on contents, not T/F output by this operator
            "estOutputTokensPerElement": inputEstimates["estOutputTokensPerElement"],
            "quality": quality,
        }

        return costEst, {"cumulative": costEst, "thisPlan": costEst, "subPlan": None}

    def __call__(self, candidate) -> bool:
        raise NotImplementedError("You are calling a method from the abstract class!")

    def __iter__(self):
        # TODO GV Why is this logic in the __iter__ method and not in execution?
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


class ParallelFilterCandidateOp(FilterOp):

    def __init__(self, streaming=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 32  # TODO this is hardcoded?
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

    def __call__(self, candidate: DataRecord):
        # start_time = time.time()
        result = self.filter(candidate)
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

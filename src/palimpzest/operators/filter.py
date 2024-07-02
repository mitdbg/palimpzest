from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator
from palimpzest.utils.model_helpers import getVisionModels
from .physical import PhysicalOperator, DataRecordsWithStats

from palimpzest.constants import *
from palimpzest.dataclasses import GenerationStats, RecordOpStats
from palimpzest.corelib import Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord, Filter
from palimpzest.operators import logical

from typing import List

import time


class FilterOp(PhysicalOperator):
    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        filter: Filter,
        targetCacheId: str = None,
        shouldProfile=False,
        max_workers=1,
        *args, **kwargs
    ):
        assert inputSchema == outputSchema, "Input and output schemas must match for FilterOp"
        super().__init__(inputSchema=inputSchema, outputSchema=outputSchema, shouldProfile=shouldProfile, *args, **kwargs)
        self.filter = filter
        self.targetCacheId = targetCacheId
        self.max_workers = max_workers


    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "filter": str(self.filter),
        }


    def __eq__(self, other: FilterOp):
        return (
            isinstance(other, self.__class__)
            and self.filter == other.filter
            and self.inputSchema == other.outputSchema
            and self.outputSchema == other.outputSchema
        )

    def copy(self):
        return self.__class__(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            filter=self.filter,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            max_workers=self.max_workers,
        )

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


class NonLLMFilter(FilterOp):
    implemented_op = logical.FilteredScan
    final = True

    @classmethod
    def implements(cls, logical_operator_class):
        if not logical_operator_class == cls.implemented_op:
            return False
        # logical_operator is a class
        if isinstance(logical_operator_class, type): 
            return logical_operator_class == cls.implemented_op

    @classmethod
    def materializes(cls, logical_operator: logical.LogicalOperator):
        if not isinstance(logical_operator, cls.implemented_op):
            return False
        return logical_operator.filter.filterFn is not None

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

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        # apply filter to input record
        start_time = time.time()
        try:
            result = self.filter.filterFn(candidate)
        except Exception as e:
            print(f"Error invoking user-defined function for filter: {e}")

        # time spent executing the filter function
        fn_call_duration_secs = time.time() - start_time

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            record_state=candidate._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=fn_call_duration_secs,
            cost_per_record=0.0,
            filter_str=self.filter.getFilterStr(),
            passed_filter=result,
            fn_call_duration_secs=fn_call_duration_secs,
            answer=result,
        )

        # set _passed_filter attribute and return
        setattr(candidate, "_passed_filter", result)

        return [candidate], [record_op_stats]


class LLMFilter(FilterOp):
    implemented_op = logical.FilteredScan
    model = None
    prompt_strategy = PromptStrategy.DSPY_COT_BOOL
   
    @classmethod
    def materializes(cls, logical_operator: logical.LogicalOperator):
        if not isinstance(logical_operator, cls.implemented_op):
            return False
        if cls.model in getVisionModels():
            return False
        return logical_operator.filter.filterCondition is not None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        doc_schema = str(self.inputSchema)
        doc_type = self.inputSchema.className()
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            self.generator = DSPyGenerator(
                self.model.value,
                self.prompt_strategy,
                doc_schema,
                doc_type,
                verbose=False, # TODO pass verbose argument
            )

        else:
            raise Exception(f"Prompt strategy {self.prompt_strategy} implemented yet")
        

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
        ) / self.max_workers

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

    def __call__(self, candidate: DataRecord) -> DataRecordsWithStats:
        start_time = time.time()

        # invoke LLM to generate filter decision (True or False)
        text_content = candidate._asJSONStr(include_bytes=False)
        response, gen_stats = None, GenerationStats()
        try:
            response, gen_stats = self.generator.generate(
                context=text_content,
                question=self.filter.filterCondition,
            )
        except Exception as e:
            print(f"Error invoking LLM for filter: {e}")

        # compute whether the record passed the filter or not
        passed_filter = (
            "true" in response.lower()
            if response is not None
            else False
        )

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            record_state=candidate._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=gen_stats.cost_per_record,
            model_name=self.model.value,
            filter_str=self.filter.getFilterStr(),
            total_input_tokens=gen_stats.total_input_tokens,
            total_output_tokens=gen_stats.total_output_tokens,
            total_input_cost=gen_stats.total_input_cost,
            total_output_cost=gen_stats.total_output_cost,
            llm_call_duration_secs=gen_stats.llm_call_duration_secs,
            answer=response,
            passed_filter=passed_filter,
        )

        # set _passed_filter attribute and return
        setattr(candidate, "_passed_filter", passed_filter)

        return [candidate], [record_op_stats]

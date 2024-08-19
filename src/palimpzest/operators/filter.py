from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator
from .physical import PhysicalOperator, DataRecordsWithStats

from palimpzest.constants import *
from palimpzest.dataclasses import GenerationStats, RecordOpStats
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord, Filter

from typing import List

import time


class FilterOp(PhysicalOperator):
    def __init__(self, filter: Filter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.inputSchema == self.outputSchema, "Input and output schemas must match for FilterOp"
        self.filter = filter

    def get_op_params(self):
        return {
            "outputSchema": self.outputSchema,
            "filter": self.filter,
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
            verbose=self.verbose,
            max_workers=self.max_workers,
        )


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
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
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

        if self.verbose:
            output_str = f"{self.filter.getFilterStr()}:\n{result}"
            print(output_str)

        return [candidate], [record_op_stats]


class LLMFilter(FilterOp):

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_BOOL,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy

        doc_schema = str(self.inputSchema)
        doc_type = self.inputSchema.className()
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            self.generator = DSPyGenerator(
                self.model.value,
                self.prompt_strategy,
                doc_schema,
                doc_type,
                verbose=self.verbose,
            )

        else:
            raise Exception(f"Prompt strategy {self.prompt_strategy} not implemented yet")

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"model": self.model, **op_params}

        return op_params

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
            model=self.model,
            prompt_strategy=self.prompt_strategy,
            filter=self.filter,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            verbose=self.verbose,
            max_workers=self.max_workers,
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS

        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

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
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
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

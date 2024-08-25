from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator, ImageTextGenerator
from .physical import PhysicalOperator, DataRecordsWithStats

from palimpzest.constants import *
from palimpzest.dataclasses import GenerationStats, RecordOpStats
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord, Filter
from palimpzest.prompts import IMAGE_FILTER_PROMPT

from typing import List

import base64
import time


class FilterOp(PhysicalOperator):
    def __init__(self, filter: Filter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.inputSchema == self.outputSchema, "Input and output schemas must match for FilterOp"
        self.filter = filter

    def __str__(self):
        op = super().__str__()
        op += f"    Filter: {str(self.filter)}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"filter": self.filter, **copy_kwargs}

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


class NonLLMFilter(FilterOp):

    def __eq__(self, other: NonLLMFilter):
        return (
            isinstance(other, self.__class__)
            and self.filter == other.filter
            and self.outputSchema == other.outputSchema
        )

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
        image_filter: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.image_filter = image_filter

        doc_schema = str(self.inputSchema)
        doc_type = self.inputSchema.className()
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            if not self.image_filter:
                self.generator = DSPyGenerator(
                    self.model.value,
                    self.prompt_strategy,
                    doc_schema,
                    doc_type,
                    verbose=self.verbose,
                )
            else:
                self.generator = ImageTextGenerator(self.model.value, self.verbose)

        else:
            raise Exception(f"Prompt strategy {self.prompt_strategy} not implemented yet")

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "image_filter": self.image_filter,
            **copy_kwargs
        }

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
            and self.image_filter == other.image_filter
            and self.outputSchema == other.outputSchema
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        if self.image_filter:
            est_num_input_tokens = 765 / 10 # 1024x1024 image is 765 tokens

        # NOTE: in truth, the DSPy COT output often generates an entire reasoning sentence,
        #       thus the true value may be higher
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
        quality = (
            (MODEL_CARDS[self.model.value]["MMLU"] / 100.0) * source_op_cost_estimates.quality
            if self.image_filter
            else (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * source_op_cost_estimates.quality
        )

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordsWithStats:
        start_time = time.time()

        # parse the content from the candidate record
        content = None
        if self.image_filter:
            base64_images = []
            if hasattr(candidate, "contents"):
                # TODO: should address this now; we need a way to infer (or have the programmer declare) what fields contain image content
                base64_images = [
                    base64.b64encode(candidate.contents).decode("utf-8")  
                ]
            else:
                base64_images = [
                    base64.b64encode(image).decode("utf-8")
                    for image in candidate.image_contents  # TODO: (see note above)
                ]
            content = base64_images
        else:
            content = candidate._asJSONStr(include_bytes=False)

        # construct the prompt; for image filters we need to wrap the filter condition in an instruction 
        prompt = self.filter.filterCondition
        if self.image_filter:
            prompt = IMAGE_FILTER_PROMPT.format(filter_condition=self.filter.filterCondition)

        # invoke LLM to generate filter decision (True or False)
        response, gen_stats = None, GenerationStats()
        try:
            response, gen_stats = self.generator.generate(context=content, question=prompt)
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

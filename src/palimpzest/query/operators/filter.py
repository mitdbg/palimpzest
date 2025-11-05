from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic.fields import FieldInfo

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_FILTER_SELECTIVITY,
    NAIVE_EST_NUM_INPUT_TOKENS,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator


class FilterOp(PhysicalOperator, ABC):
    def __init__(self, filter: Filter, desc: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.input_schema == self.output_schema, "Input and output schemas must match for FilterOp"
        self.filter_obj = filter
        self.desc = desc

    def __str__(self):
        op = super().__str__()
        op += f"    Filter: {str(self.filter_obj)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"filter": str(self.filter_obj), "desc": self.desc, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"filter": self.filter_obj, "desc": self.desc, **op_params}

    @abstractmethod
    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        """
        This abstract method will be implemented by subclasses of FilterOp to process the input DataRecord
        and generate the True / False determination of whether the input record passes the filter. A dictionary
        mapping a "passed_operator" key to the T/F boolean is returned along with the GenerationStats object.

        For example, if the input DataRecord (i.e. `candidate`) contains an image of a dog, and the filter
        operation is supposed to filter for images with dogs, then the output would be:

        ({"passed_operator": True}, GenerationStats(...))

        A post-condition of this method is that the "passed_operator" key must be present in the output dictionary,
        and it's value must be a boolean. If there is an error, then the value for "passed_operator" must be False.
        """
        pass

    def _create_record_set(
        self,
        candidate: DataRecord,
        passed_operator: bool,
        generation_stats: GenerationStats,
        total_time: float,
        answer: dict[str, Any],
    ) -> DataRecordSet:
        """
        Given an input DataRecord and a determination of whether it passed the filter or not,
        construct the resulting RecordSet.
        """
        # create new DataRecord and set passed_operator attribute
        dr = DataRecord.from_parent(schema=candidate.schema, data_item={}, parent_record=candidate)
        dr._passed_operator = passed_operator

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=total_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=self.get_model_name(),
            filter_str=self.filter_obj.get_filter_str(),
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_embedding_input_tokens=generation_stats.total_embedding_input_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            total_embedding_cost=generation_stats.total_embedding_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            answer=answer,
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # apply the filter operation
        field_answers, generation_stats = self.filter(candidate)

        # create and return record set
        record_set = self._create_record_set(
            candidate,
            field_answers["passed_operator"],
            generation_stats,
            time.time() - start_time,
            field_answers
        )

        return record_set


class NonLLMFilter(FilterOp):

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate 1 ms single-threaded execution for filter function
        time_per_record = 0.001

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )
    
    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        # apply filter function to input record
        start_time = time.time()
        answer = {}
        try:
            # execute the UDF filter
            passed_operator = self.filter_obj.filter_fn(candidate.to_dict())
            answer = {"passed_operator": passed_operator}

            if self.verbose:
                print(f"{self.filter_obj.get_filter_str()}:\n{passed_operator}")

        except Exception as e:
            print(f"Error invoking user-defined function for filter: {e}")
            raise e

        # create generation stats object containing the time spent executing the UDF function
        generation_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)

        return answer, generation_stats


class LLMFilter(FilterOp):
    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.FILTER,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        if model is not None:
            self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "model": None if self.model is None else self.model.value,
            "prompt_strategy": None if self.prompt_strategy is None else self.prompt_strategy.value,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }

        return op_params

    def get_model_name(self):
        return None if self.model is None else self.model.value

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        if self.is_image_op():
            est_num_input_tokens = 765 / 10  # 1024x1024 image is 765 tokens

        # NOTE: the output often generates an entire reasoning sentence, thus the true value may be higher
        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        usd_per_input_token = (
            MODEL_CARDS[self.model.value]["usd_per_audio_input_token"]
            if self.is_audio_op()
            else MODEL_CARDS[self.model.value]["usd_per_input_token"]
        )
        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["overall"] / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        # get the set of input fields to use for the filter operation
        input_fields = self.get_input_fields()

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "filter_condition": self.filter_obj.filter_condition}

        # generate output; NOTE: FieldInfo is used to indicate the output type; thus, the desc is not needed
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the record passed the filter operation")}
        field_answers, _, generation_stats, _ = self.generator(candidate, fields, **gen_kwargs)

        return field_answers, generation_stats

from __future__ import annotations

from typing import Any

from pydantic.fields import FieldInfo

from palimpzest.constants import MODEL_CARDS, Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats, OperatorCostEstimates
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter

# TYPE DEFINITIONS
FieldName = str


class CritiqueAndRefineConvert(LLMConvert):

    def __init__(
        self,
        critic_model: Model,
        refine_model: Model,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.critic_model = critic_model
        self.refine_model = refine_model

        # create generators
        self.critic_generator = Generator(self.critic_model, PromptStrategy.MAP_CRITIC, self.reasoning_effort, self.api_base, self.cardinality, self.desc, self.verbose)
        self.refine_generator = Generator(self.refine_model, PromptStrategy.MAP_REFINE, self.reasoning_effort, self.api_base, self.cardinality, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Critic Model: {self.critic_model}\n"
        op += f"    Refine Model: {self.refine_model}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "critic_model": self.critic_model.value,
            "refine_model": self.refine_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "critic_model": self.critic_model,
            "refine_model": self.refine_model,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are invoking `self.model`, then critiquing its output with `self.critic_model`, and
        finally refining the output with `self.refine_model`. Thus, we roughly expect to incur the cost
        and time of three LLMConverts. In practice, this naive quality estimate will be overwritten by the
        CostModel's estimate once it executes a few instances of the operator.
        """
        # get naive cost estimates for first LLM call and multiply by 3 for now;
        # of course we should sum individual estimates for each model, but this is a rough estimate
        # and in practice we will need to revamp our naive cost estimates in the near future
        naive_op_cost_estimates = 3 * super().naive_cost_estimates(source_op_cost_estimates)

        # for naive setting, estimate quality as quality of refine model
        model_quality = MODEL_CARDS[self.refine_model.value]["overall"] / 100.0
        naive_op_cost_estimates.quality = model_quality
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # NOTE: when I merge in the `abacus` branch, I will want to update this to reflect the changes I made to reasoning extraction
        # execute the initial model
        original_gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}
        field_answers, reasoning, original_gen_stats, original_messages = self.generator(candidate, fields, **original_gen_kwargs)
        original_output = f"REASONING: {reasoning}\nANSWER: {field_answers}\n"

        # execute the critic model
        critic_gen_kwargs = {"original_output": original_output, "original_messages": original_messages, **original_gen_kwargs}
        _, reasoning, critic_gen_stats, _ = self.critic_generator(candidate, fields, json_output=False, **critic_gen_kwargs)
        critique_output = f"CRITIQUE: {reasoning}\n"

        # execute the refinement model
        refine_gen_kwargs = {"critique_output": critique_output, **critic_gen_kwargs}
        field_answers, reasoning, refine_gen_stats, _ = self.refine_generator(candidate, fields, **refine_gen_kwargs)

        # compute the total generation stats
        generation_stats = original_gen_stats + critic_gen_stats + refine_gen_stats

        return field_answers, generation_stats


class CritiqueAndRefineFilter(LLMFilter):

    def __init__(
        self,
        critic_model: Model,
        refine_model: Model,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.critic_model = critic_model
        self.refine_model = refine_model

        # create generators
        self.critic_generator = Generator(self.critic_model, PromptStrategy.FILTER_CRITIC, self.reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)
        self.refine_generator = Generator(self.refine_model, PromptStrategy.FILTER_REFINE, self.reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Critic Model: {self.critic_model}\n"
        op += f"    Refine Model: {self.refine_model}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "critic_model": self.critic_model.value,
            "refine_model": self.refine_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "critic_model": self.critic_model,
            "refine_model": self.refine_model,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are invoking `self.model`, then critiquing its output with `self.critic_model`, and
        finally refining the output with `self.refine_model`. Thus, we roughly expect to incur the cost
        and time of three LLMFilters. In practice, this naive quality estimate will be overwritten by the
        CostModel's estimate once it executes a few instances of the operator.
        """
        # get naive cost estimates for first LLM call and multiply by 3 for now;
        # of course we should sum individual estimates for each model, but this is a rough estimate
        # and in practice we will need to revamp our naive cost estimates in the near future
        naive_op_cost_estimates = 3 * super().naive_cost_estimates(source_op_cost_estimates)

        # for naive setting, estimate quality as quality of refine model
        model_quality = MODEL_CARDS[self.refine_model.value]["overall"] / 100.0
        naive_op_cost_estimates.quality = model_quality
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # construct output fields
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the record passed the filter operation")}

        # NOTE: when I merge in the `abacus` branch, I will want to update this to reflect the changes I made to reasoning extraction
        # execute the initial model
        original_gen_kwargs = {"project_cols": input_fields, "filter_condition": self.filter_obj.filter_condition}
        field_answers, reasoning, original_gen_stats, original_messages = self.generator(candidate, fields, **original_gen_kwargs)
        original_output = f"REASONING: {reasoning}\nANSWER: {str(field_answers['passed_operator']).upper()}\n"

        # execute the critic model
        critic_gen_kwargs = {"original_output": original_output, "original_messages": original_messages, **original_gen_kwargs}
        _, reasoning, critic_gen_stats, _ = self.critic_generator(candidate, fields, json_output=False, **critic_gen_kwargs)
        critique_output = f"CRITIQUE: {reasoning}\n"

        # execute the refinement model
        refine_gen_kwargs = {"critique_output": critique_output, **critic_gen_kwargs}
        field_answers, reasoning, refine_gen_stats, _ = self.refine_generator(candidate, fields, **refine_gen_kwargs)

        # compute the total generation stats
        generation_stats = original_gen_stats + critic_gen_stats + refine_gen_stats

        return field_answers, generation_stats

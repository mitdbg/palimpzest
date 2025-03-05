from __future__ import annotations

from typing import Any

from palimpzest.constants import MODEL_CARDS, Model, PromptStrategy
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import generator_factory
from palimpzest.query.operators.convert import LLMConvert

# TYPE DEFINITIONS
FieldName = str


class ArchonConvert(LLMConvert):

    def __init__(
        self,
        generator_models: list[Model],
        fuser_model: Model,
        critic_model: Model,
        refine_model: Model,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generator_models = generator_models
        self.fuser_model = fuser_model
        self.critic_model = critic_model
        self.refine_model = refine_model
        
        if self.prompt_strategy == PromptStrategy.COT_QA:
            self.generator_prompt_strategy = PromptStrategy.COT_QA_ARCHON_GENERATOR
            self.fuser_prompt_strategy = PromptStrategy.COT_QA_ARCHON_FUSER
            self.critic_prompt_strategy = PromptStrategy.COT_QA_ARCHON_CRITIC
            self.refinement_prompt_strategy = PromptStrategy.COT_QA_ARCHON_REFINE
        elif self.prompt_strategy == PromptStrategy.COT_QA_IMAGE:
            self.generator_prompt_strategy = PromptStrategy.COT_QA_IMAGE_ARCHON_GENERATOR
            self.fuser_prompt_strategy = PromptStrategy.COT_QA_IMAGE_ARCHON_FUSER
            self.critic_prompt_strategy = PromptStrategy.COT_QA_IMAGE_ARCHON_CRITIC
            self.refinement_prompt_strategy = PromptStrategy.COT_QA_IMAGE_ARCHON_REFINE
        else:
            raise ValueError(f"Unsupported prompt strategy: {self.prompt_strategy}")

        # create generators
        self.generator_generators = [
            generator_factory(model, self.generator_prompt_strategy, self.cardinality, self.verbose)
            for model in generator_models
        ]
        self.fuser_generator = generator_factory(self.fuser_model, self.fuser_prompt_strategy, self.cardinality, self.verbose)
        self.critic_generator = generator_factory(self.critic_model, self.critic_prompt_strategy, self.cardinality, self.verbose)
        self.refine_generator = generator_factory(self.refine_model, self.refinement_prompt_strategy, self.cardinality, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Generator Models: {self.generator_models}\n"
        op += f"    Generator Prompt Strategy: {self.generator_prompt_strategy}\n"
        op += f"    Critic Model: {self.critic_model}\n"
        op += f"    Critic Prompt Strategy: {self.critic_prompt_strategy}\n"
        op += f"    Fuser Model: {self.fuser_model}\n"
        op += f"    Fuser Prompt Strategy: {self.fuser_prompt_strategy}\n"
        op += f"    Refine Model: {self.refine_model}\n"
        op += f"    Refinement Prompt Strategy: {self.refinement_prompt_strategy}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "generator_models": [model.value for model in self.generator_models],
            "critic_model": self.critic_model.value,
            "fuser_model": self.fuser_model.value,
            "refine_model": self.refine_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "generator_model": self.generator_model,
            "critic_model": self.critic_model,
            "fuser_model": self.fuser_model,
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

        #NEED TO EDIT FOR ARCHON
        naive_op_cost_estimates = 3 * super().naive_cost_estimates(source_op_cost_estimates)

        # for naive setting, estimate quality as quality of refine model
        model_quality = MODEL_CARDS[self.refine_model.value]["overall"] / 100.0
        naive_op_cost_estimates.quality = model_quality
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # NOTE: when I merge in the `abacus` branch, I will want to update this to reflect the changes I made to reasoning extraction
        # execute the generator model
        # execute generator models in sequence
        generator_final_answers, generator_stats = [], []
        for generator_generator in self.generator_generators:
            original_gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}
            field_answers, reasoning, original_gen_stats = generator_generator(candidate, fields, **original_gen_kwargs)
            generator_final_answers.append(f"REASONING: {reasoning}\nANSWER:{field_answers}\n")
            generator_stats.append(original_gen_stats)

        formatted_candidates = "\n\n".join(f"[{i+1}] {generator_final_answers[i]}" for i in range(len(generator_final_answers)))

        # execute the critic model
        critic_gen_kwargs = {"original_output": formatted_candidates, **original_gen_kwargs}
        field_answers, reasoning, critic_gen_stats = self.critic_generator(candidate, fields, **critic_gen_kwargs)
        critic_output = f"REASONING: {reasoning}\nANSWER:{field_answers}\n"
        #critic_messages = self.fuser_generator.get_messages()

        #formatted_candidates_with_critiques = "Responses from Models:" + formatted_candidates + "\n + \n + Respective Critiques: \n + \n" + field_answers
        # execute the fuser model
        fuser_gen_kwargs = {"original_output": formatted_candidates, "critique_output": field_answers, **original_gen_kwargs}
        field_answers, reasoning, fuser_gen_stats = self.fuser_generator(candidate, fields, **fuser_gen_kwargs)
        fuser_output = f"REASONING: {reasoning}\nANSWER:{field_answers}\n"

        # execute the refinement model
        #refine_gen_kwargs = {"fuser_output": field_answers, **critic_gen_kwargs}
        #field_answers, reasoning, refine_gen_stats = self.refine_generator(candidate, fields, **refine_gen_kwargs)

        # compute the total generation stats
        generation_stats = sum(generator_stats) + critic_gen_stats + fuser_gen_stats 

        return field_answers, generation_stats

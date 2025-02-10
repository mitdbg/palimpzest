from __future__ import annotations

from typing import Any

from palimpzest.constants import MODEL_CARDS, Model, PromptStrategy
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import generator_factory
from palimpzest.query.operators.convert import LLMConvert

# TYPE DEFINITIONS
FieldName = str


class MixtureOfAgentsConvert(LLMConvert):

    def __init__(
        self,
        proposer_models: list[Model],
        temperatures: list[float],
        aggregator_model: Model,
        proposer_prompt_strategy: PromptStrategy = PromptStrategy.COT_MOA_PROPOSER,
        aggregator_prompt_strategy: PromptStrategy = PromptStrategy.COT_MOA_AGG,
        proposer_prompt: str | None = None,
        *args,
        **kwargs,
    ):
        kwargs["model"] = None
        kwargs["prompt_strategy"] = None
        super().__init__(*args, **kwargs)
        sorted_proposers, sorted_temps = zip(*[(m, t) for m, t in sorted(zip(proposer_models, temperatures), key=lambda pair: pair[0])])
        self.proposer_models = list(sorted_proposers)
        self.temperatures = list(sorted_temps)
        self.aggregator_model = aggregator_model
        self.proposer_prompt_strategy = proposer_prompt_strategy
        self.aggregator_prompt_strategy = aggregator_prompt_strategy
        self.proposer_prompt = proposer_prompt

        # create generators
        self.proposer_generators = [
            generator_factory(model, self.proposer_prompt_strategy, self.cardinality, self.verbose)
            for model in proposer_models
        ]
        self.aggregator_generator = generator_factory(aggregator_model, self.aggregator_prompt_strategy, self.cardinality, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Proposer Models: {self.proposer_models}\n"
        op += f"    Temperatures: {self.temperatures}\n"
        op += f"    Aggregator Model: {self.aggregator_model}\n"
        op += f"    Proposer Prompt Strategy: {self.proposer_prompt_strategy.value}\n"
        op += f"    Aggregator Prompt Strategy: {self.aggregator_prompt_strategy.value}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "proposer_models": [model.value for model in self.proposer_models],
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model.value,
            "proposer_prompt_strategy": self.proposer_prompt_strategy.value,
            "aggregator_prompt_strategy": self.aggregator_prompt_strategy.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "proposer_models": self.proposer_models,
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model,
            "proposer_prompt_strategy": self.proposer_prompt_strategy,
            "aggregator_prompt_strategy": self.aggregator_prompt_strategy,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are using multiple proposer models with different temperatures to synthesize
        answers, which are then aggregated and summarized by a single aggregator model. Thus, we
        roughly expect to incur the cost and time of an LLMConvert * (len(proposer_models) + 1).
        In practice, this naive quality estimate will be overwritten by the CostModel's estimate
        once it executes a few code generated examples.
        """
        # temporarily set self.model so that super().naive_cost_estimates(...) can compute an estimate
        self.model = self.proposer_models[0]

        # get naive cost estimates for single LLM call and scale it by number of LLMs used in MoA
        naive_op_cost_estimates = super().naive_cost_estimates(source_op_cost_estimates)
        naive_op_cost_estimates.time_per_record *= (len(self.proposer_models) + 1)
        naive_op_cost_estimates.time_per_record_lower_bound = naive_op_cost_estimates.time_per_record
        naive_op_cost_estimates.time_per_record_upper_bound = naive_op_cost_estimates.time_per_record
        naive_op_cost_estimates.cost_per_record *= (len(self.proposer_models) + 1)
        naive_op_cost_estimates.cost_per_record_lower_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.cost_per_record_upper_bound = naive_op_cost_estimates.cost_per_record

        # for naive setting, estimate quality as mean of all model qualities
        model_qualities = [
            MODEL_CARDS[model.value]["overall"] / 100.0
            for model in self.proposer_models + [self.aggregator_model]
        ]
        naive_op_cost_estimates.quality = sum(model_qualities)/(len(self.proposer_models) + 1)
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # execute generator models in sequence
        proposer_model_final_answers, proposer_model_generation_stats = [], []
        for proposer_generator, temperature in zip(self.proposer_generators, self.temperatures):
            gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema, "temperature": temperature}
            field_answers, reasoning, generation_stats = proposer_generator(candidate, fields, **gen_kwargs)
            proposer_model_final_answers.append(f"REASONING: {reasoning}\nANSWER:{field_answers}\n")
            proposer_model_generation_stats.append(generation_stats)

        # call the aggregator
        gen_kwargs = {
            "project_cols": input_fields,
            "output_schema": self.output_schema,
            "model_responses": proposer_model_final_answers,
        }
        field_answers, _, aggregator_gen_stats = self.aggregator_generator(candidate, fields, **gen_kwargs)

        # compute the total generation stats
        generation_stats = sum(proposer_model_generation_stats) + aggregator_gen_stats

        return field_answers, generation_stats

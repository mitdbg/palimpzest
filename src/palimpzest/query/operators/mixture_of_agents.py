from __future__ import annotations

from pydantic.fields import FieldInfo

from palimpzest.constants import MODEL_CARDS, Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats, OperatorCostEstimates
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter

# TYPE DEFINITIONS
FieldName = str


class MixtureOfAgentsConvert(LLMConvert):

    def __init__(
        self,
        proposer_models: list[Model],
        temperatures: list[float],
        aggregator_model: Model,
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

        # create generators
        self.proposer_generators = [
            Generator(model, PromptStrategy.MAP_MOA_PROPOSER, self.reasoning_effort, self.api_base, self.cardinality, self.desc, self.verbose)
            for model in proposer_models
        ]
        self.aggregator_generator = Generator(aggregator_model, PromptStrategy.MAP_MOA_AGG, self.reasoning_effort, self.api_base, self.cardinality, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Proposer Models: {self.proposer_models}\n"
        op += f"    Temperatures: {self.temperatures}\n"
        op += f"    Aggregator Model: {self.aggregator_model}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "proposer_models": [model.value for model in self.proposer_models],
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "proposer_models": self.proposer_models,
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are using multiple proposer models with different temperatures to synthesize
        answers, which are then aggregated and summarized by a single aggregator model. Thus, we
        roughly expect to incur the cost and time of an LLMConvert * (len(proposer_models) + 1).
        In practice, this naive quality estimate will be overwritten by the CostModel's estimate
        once it executes a few instances of the operator.
        """
        # temporarily set self.model and self.prompt_strategy so that super().naive_cost_estimates(...) can compute an estimate
        self.model = self.proposer_models[0]
        self.prompt_strategy = PromptStrategy.MAP_MOA_PROPOSER

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

        # reset self.model to be None
        self.model = None
        self.prompt_strategy = None

        return naive_op_cost_estimates

    def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[str, list], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # execute generator models in sequence
        proposer_model_final_answers, proposer_model_generation_stats = [], []
        for proposer_generator, temperature in zip(self.proposer_generators, self.temperatures):
            gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema, "temperature": temperature}
            _, reasoning, generation_stats, _ = proposer_generator(candidate, fields, json_output=False, **gen_kwargs)
            proposer_text = f"REASONING: {reasoning}\n"
            proposer_model_final_answers.append(proposer_text)
            proposer_model_generation_stats.append(generation_stats)

        # call the aggregator
        gen_kwargs = {
            "project_cols": input_fields,
            "output_schema": self.output_schema,
            "model_responses": proposer_model_final_answers,
        }
        field_answers, _, aggregator_gen_stats, _ = self.aggregator_generator(candidate, fields, **gen_kwargs)

        # compute the total generation stats
        generation_stats = sum(proposer_model_generation_stats) + aggregator_gen_stats

        return field_answers, generation_stats


class MixtureOfAgentsFilter(LLMFilter):

    def __init__(
        self,
        proposer_models: list[Model],
        temperatures: list[float],
        aggregator_model: Model,
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

        # create generators
        self.proposer_generators = [
            Generator(model, PromptStrategy.FILTER_MOA_PROPOSER, self.reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)
            for model in proposer_models
        ]
        self.aggregator_generator = Generator(aggregator_model, PromptStrategy.FILTER_MOA_AGG, self.reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Proposer Models: {self.proposer_models}\n"
        op += f"    Temperatures: {self.temperatures}\n"
        op += f"    Aggregator Model: {self.aggregator_model}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "proposer_models": [model.value for model in self.proposer_models],
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "proposer_models": self.proposer_models,
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are using multiple proposer models with different temperatures to synthesize
        answers, which are then aggregated and summarized by a single aggregator model. Thus, we
        roughly expect to incur the cost and time of an LLMFilter * (len(proposer_models) + 1).
        In practice, this naive quality estimate will be overwritten by the CostModel's estimate
        once it executes a few instances of the operator.
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

        # reset self.model to be None
        self.model = None

        return naive_op_cost_estimates

    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        # get input fields
        input_fields = self.get_input_fields()

        # construct output fields
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the record passed the filter operation")}

        # execute generator models in sequence
        proposer_model_final_answers, proposer_model_generation_stats = [], []
        for proposer_generator, temperature in zip(self.proposer_generators, self.temperatures):
            gen_kwargs = {"project_cols": input_fields, "filter_condition": self.filter_obj.filter_condition, "temperature": temperature}
            _, reasoning, generation_stats, _ = proposer_generator(candidate, fields, json_output=False, **gen_kwargs)
            proposer_text = f"REASONING: {reasoning}\n"
            proposer_model_final_answers.append(proposer_text)
            proposer_model_generation_stats.append(generation_stats)

        # call the aggregator
        gen_kwargs = {
            "project_cols": input_fields,
            "filter_condition": self.filter_obj.filter_condition,
            "model_responses": proposer_model_final_answers,
        }
        field_answers, _, aggregator_gen_stats, _ = self.aggregator_generator(candidate, fields, **gen_kwargs)

        # compute the total generation stats
        generation_stats = sum(proposer_model_generation_stats) + aggregator_gen_stats

        return field_answers, generation_stats

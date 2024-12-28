from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from typing import Any

from palimpzest import prompts
from palimpzest.constants import (
    MODEL_CARDS,
    PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS,
    Cardinality,
    Model,
)
from palimpzest.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.generators.generators import DSPyGenerator, ImageTextGenerator
from palimpzest.operators.convert import LLMConvert

# TYPE DEFINITIONS
FieldName = str


class MixtureOfAgentsConvert(LLMConvert):

    def __init__(
        self,
        proposer_models: list[Model],
        temperatures: list[float],
        aggregator_model: Model,
        proposer_prompt: str | None = None,
        *args, **kwargs
    ):
        kwargs["model"] = None
        super().__init__(*args, **kwargs)
        sorted_proposers, sorted_temps = zip(*[(m, t) for m, t in sorted(zip(proposer_models, temperatures), key=lambda pair: pair[0])])
        self.proposer_models = list(sorted_proposers)
        self.temperatures = list(sorted_temps)
        self.aggregator_model = aggregator_model
        self.proposer_prompt = proposer_prompt

        # create generators
        doc_schema = str(self.output_schema)
        doc_type = self.output_schema.class_name()

        self.proposer_generators = []
        for model in proposer_models:
            generator = (
                ImageTextGenerator(model, self.verbose)
                if self.image_conversion
                else DSPyGenerator(model, self.prompt_strategy, doc_schema, doc_type, self.verbose)
            )
            self.proposer_generators.append(generator)

        self.aggregator_generator = DSPyGenerator(aggregator_model, self.prompt_strategy, doc_schema, doc_type, self.verbose)

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
            **op_params
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

    def _dspy_generate_fields(
            self,
            generator: DSPyGenerator | ImageTextGenerator,
            prompt: str,
            content: str | list[str] | None = None,
            temperature: float=0.0,
        ) -> tuple[str | GenerationStats]:
        # generate LLM response and capture statistics
        answer:str
        query_stats:GenerationStats

        try:
            answer, rationale, query_stats = generator.generate(context=content, prompt=prompt, temperature=temperature)
        except Exception as e:
            print(f"DSPy generation error: {e}")
            return "", "", GenerationStats()

        return answer, rationale, query_stats

    def _construct_proposer_prompt(self, fields_to_generate: list[str], model: Model) -> str:
        # set defaults
        doc_type = self.output_schema.class_name()

        # build string of input fields and their descriptions
        multiline_input_field_description = ""
        input_fields = (
            self.input_schema.field_names()
            if not self.depends_on
            else [field.split(".")[-1] for field in self.depends_on]
        )
        for field_name in input_fields:
            field_desc = getattr(self.input_schema, field_name).desc
            multiline_input_field_description += prompts.INPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # build string of output fields and their descriptions
        multiline_output_field_description = ""
        for field_name in fields_to_generate:
            field_desc = getattr(self.output_schema, field_name).desc
            multiline_output_field_description += prompts.OUTPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # add input/output schema descriptions (if they have a docstring)
        optional_input_desc = (
            ""
            if self.input_schema.__doc__ is None
            else prompts.OPTIONAL_INPUT_DESC.format(desc=self.input_schema.__doc__)
        )
        optional_output_desc = (
            ""
            if self.output_schema.__doc__ is None
            else prompts.OPTIONAL_OUTPUT_DESC.format(desc=self.output_schema.__doc__)
        )

        # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
        if self.cardinality == Cardinality.ONE_TO_MANY:
            target_output_descriptor = prompts.MOA_ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            output_single_or_plural = prompts.ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL
        else:
            target_output_descriptor = prompts.MOA_ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            output_single_or_plural = prompts.ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL

        # construct promptQuestion
        # optional_desc = "" if self.desc is None else prompts.OPTIONAL_DESC.format(desc=self.desc)
        model_instruction = prompts.LLAMA_INSTRUCTION if model in [Model.LLAMA3, Model.LLAMA3_V] else ""
        if not self.image_conversion:
            prompt_question = prompts.MOA_STRUCTURED_CONVERT_PROMPT
        else:
            prompt_question = prompts.MOA_IMAGE_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            target_output_descriptor=target_output_descriptor,
            input_type=self.input_schema.class_name(),
            output_single_or_plural=output_single_or_plural,
            optional_input_desc=optional_input_desc,
            optional_output_desc=optional_output_desc,
            multiline_input_field_description=multiline_input_field_description,
            multiline_output_field_description=multiline_output_field_description,
            doc_type=doc_type,
            model_instruction=model_instruction,
        )

        return prompt_question

    def _construct_aggregator_prompt(self, fields_to_generate: list[str]) -> str:
        # set defaults
        doc_type = self.output_schema.class_name()

        # build string of output fields and their descriptions
        multiline_output_field_description = ""
        for field_name in fields_to_generate:
            field_desc = getattr(self.output_schema, field_name).desc
            multiline_output_field_description += prompts.OUTPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # add output schema description (if it has a docstring)
        optional_output_desc = (
            ""
            if self.output_schema.__doc__ is None
            else prompts.OPTIONAL_OUTPUT_DESC.format(desc=self.output_schema.__doc__)
        )

        # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
        if self.cardinality == Cardinality.ONE_TO_MANY:
            target_output_descriptor = prompts.ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            output_single_or_plural = prompts.ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL
            appendix_instruction = prompts.ONE_TO_MANY_APPENDIX_INSTRUCTION.format(fields=fields_to_generate)
        else:
            target_output_descriptor = prompts.ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            output_single_or_plural = prompts.ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL

            fields_example_dict = {}
            for field in fields_to_generate:
                type_str = self.output_schema.json_schema()['properties'][field]['type']
                if type_str == "string":
                    fields_example_dict[field] = "abc"
                elif type_str == "numeric":
                    fields_example_dict[field] = 123
                elif type_str == "boolean":
                    fields_example_dict[field] = True
                elif type_str == "List[string]":
                    fields_example_dict[field] = ["<str>", "<str>", "..."]
                elif type_str == "List[numeric]":
                    fields_example_dict[field] = ["<int | float>", "<int | float>", "..."]
                elif type_str == "List[boolean]":
                    fields_example_dict[field] = ["<bool>", "<bool>", "..."]

            fields_example_dict_str = json.dumps(fields_example_dict, indent=2)
            appendix_instruction = prompts.ONE_TO_ONE_APPENDIX_INSTRUCTION.format(fields=fields_to_generate, fields_example_dict=fields_example_dict_str)

        # construct promptQuestion
        optional_desc = "" if self.desc is None else prompts.OPTIONAL_DESC.format(desc=self.desc)
        model_instruction = prompts.LLAMA_INSTRUCTION if self.aggregator_model in [Model.LLAMA3, Model.LLAMA3_V] else ""
        prompt_question = prompts.MOA_AGGREGATOR_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            target_output_descriptor=target_output_descriptor,
            output_single_or_plural=output_single_or_plural,
            optional_output_desc=optional_output_desc,
            multiline_output_field_description=multiline_output_field_description,
            optional_desc=optional_desc,
            appendix_instruction=appendix_instruction,
            model_instruction=model_instruction,
        )

        return prompt_question

    def _call_proposer(self, proposer_generator, temperature, fields, candidate):
        # TODO: maybe we accept general prompts; but this should also work for default prompts;
        # ask the model to output natural language instead of JSON, and justify its response with
        # citations from the Context
        # get the proposer prompt (this formats the Question in the DSPy/ImageTextGenerator)
        proposer_model = proposer_generator.model
        proposer_prompt = self._construct_proposer_prompt(fields_to_generate=fields, model=proposer_model)
        proposer_candidate_content = self._get_candidate_content(proposer_model, candidate)
        answer, rationale, generation_stats = self._dspy_generate_fields(
            generator=proposer_generator,
            prompt=proposer_prompt,
            content=proposer_candidate_content,
            temperature=temperature,
        )
        final_answer = f"Reasoning: Let's think step by step in order to {rationale}\n\nAnswer: {answer}"

        return final_answer, generation_stats

    def _call_aggregator(self, proposer_model_answers: list[str], fields: list[str]) -> str:
        # TODO: one issue here is that the model does not see the input fields, which are specified
        # in the standard self._construct_query_prompt, so this may cause confusion and it would
        # be better to modify the aggregator_prompt to only specify the expected output; e.g. add aggregator=True to the call below
        aggregator_prompt = self._construct_aggregator_prompt(fields_to_generate=fields)

        # format proposer_model_answers for DSPyCOT prompt
        responses = "\n"
        for idx, proposer_model_answer in enumerate(proposer_model_answers):
            responses += f"Model {idx + 1} Response:\n{proposer_model_answer}\n\n"

        # remove final \n
        responses = responses[:-1]

        answer, _, generation_stats = self._dspy_generate_fields(
            generator=self.aggregator_generator,
            prompt=aggregator_prompt,
            content=responses,
        )

        return answer, generation_stats

    def convert(self, candidate, fields) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # call proposers asynchronously in parallel
        proposer_model_answers, proposer_model_stats = [], []
        with ThreadPoolExecutor(max_workers=len(self.proposer_generators)) as executor:
            futures = []
            for proposer_generator, temperature in zip(self.proposer_generators, self.temperatures):
                generate_answer = partial(self._call_proposer, proposer_generator, temperature, fields, candidate)
                futures.append(executor.submit(generate_answer))

            # block until all futures have finished
            done_futures, not_done_futures = [], futures
            while len(not_done_futures) > 0:
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

            # get outputs
            proposer_model_answers, proposer_model_stats = zip(*[future.result() for future in done_futures])

        # call the aggregator
        final_answer, aggregator_gen_stats = self._call_aggregator(proposer_model_answers, fields)

        # compute the total generation stats
        generation_stats = sum(proposer_model_stats) + aggregator_gen_stats

        # parse the final answer
        json_answers = self.parse_answer(final_answer, fields, self.aggregator_model)

        return json_answers, generation_stats

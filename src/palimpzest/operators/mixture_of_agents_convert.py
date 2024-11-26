from __future__ import annotations

from palimpzest import prompts
from palimpzest.constants import (
    Cardinality,
    Model,
    MODEL_CARDS,
    PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS,
    PromptStrategy,
)
from palimpzest.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.generators import DSPyGenerator, ImageTextGenerator
from palimpzest.operators import LLMConvert

from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from typing import Any

import json

# TYPE DEFINITIONS
FieldName = str

# TODO: Create ImplementationRule for this
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
        self.proposer_models = proposer_models
        self.temperatures = temperatures
        self.aggregator_model = aggregator_model
        self.proposer_prompt = proposer_prompt

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.proposer_models == other.proposer_models
            and self.temperatures == other.temperatures
            and self.aggregator_model == other.aggregator_model
            and self.cardinality == other.cardinality
            and self.prompt_strategy == other.prompt_strategy
            and self.outputSchema == other.outputSchema
            and self.max_workers == other.max_workers
        )

    def __str__(self):
        op = super().__str__()
        op += f"    Proposer Models: {self.proposer_models}\n"
        op += f"    Temperatures: {self.temperatures}\n"
        op += f"    Aggregator Model: {self.aggregator_model}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {
            "proposer_models": self.proposer_models,
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model,
            **copy_kwargs
        }

    def get_op_params(self):
        """
        NOTE: we do not include self.cache_across_plans because (for now) get_op_params()
        is only supposed to return hyperparameters which affect operator performance.
        """
        op_params = super().get_op_params()
        op_params = {
            "proposer_models": self.proposer_models,
            "temperatures": self.temperatures,
            "aggregator_model": self.aggregator_model,
            **op_params,
        }

        return op_params

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently, we are using multiple proposer models with different temperatures to synthesize
        answers, which are then aggregated and summarized by a single aggregator model. Thus, we
        roughly expect to incur the cost and time of an LLMConvert * (len(proposer_models) + 1).
        In practice, this naive quality estimate will be overwritten by the CostModel's estimate
        once it executes a few code generated examples.
        """
        # temporarily set self.model so that super().naiveCostEstimates(...) can compute an estimate
        self.model = self.proposer_models[0]

        # get naive cost estimates for single LLM call and scale it by number of LLMs used in MoA
        naive_op_cost_estimates = super().naiveCostEstimates(source_op_cost_estimates)
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
            model: Model,
            prompt: str,
            content: str | list[str] | None = None,
            prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
            temperature: float=0.0,
            verbose: bool = False,
        ) -> tuple[str | GenerationStats]:
        # create DSPy generator and generate
        doc_schema = str(self.outputSchema)
        doc_type = self.outputSchema.className()

        # generate LLM response and capture statistics
        answer:str
        query_stats:GenerationStats

        if self.image_conversion:
            generator = ImageTextGenerator(model.value, verbose)
        else:
            generator = DSPyGenerator(
                model.value, prompt_strategy, doc_schema, doc_type, verbose
            )

        try:
            answer, query_stats = generator.generate(context=content, question=prompt, temperature=temperature)
        except Exception as e:
            print(f"DSPy generation error: {e}")
            return "", GenerationStats()

        return answer, query_stats

    def _construct_proposer_prompt(self, fields_to_generate: list[str]) -> str:
        # set defaults
        doc_type = self.outputSchema.className()

        # build string of input fields and their descriptions
        multilineInputFieldDescription = ""
        input_fields = (
            self.inputSchema.fieldNames()
            if not self.depends_on
            else [field.split(".")[-1] for field in self.depends_on]
        )
        for field_name in input_fields:
            field_desc = getattr(self.inputSchema, field_name).desc
            multilineInputFieldDescription += prompts.INPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # build string of output fields and their descriptions
        multilineOutputFieldDescription = ""
        for field_name in fields_to_generate:
            field_desc = getattr(self.outputSchema, field_name).desc
            multilineOutputFieldDescription += prompts.OUTPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # add input/output schema descriptions (if they have a docstring)
        optionalInputDesc = (
            ""
            if self.inputSchema.__doc__ is None
            else prompts.OPTIONAL_INPUT_DESC.format(desc=self.inputSchema.__doc__)
        )
        optionalOutputDesc = (
            ""
            if self.outputSchema.__doc__ is None
            else prompts.OPTIONAL_OUTPUT_DESC.format(desc=self.outputSchema.__doc__)
        )

        # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
        if self.cardinality == Cardinality.ONE_TO_MANY:
            targetOutputDescriptor = prompts.MOA_ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL
        else:
            targetOutputDescriptor = prompts.MOA_ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL

        # construct promptQuestion
        optional_desc = "" if self.desc is None else prompts.OPTIONAL_DESC.format(desc=self.desc)
        if not self.image_conversion:
            prompt_question = prompts.MOA_STRUCTURED_CONVERT_PROMPT
        else:
            prompt_question = prompts.MOA_IMAGE_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            targetOutputDescriptor=targetOutputDescriptor,
            input_type=self.inputSchema.className(),
            outputSingleOrPlural=outputSingleOrPlural,
            optionalInputDesc=optionalInputDesc,
            optionalOutputDesc=optionalOutputDesc,
            multilineInputFieldDescription=multilineInputFieldDescription,
            multilineOutputFieldDescription=multilineOutputFieldDescription,
            doc_type=doc_type,
        )

        return prompt_question

    def _construct_aggregator_prompt(self, fields_to_generate: list[str]) -> str:
        # set defaults
        doc_type = self.outputSchema.className()

        # build string of output fields and their descriptions
        multilineOutputFieldDescription = ""
        for field_name in fields_to_generate:
            field_desc = getattr(self.outputSchema, field_name).desc
            multilineOutputFieldDescription += prompts.OUTPUT_FIELD.format(
                field_name=field_name, field_desc=field_desc
            )

        # add output schema description (if it has a docstring)
        optionalOutputDesc = (
            ""
            if self.outputSchema.__doc__ is None
            else prompts.OPTIONAL_OUTPUT_DESC.format(desc=self.outputSchema.__doc__)
        )

        # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
        if self.cardinality == Cardinality.ONE_TO_MANY:
            targetOutputDescriptor = prompts.ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL
            appendixInstruction = prompts.ONE_TO_MANY_APPENDIX_INSTRUCTION.format(fields=fields_to_generate)
        else:
            targetOutputDescriptor = prompts.ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL

            fields_example_dict = {}
            for field in fields_to_generate:
                type_str = self.outputSchema.jsonSchema()['properties'][field]['type']
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
            appendixInstruction = prompts.ONE_TO_ONE_APPENDIX_INSTRUCTION.format(fields=fields_to_generate, fields_example_dict=fields_example_dict_str)

        # construct promptQuestion
        optional_desc = "" if self.desc is None else prompts.OPTIONAL_DESC.format(desc=self.desc)
        prompt_question = prompts.MOA_AGGREGATOR_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            targetOutputDescriptor=targetOutputDescriptor,
            outputSingleOrPlural=outputSingleOrPlural,
            optionalOutputDesc=optionalOutputDesc,
            multilineOutputFieldDescription=multilineOutputFieldDescription,
            optional_desc=optional_desc,
            appendixInstruction=appendixInstruction,
        )

        return prompt_question

    def _call_proposer(self, proposer_model, temperature, proposer_prompt, candidate, verbose):
        proposer_candidate_content = self._get_candidate_content(proposer_model, candidate)
        answer, generation_stats = self._dspy_generate_fields(
            model=proposer_model,
            prompt=proposer_prompt,
            content=proposer_candidate_content,
            temperature=temperature,
            verbose=verbose,
        )
        return answer, generation_stats

    def _call_aggregator(self, aggregator_prompt: str, proposer_model_answers: list[str]) -> str:
        # format proposer_model_answers for DSPyCOT prompt
        responses = "\n"
        for idx, proposer_model_answer in enumerate(proposer_model_answers):
            responses += f"Model {idx + 1} Response:\n{proposer_model_answer}\n\n"

        # remove final \n
        responses = responses[:-1]

        answer, generation_stats = self._dspy_generate_fields(
            model=self.aggregator_model,
            prompt=aggregator_prompt,
            content=responses,
            prompt_strategy=PromptStrategy.DSPY_COT_MOA_AGG,
            verbose=self.verbose,
        )

        return answer, generation_stats

    def convert(self, candidate, fields) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # TODO: maybe we accept general prompts; but this should also work for default prompts;
        # ask the model to output natural language instead of JSON, and justify its response with
        # citations from the Context
        # get the proposer prompt (this formats the Question in the DSPy/ImageTextGenerator)
        proposer_prompt = self._construct_proposer_prompt(fields_to_generate=fields)

        # proposer_prompt = (
        #     self._construct_query_prompt(fields_to_generate=fields)
        #     if self.proposer_prompt is None
        #     else self.proposer_prompt
        # )
        # TODO: one issue here is that the model does not see the input fields, which are specified
        # in the standard self._construct_query_prompt, so this may cause confusion and it would
        # be better to modify the aggregator_prompt to only specify the expected output; e.g. add aggregator=True to the call below
        aggregator_prompt = self._construct_aggregator_prompt(fields_to_generate=fields)
        
        # aggregator_prompt = self._construct_query_prompt(fields_to_generate=fields)

        # call proposers asynchronously in parallel
        proposer_model_answers, proposer_model_stats = [], []
        # for proposer_model, temperature in zip(self.proposer_models, self.temperatures):
        #     answer, stats = self._call_proposer(proposer_model, temperature, proposer_prompt, candidate, self.verbose)
        #     proposer_model_answers.append(answer)
        #     proposer_model_stats.append(stats)
        with ThreadPoolExecutor(max_workers=len(self.proposer_models)) as executor:
            futures = []
            for proposer_model, temperature in zip(self.proposer_models, self.temperatures):
                generate_answer = partial(self._call_proposer, proposer_model, temperature, proposer_prompt, candidate, self.verbose)
                futures.append(executor.submit(generate_answer))

            # block until all futures have finished
            done_futures, not_done_futures = [], futures
            while len(not_done_futures) > 0:
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

            # get outputs
            proposer_model_answers, proposer_model_stats = zip(*[future.result() for future in done_futures])

        # call the aggregator model
        final_answer, aggregator_gen_stats = self._call_aggregator(aggregator_prompt, proposer_model_answers)

        # compute the total generation stats
        generation_stats = sum(proposer_model_stats) + aggregator_gen_stats

        # parse the final answer
        json_answers = self.parse_answer(final_answer, fields)

        return json_answers, generation_stats

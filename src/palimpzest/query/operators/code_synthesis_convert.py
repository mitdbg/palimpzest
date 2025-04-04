from __future__ import annotations

from typing import Any

from palimpzest.constants import Cardinality, GPT_4o_MODEL_CARD, Model
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.prompts import ADVICEGEN_PROMPT, CODEGEN_PROMPT, EXAMPLE_PROMPT
from palimpzest.query.generators.generators import code_ensemble_execution, generator_factory
from palimpzest.query.operators.convert import LLMConvert, LLMConvertBonded
from palimpzest.utils.sandbox import API

# TYPE DEFINITIONS
FieldName = str
CodeName = str
Code = str
DataRecordDict = dict[str, Any]
Exemplar = tuple[DataRecordDict, DataRecordDict]
CodeEnsemble = dict[CodeName, Code]


class CodeSynthesisConvert(LLMConvert):
    def __init__(
        self,
        exemplar_generation_model: Model = Model.GPT_4o,
        code_synth_model: Model = Model.GPT_4o,
        fallback_model: Model = Model.GPT_4o_MINI,
        *args,
        **kwargs,
    ):
        kwargs["model"] = None
        super().__init__(*args, **kwargs)

        # set models
        self.exemplar_generation_model = exemplar_generation_model
        self.code_synth_model = code_synth_model
        self.fallback_model = fallback_model

        # initialize parameters
        self.field_to_code_ensemble = None
        self.exemplars = []
        self.code_synthesized = False
        self.code_champion_generator = generator_factory(
            model=self.code_synth_model,
            prompt_strategy=self.prompt_strategy,
            cardinality=Cardinality.ONE_TO_ONE,
            verbose=self.verbose,
        )
        self.field_to_code_ensemble = {}

    def __str__(self):
        op = super().__str__()
        op += f"    Code Synth Strategy: {self.__class__.__name__}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "exemplar_generation_model": self.exemplar_generation_model.value,
            "code_synth_model": self.code_synth_model.value,
            "fallback_model": self.fallback_model.value,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "exemplar_generation_model": self.exemplar_generation_model,
            "code_synth_model": self.code_synth_model,
            "fallback_model": self.fallback_model,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Currently we are using GPT-4 to generate code which we can then invoke on subsequent
        inputs to this operator. To reflect this in our naive cost estimates, we assume that
        the time_per_record is low (about the time it takes to execute a cheap python function)
        and that the cost_per_record is also low (we amortize the cost of code generation across
        all records). For our quality estimate, we naively assume some degredation in quality.
        In practice, this naive quality estimate will be overwritten by the CostModel's estimate
        once it executes a few code generated examples.
        """
        naive_op_cost_estimates = super().naive_cost_estimates(source_op_cost_estimates)
        naive_op_cost_estimates.time_per_record = 1e-5
        naive_op_cost_estimates.time_per_record_lower_bound = 1e-5
        naive_op_cost_estimates.time_per_record_upper_bound = 1e-5
        naive_op_cost_estimates.cost_per_record = 1e-6  # amortize code synth cost across records
        naive_op_cost_estimates.cost_per_record_lower_bound = 1e-6
        naive_op_cost_estimates.cost_per_record_upper_bound = 1e-6
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * (GPT_4o_MODEL_CARD["code"] / 100.0) * 0.7
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def _should_synthesize(
        self, exemplars: list[Exemplar], num_exemplars: int = 1, code_regenerate_frequency: int = 200, *args, **kwargs
    ) -> bool:
        """This function determines whether code synthesis should be performed based on the strategy and the number of exemplars available."""
        raise NotImplementedError("This method should be implemented in a subclass")

    def _synthesize_field_code(
        self,
        candidate: DataRecord,
        api: API,
        output_field_name: str,
        code_ensemble_num: int = 1,  # if strategy != SINGLE
        num_exemplars: int = 1,  # if strategy != EXAMPLE_ENSEMBLE
    ) -> tuple[dict[CodeName, Code], GenerationStats]:
        """This method is responsible for synthesizing the code on a per-field basis.
        Wrapping different calls to the LLM and returning a set of per-field query statistics.
        The format of the code ensemble dictionary is {code_name: code} where code_name is a string and code is a string representing the code.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def synthesize_code_ensemble(self, fields_to_generate, candidate: DataRecord, *args, **kwargs):
        """This function is a wrapper around specific code synthesis methods
        that wraps the synthesized code per-field in a dictionary and returns the stats object.
        """
        # synthesize the per-field code ensembles
        field_to_code_ensemble = {}
        generation_stats = GenerationStats()
        for field_name in fields_to_generate:
            api = API.from_input_output_schemas(
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                field_name=field_name,
                input_fields=candidate.get_field_names(),
            )

            # TODO here _synthesize_code should be called with the right parameters per-code-strategy?!
            code_ensemble, code_synth_stats = self._synthesize_field_code(candidate, api, field_name)

            # update mapping from fields to code ensemble and generation stats
            field_to_code_ensemble[field_name] = code_ensemble
            generation_stats += code_synth_stats

            if self.verbose:
                for code_name, code in code_ensemble.items():
                    print(f"CODE NAME: {code_name}")
                    print("-----------------------")
                    print(code)

        # set field_to_code_ensemble and code_synthesized to True
        return field_to_code_ensemble, generation_stats

    def _bonded_query_fallback(
        self, candidate: DataRecord
    ) -> tuple[dict[FieldName, list[Any] | None], GenerationStats]:
        fields_to_generate = self.get_fields_to_generate(candidate)
        projected_candidate = candidate.copy(include_bytes=False, project_cols=self.depends_on)

        # execute the bonded convert
        bonded_op = LLMConvertBonded(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            model=self.exemplar_generation_model,
            prompt_strategy=self.prompt_strategy,
        )
        field_answers, generation_stats = bonded_op.convert(projected_candidate, fields_to_generate)
        assert all([field in field_answers for field in fields_to_generate]), "Not all fields were generated!"

        # for the vanilla LLMConvert, we simply replace any None values with an empty list
        field_answers = {field: [] if answers is None else answers for field, answers in field_answers.items()}

        # transform the mapping from fields to answers into a (list of) DataRecord(s)
        drs, _ = self._create_data_records_from_field_answers(field_answers, candidate)

        # NOTE: this now includes bytes input fields which will show up as: `field_name = "<bytes>"`;
        #       keep an eye out for a regression in code synth performance and revert if necessary
        # update operator's set of exemplars
        exemplars = [(projected_candidate.to_dict(include_bytes=False), dr.to_dict(include_bytes=False)) for dr in drs]
        self.exemplars.extend(exemplars)

        return field_answers, generation_stats

    def is_image_conversion(self):
        """Code synthesis is disallowed on image conversions, so this must be False."""
        return False

    def convert(
        self, candidate: DataRecord, fields: list[str] | None = None
    ) -> tuple[dict[FieldName, list[Any] | None], GenerationStats]:
        # get the dictionary fields for the candidate
        candidate_dict = candidate.to_dict(include_bytes=False, project_cols=self.depends_on)

        # Check if code was already synthesized, or if we have at least one converted sample
        generation_stats = GenerationStats()
        if self._should_synthesize():
            self.field_to_code_ensemble, total_code_synth_stats = self.synthesize_code_ensemble(fields, candidate)
            self.code_synthesized = True
            generation_stats += total_code_synth_stats

        # if we have yet to synthesize code (perhaps b/c we are waiting for more exemplars),
        # use the exemplar generation model to perform the convert (and generate high-quality
        # exemplars) using a bonded query
        if not len(self.field_to_code_ensemble):
            return self._bonded_query_fallback(candidate)

        # if we have synthesized code run it on each field
        field_answers = {}
        for field_name in fields:
            # create api instance for executing python code
            api = API.from_input_output_schemas(
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                field_name=field_name,
                input_fields=candidate_dict.keys(),
            )
            code_ensemble = self.field_to_code_ensemble[field_name]

            # execute the code ensemble to get the answer
            answer, _, exec_stats = code_ensemble_execution(api, code_ensemble, candidate_dict)

            # if the answer is not None, update the field_answers
            # NOTE: the answer will not be a list because code synth. is disallowed for one-to-many converts
            if answer is not None:
                generation_stats += exec_stats
                field_answers[field_name] = [answer]

            else:
                # if there is a failure, run a conventional llm convert query for the field
                if self.verbose:
                    print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")

                # execute the conventional llm convert
                convert_op = LLMConvertBonded(
                    input_schema=self.input_schema,
                    output_schema=self.output_schema,
                    model=self.fallback_model,
                    prompt_strategy=self.prompt_strategy,
                )
                single_field_answers, single_field_stats = convert_op.convert(candidate, [field_name])

                # include code execution time in single_field_stats
                single_field_stats.fn_call_duration_secs += exec_stats.fn_call_duration_secs

                # update generation_stats
                generation_stats += single_field_stats

                # update field answers
                # NOTE: because code synth. is disallowed for one-to-many queries, we make the first answer a singleton
                field_answers[field_name] = (
                    [single_field_answers[field_name][0]]
                    if single_field_answers[field_name] is not None and len(single_field_answers[field_name]) > 0
                    else []
                )

        assert all([field in field_answers for field in fields]), "Not all fields were generated!"

        # for the vanilla LLMConvert, we simply replace any None values with an empty list
        field_answers = {field: [] if answers is None else answers for field, answers in field_answers.items()}

        return field_answers, generation_stats


class CodeSynthesisConvertNone(CodeSynthesisConvert):
    def _should_synthesize(self, *args, **kwargs):
        return False

    def _synthesize_field_code(self, candidate: DataRecord, api: API, *args, **kwargs):
        code = api.api_def() + "  return None\n"
        code_ensemble = {"{api.name}_v0": code}
        return code_ensemble, GenerationStats()


class CodeSynthesisConvertSingle(CodeSynthesisConvert):
    def _should_synthesize(self, num_exemplars: int = 1, *args, **kwargs) -> bool:
        """This function determines whether code synthesis
        should be performed based on the strategy and the number of exemplars available."""
        if len(self.exemplars) < num_exemplars:
            return False
        return not self.code_synthesized

    def _code_synth_single(
        self,
        candidate: DataRecord,
        api: API,
        output_field_name: str,
        exemplars: list[Exemplar] | None = None,
        advice: str | None = None,
        language="Python",
    ):
        if exemplars is None:
            exemplars = []

        context = {
            "language": language,
            "api": api.args_call(),
            "output": api.output,
            "inputs_desc": "\n".join(
                [f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]
            ),
            "output_desc": api.output_desc,
            "examples_desc": "\n".join(
                [
                    EXAMPLE_PROMPT.format(
                        idx=f" {i}",
                        example_inputs="\n".join(
                            [f"- {field_name} = {repr(example[0][field_name])}" for field_name in example[0]]
                        ),
                        example_output=f"{example[1][output_field_name]}",
                    )
                    for i, example in enumerate(exemplars)
                ]
            ),
            "advice": f"Hint: {advice}" if advice else "",
        }

        prompt = CODEGEN_PROMPT.format(**context)
        if self.verbose:
            print("PROMPT")
            print("-------")
            print(f"{prompt}")

        # set prompt for generator
        gen_kwargs = {"prompt": prompt, "parse_answer": lambda text: text.split("answer:")[-1].split("---")[0].strip()}

        # invoke the champion model to generate the code
        pred, _, stats, _ = self.code_champion_generator(candidate, None, json_output=False, **gen_kwargs)
        ordered_keys = [f"```{language}", f"```{language.lower()}", "```"]
        code = None
        if not pred:
            return code, stats

        for key in ordered_keys:
            if key in pred:
                code = pred.split(key)[1].split("```")[0].strip()
                break

        if self.verbose:
            print("-------")
            print("SYNTHESIZED CODE")
            print("---------------")
            print(f"{code}")

        return code, stats

    def _synthesize_field_code(
        self, candidate: DataRecord, api: API, output_field_name: str, num_exemplars: int = 1, *args, **kwargs
    ):
        code, generation_stats = self._code_synth_single(
            candidate, api, output_field_name, exemplars=self.exemplars[:num_exemplars]
        )
        code_ensemble = {f"{api.name}_v0": code}
        return code_ensemble, generation_stats


# NOTE A nicer truly class based approach would re-implement the code_synth_single method with calls to
# __super__ and then only re-implement the differences instead of having the code in the superclass know
# about the subclass-specific parameters (i.e., advice).
class CodeSynthesisConvertExampleEnsemble(CodeSynthesisConvertSingle):
    def _should_synthesize(self, num_exemplars: int = 1, *args, **kwargs) -> bool:
        if len(self.exemplars) < num_exemplars:
            return False
        return not self.code_synthesized

    def _synthesize_field_code(
        self, candidate: DataRecord, api: API, output_field_name: str, code_ensemble_num: int = 1, *args, **kwargs
    ):
        # creates an ensemble of `code_ensemble_num` synthesized functions; each of
        # which uses a different exemplar (modulo the # of exemplars) for its synthesis
        code_ensemble = {}
        generation_stats = GenerationStats()
        for i in range(code_ensemble_num):
            code_name = f"{api.name}_v{i}"
            exemplar = self.exemplars[i % len(self.exemplars)]
            code, stats = self._code_synth_single(candidate, api, output_field_name, exemplars=[exemplar])
            code_ensemble[code_name] = code
            generation_stats += stats

        return code_ensemble, generation_stats


class CodeSynthesisConvertAdviceEnsemble(CodeSynthesisConvertSingle):
    def _should_synthesize(self, *args, **kwargs):
        return False

    def _parse_multiple_outputs(self, text, outputs=None):
        if outputs is None:
            outputs = ["Thought", "Action"]
        data = {}
        for key in reversed(outputs):
            if key + ":" in text:
                remain, value = text.rsplit(key + ":", 1)
                data[key.lower()] = value.strip()
                text = remain
            else:
                data[key.lower()] = None
        return data

    def _synthesize_advice(
        self,
        candidate: DataRecord,
        api: API,
        output_field_name: str,
        exemplars: list[Exemplar] | None = None,
        language="Python",
        n_advices=4,
        limit: int = 3,
    ):
        if exemplars is None:
            exemplars = []
        context = {
            "language": language,
            "api": api.args_call(),
            "output": api.output,
            "inputs_desc": "\n".join(
                [f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]
            ),
            "output_desc": api.output_desc,
            "examples_desc": "\n".join(
                [
                    EXAMPLE_PROMPT.format(
                        idx=f" {i}",
                        example_inputs="\n".join(
                            [f"- {field_name} = {repr(example[0][field_name])}" for field_name in example[0]]
                        ),
                        example_output=f"{example[1][output_field_name]}",
                    )
                    for i, example in enumerate(exemplars)
                ]
            ),
            "n": n_advices,
        }
        prompt = ADVICEGEN_PROMPT.format(**context)

        # set prompt for generator
        gen_kwargs = {"prompt": prompt, "parse_answer": lambda text: text.split("answer:")[-1].split("---")[0].strip()}

        pred, _, stats, _ = self.code_champion_generator(candidate, None, json_output=False, **gen_kwargs)
        advs = self._parse_multiple_outputs(pred, outputs=[f"Idea {i}" for i in range(1, limit + 1)])

        return advs, stats

    def _synthesize_field_code(
        self,
        candidate: DataRecord,
        api: API,
        output_field_name: str,
        code_ensemble_num: int = 1,
        num_exemplars: int = 1,
        *args,
        **kwargs,
    ):
        # a more advanced approach in which advice is first solicited, and then
        # provided as context when synthesizing the code ensemble
        output_stats = {}
        # solicit advice
        advices, adv_stats = self._synthesize_advice(
            candidate, api, output_field_name, exemplars=self.exemplars[:num_exemplars], n_advices=code_ensemble_num
        )
        for key, value in adv_stats.items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    output_stats[k2] = output_stats.get(k2, 0) + v2
            else:
                output_stats[key] += output_stats.get(key, type(value)()) + value

        code_ensemble = {}
        # synthesize code ensemble
        for i, adv in enumerate(advices):
            code_name = f"{api.name}_v{i}"
            code, stats = self._code_synth_single(
                candidate, api, output_field_name, exemplars=self.exemplars[:num_exemplars], advice=adv
            )
            code_ensemble[code_name] = code
            for key in output_stats:
                output_stats[key] += stats[key]
        return code_ensemble, output_stats


class CodeSynthesisConvertAdviceEnsembleValidation(CodeSynthesisConvert):
    def _should_synthesize(self, code_regenerate_frequency: int = 200, *args, **kwargs):
        return len(self.exemplars) % code_regenerate_frequency == 0

    def _synthesize_field_code(
        self, api: API, output_field_name: str, exemplars: list[Exemplar] = None, *args, **kwargs
    ):
        # TODO this was not implemented ?
        if exemplars is None:
            exemplars = []
        raise Exception("not implemented yet")

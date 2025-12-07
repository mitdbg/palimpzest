from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

from pydantic.fields import FieldInfo

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    NAIVE_EST_ONE_TO_MANY_SELECTIVITY,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator


class ConvertOp(PhysicalOperator, ABC):
    def __init__(
        self,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        udf: Callable | None = None,
        desc: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cardinality = cardinality
        self.udf = udf
        self.desc = desc

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "cardinality": self.cardinality.value,
            "udf": self.udf,
            "desc": self.desc,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "cardinality": self.cardinality,
            "udf": self.udf,
            "desc": self.desc,
            **op_params,
        }

        return op_params

    def _create_data_records_from_field_answers(
        self,
        field_answers: dict[str, list],
        candidate: DataRecord,
    ) -> list[DataRecord]:
        """
        Given a mapping from each field to its (list of) generated value(s), we construct the corresponding
        list of output DataRecords.
        """
        # get the number of records generated; for some convert operations it is possible for fields to
        # have different lengths of generated values, so we take the maximum length of any field's values
        # to be the number of records generated
        n_records = max([len(lst) for lst in field_answers.values()])
        successful_convert = n_records > 0

        drs = []
        for idx in range(max(n_records, 1)):
            # parse newly generated fields from the field_answers dictionary for this field; if the list
            # of generated values is shorter than the number of records, we fill in with None
            data_item = {}
            for field in self.generated_fields:
                data_item[field] = field_answers[field][idx] if idx < len(field_answers[field]) else None

            # initialize record with the correct output schema, data_item, parent record, and cardinality idx
            dr = DataRecord.from_parent(self.output_schema, data_item, parent_record=candidate, cardinality_idx=idx)

            # append data record to list of output data records
            drs.append(dr)

        return drs, successful_convert

    def _create_record_set(
        self,
        records: list[DataRecord],
        field_names: list[str],
        generation_stats: GenerationStats,
        total_time: float,
        successful_convert: bool,
    ) -> DataRecordSet:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        # amortize the generation stats across all generated records
        per_record_stats = generation_stats / len(records)
        time_per_record = total_time / len(records)

        # create the RecordOpStats objects for each output record
        record_op_stats_lst = [
            RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=time_per_record,
                cost_per_record=per_record_stats.cost_per_record,
                model_name=self.get_model_name(),
                answer={field_name: getattr(dr, field_name, None) for field_name in field_names},
                input_fields=list(self.input_schema.model_fields),
                generated_fields=field_names,
                total_input_tokens=per_record_stats.total_input_tokens,
                total_output_tokens=per_record_stats.total_output_tokens,
                total_embedding_input_tokens=per_record_stats.total_embedding_input_tokens,
                total_input_cost=per_record_stats.total_input_cost,
                total_output_cost=per_record_stats.total_output_cost,
                total_embedding_cost=per_record_stats.total_embedding_cost,
                llm_call_duration_secs=per_record_stats.llm_call_duration_secs,
                fn_call_duration_secs=per_record_stats.fn_call_duration_secs,
                total_llm_calls=per_record_stats.total_llm_calls,
                total_embedding_llm_calls=per_record_stats.total_embedding_llm_calls,
                failed_convert=(not successful_convert),
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            for dr in records
        ]

        # create and return the DataRecordSet
        return DataRecordSet(records, record_op_stats_lst)

    @abstractmethod
    def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[str, list], GenerationStats]:
        """
        This abstract method will be implemented by subclasses of ConvertOp to process the input DataRecord
        and generate the value(s) for each of the specified fields. If the convert operator is a one-to-many
        convert, then each field will have a corresponding list of output values. The dictionary mapping each
        generated field to its (list of) value(s) is returned along with the GenerationStats object.

        For example, if the input DataRecord (i.e. `candidate`) contains the contents of a scientific paper,
        and the convert operation is supposed to extract the name and affiliation of each author into its own
        DataRecord, then the output could be:

        ({"author": ["Jane Smith", "John Doe"], "affiliation": ["MIT", "Stanford University"]}, GenerationStats(...))

        Even if the convert operation is a one-to-one convert (i.e. it always generates one output DataRecord
        for each input DataRecord), the output should still map each field to a singleton list containing its value.

        A post-condition of this method is that every field in `fields` must be present in the output dictionary.
        If there is an error in generating a field, then the value for that field must be None.
        """
        pass

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        """
        This method converts an input DataRecord into an output DataRecordSet. The output DataRecordSet contains the
        DataRecord(s) output by the operator's convert() method and their corresponding RecordOpStats objects.
        Some subclasses may override this __call__method to implement their own custom logic.
        """
        start_time = time.time()

        # get fields to generate with this convert
        fields_to_generate = self.get_fields_to_generate(candidate)

        # execute the convert
        field_answers: dict[str, list]
        fields = {field: field_type for field, field_type in self.output_schema.model_fields.items() if field in fields_to_generate}
        field_answers, generation_stats = self.convert(candidate=candidate, fields=fields)
        assert all([field in field_answers for field in fields_to_generate]), "Not all fields were generated!"

        # replace any None values with an empty list; subclasses may override __call__ to change this behavior
        field_answers = {field: [] if answers is None else answers for field, answers in field_answers.items()}

        # transform the mapping from fields to answers into a (list of) DataRecord(s)
        drs, successful_convert = self._create_data_records_from_field_answers(field_answers, candidate)

        # construct and return DataRecordSet
        record_set = self._create_record_set(
            records=drs,
            field_names=fields_to_generate,
            generation_stats=generation_stats,
            total_time=time.time() - start_time,
            successful_convert=successful_convert,
        )

        return record_set


class NonLLMConvert(ConvertOp):
    def __str__(self):
        op = super().__str__()
        op += f"    UDF: {self.udf.__name__}\n"
        return op

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the NonLLMConvert operation. These estimates assume
        that the UDF convert (1) has no cost and (2) has perfect quality.
        """
        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality == Cardinality.ONE_TO_ONE else NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate 1 ms single-threaded execution for udf function
        time_per_record = 0.001

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[str, list], GenerationStats]:
        # apply UDF to input record
        start_time = time.time()
        field_answers = {}
        try:
            # execute the UDF function
            answer = self.udf(candidate.to_dict())

            if self.cardinality == Cardinality.ONE_TO_ONE:
                # answer should be a dictionary
                assert isinstance(answer, dict), (
                    "UDF must return a dictionary mapping each generated field to its value for one-to-one converts"
                )

                # wrap each answer in a list
                field_answers = {field_name: [answer[field_name]] for field_name in fields}

            else:
                assert isinstance(answer, list), "UDF must return a list of dictionaries for one-to-many converts"
                field_answers = {field_name: [] for field_name in fields}
                for answer_dict in answer:
                    assert isinstance(answer_dict, dict), "Each element of list returned by UDF must be a dictionary"
                    for field_name in fields:
                        field_answers[field_name].append(answer_dict.get(field_name, None))

            if self.verbose:
                print(f"{self.udf.__name__}:\n{answer}")

        except Exception as e:
            print(f"Error invoking user-defined function for convert: {e}")
            raise e

        # create generation stats object containing the time spent executing the UDF function
        generation_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)

        return field_answers, generation_stats


class LLMConvert(ConvertOp):
    """
    This is the base class for convert operations which use an LLM to generate the output fields.
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.MAP,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        if model is not None:
            self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, self.cardinality, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        op += f"    Reasoning Effort: {self.reasoning_effort}\n"
        return op

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

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the LLMConvert operation. Implicitly, these estimates
        assume the use of a single LLM call for each input record. Child classes of LLMConvert
        may call this function through super() and adjust these estimates as needed (or they can
        completely override this function).
        """
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

        # get est. of conversion time per record from model card;
        model_name = self.model.value
        model_conversion_time_per_record = MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        usd_per_input_token = MODEL_CARDS[model_name].get("usd_per_input_token")
        if getattr(self, "prompt_strategy", None) is not None and self.is_audio_op():
            usd_per_input_token = MODEL_CARDS[model_name]["usd_per_audio_input_token"]

        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality == Cardinality.ONE_TO_ONE else NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["overall"] / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )


class LLMConvertBonded(LLMConvert):

    def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[str, list], GenerationStats]:
        # get the set of input fields to use for the convert operation
        input_fields = self.get_input_fields()

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

        # generate outputs for all fields in a single query
        field_answers, _, generation_stats, _ = self.generator(candidate, fields, **gen_kwargs)

        # if there was an error for any field, execute a conventional query on that field
        if len(field_answers) > 1:
            for field_name, answers in field_answers.items():
                if answers is None:
                    single_field_answers, _, single_field_stats, _ = self.generator(candidate, {field_name: fields[field_name]}, **gen_kwargs)
                    field_answers.update(single_field_answers)
                    generation_stats += single_field_stats

        return field_answers, generation_stats

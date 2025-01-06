from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import Any, Callable

from PIL import Image

from palimpzest import prompts
from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    NAIVE_EST_ONE_TO_MANY_SELECTIVITY,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.generators.generators import DSPyGenerator, ImageTextGenerator
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.utils.generation_helpers import get_json_from_answer

# TYPE DEFINITIONS
FieldName = str


class ConvertOp(PhysicalOperator):
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
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "cardinality": self.cardinality,
            "udf": self.udf,
            "desc": self.desc,
            **op_params
        }

        return op_params

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")


class NonLLMConvert(ConvertOp):
    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.output_schema == other.output_schema
            and self.input_schema == other.input_schema
            and self.cardinality == other.cardinality
            and self.udf == other.udf
        )

    def __str__(self):
        op = super().__str__()
        op += f"    UDF: {str(self.udf)}\n"
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

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        """
        NOTE: if the UDF throws an exception we immediately raise it to the user. This way computation
              (and money) are not wasted on a workload which will fail to execute correctly.

              If the UDF outputs None or an empty list, then we will treat this as a failed convert.
        """
        # apply UDF to input record
        start_time = time.time()
        try:
            drs = self.udf(candidate)
        except Exception as e:
            print(f"Error invoking user-defined function for convert: {e}")
            raise e

        # time spent executing the UDF function
        fn_call_duration_secs = time.time() - start_time

        # determine whether or not the convert was successful
        successful_convert = drs is not None and (isinstance(drs, DataRecord) or len(drs) > 0)

        # if drs is a single record output, wrap it in a list
        if successful_convert and isinstance(drs, DataRecord):
            drs = [drs]

        # compute the total number of records we need to create RecordOpStats for;
        # we still create an object even if the convert fails
        num_records = len(drs) if successful_convert else 1

        # construct RecordOpStats
        record_op_stats_lst = []
        for idx in range(num_records):
            # compute variables which depend on data record
            record_id, record_parent_id, record_source_id, record_state, answer = None, candidate._id, candidate._source_id, None, None
            if successful_convert:
                dr = drs[idx]
                record_id = dr._id
                record_parent_id = dr._parent_id
                record_source_id = dr._source_id
                record_state = dr.as_dict(include_bytes=False)
                answer = {field_name: getattr(dr, field_name) for field_name in self.output_schema.field_names()},

            record_op_stats = RecordOpStats(
                record_id=record_id,
                record_parent_id=record_parent_id,
                record_source_id=record_source_id,
                record_state=record_state,
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=fn_call_duration_secs / len(drs),
                cost_per_record=0.0,
                answer=answer,
                input_fields=self.input_schema.field_names(),
                generated_fields=self.output_schema.field_names(),
                fn_call_duration_secs=fn_call_duration_secs / len(drs),
                failed_convert=(not successful_convert),
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct record set
        record_set = DataRecordSet(drs, record_op_stats_lst)

        return record_set


class LLMConvert(ConvertOp):
    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
        image_conversion: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.image_conversion = image_conversion

        # create DSPy generator
        if self.model is not None:
            doc_schema = str(self.output_schema)
            doc_type = self.output_schema.class_name()
            if self.image_conversion:
                self.generator = ImageTextGenerator(self.model, self.verbose)
            else:
                self.generator = DSPyGenerator(
                    self.model, self.prompt_strategy, doc_schema, doc_type, self.verbose
                )

    def __str__(self):
        op = super().__str__()
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "model": None if self.model is None else self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "image_conversion": self.image_conversion,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "image_conversion": self.image_conversion,
            **op_params,
        }

        return op_params

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

        if self.image_conversion:
            est_num_input_tokens = 765 / 10  # 1024x1024 image is 765 tokens

        # get est. of conversion time per record from model card;
        # NOTE: model will only be None for code synthesis, which uses GPT-3.5 as fallback
        model_name = self.model.value if getattr(self, "model", None) is not None else Model.GPT_4o_MINI.value
        model_conversion_time_per_record = (
            MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[model_name]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality == Cardinality.ONE_TO_ONE else NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["overall"] / 100.0) * source_op_cost_estimates.quality

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def _construct_query_prompt(
        self,
        fields_to_generate: list[str],
        model: Model,
    ) -> str:
        """
        This function constructs the prompt for a bonded query.
        """
        # set defaults
        doc_type = self.output_schema.class_name()
        # build string of input fields and their descriptions
        multiline_input_field_description = ""
        depends_on_fields = (
            [field.split(".")[-1] for field in self.depends_on]
            if self.depends_on is not None and len(self.depends_on) > 0
            else None
        )
        input_fields = (
            self.input_schema.field_names()
            if depends_on_fields is None
            else [field for field in self.input_schema.field_names() if field in depends_on_fields]
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

        # add optional model instruction
        model_instruction = prompts.LLAMA_INSTRUCTION if model in [Model.LLAMA3, Model.LLAMA3_V] else ""

        # construct sentence fragments which depend on cardinality of conversion (pz.Cardinality.ONE_TO_ONE or pz.Cardinality.ONE_TO_MANY)
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
        if not self.image_conversion:
            prompt_question = prompts.STRUCTURED_CONVERT_PROMPT
        else:
            prompt_question = prompts.IMAGE_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            target_output_descriptor=target_output_descriptor,
            input_type=self.input_schema.class_name(),
            output_single_or_plural=output_single_or_plural,
            optional_input_desc=optional_input_desc,
            optional_output_desc=optional_output_desc,
            multiline_input_field_description=multiline_input_field_description,
            multiline_output_field_description=multiline_output_field_description,
            appendix_instruction=appendix_instruction,
            optional_desc=optional_desc,
            model_instruction = model_instruction,
        )
        # TODO: add this for boolean questions?
        # if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
        #     promptQuestion += "\nRemember, your output MUST be one of TRUE or FALSE."

        return prompt_question

    def _create_record_set(
        self,
        records: list[DataRecord],
        fields: list[str],
        generation_stats: GenerationStats,
        total_time: float,
        successful_convert: bool,
    ) -> DataRecordSet:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        record_op_stats_lst = []

        # compute variables
        num_records = len(records)
        per_record_stats = generation_stats / num_records
        model = getattr(self, "model", None)

        # NOTE: in some cases, we may generate outputs which fail to parse correctly,
        #       thus `record_set` contains an empty list, but we still want to capture the
        #       the cost of the failed generation; in this case, we set num_records = 1
        #       and compute a RecordOpStats with some fields None'd out and failed_convert=True
        for idx in range(num_records):
            # compute variables which depend on data record
            dr = records[idx]
            record_id = dr._id
            record_parent_id = dr._parent_id
            record_source_id = dr._source_id
            record_state = dr.as_dict(include_bytes=False)
            answer = {field_name: getattr(dr, field_name) for field_name in fields}

            record_op_stats = RecordOpStats(
                record_id=record_id,
                record_parent_id=record_parent_id,
                record_source_id=record_source_id,
                record_state=record_state,
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=total_time / num_records,
                cost_per_record=per_record_stats.cost_per_record,
                model_name=model.value if model else None,
                answer=answer,
                input_fields=self.input_schema.field_names(),
                generated_fields=fields,
                total_input_tokens=per_record_stats.total_input_tokens,
                total_output_tokens=per_record_stats.total_output_tokens,
                total_input_cost=per_record_stats.total_input_cost,
                total_output_cost=per_record_stats.total_output_cost,
                llm_call_duration_secs=per_record_stats.llm_call_duration_secs,
                fn_call_duration_secs=per_record_stats.fn_call_duration_secs,
                failed_convert=(not successful_convert),
                image_operation=self.image_conversion,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # create and return the DataRecordSet
        record_set = DataRecordSet(records, record_op_stats_lst)

        return record_set

    def _create_data_record_from_json(
        self,
        json_obj: Any,
        candidate: DataRecord,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # initialize data record
        dr = DataRecord.from_parent(self.output_schema, parent_record=candidate, cardinality_idx=cardinality_idx)

        # TODO: This inherits all pre-computed fields in an incremental fashion. The positive / pros of this approach is that it enables incremental schema computation, which tends to feel more natural for the end-user. The downside is it requires us to support an explicit projection to eliminate unwanted input / intermediate computation.
        #
        # first, copy all fields from input schema
        for field_name in candidate.get_fields():
            setattr(dr, field_name, getattr(candidate, field_name, None))

        # get input field names and output field names
        input_fields = self.input_schema.field_names()
        output_fields = self.output_schema.field_names()

        # parse newly generated fields from the generated jsonObj
        for field_name in output_fields:
            if field_name not in input_fields:
                # parse the json object and set the DataRecord's fields with their generated values
                setattr(
                    dr, field_name, json_obj.get(field_name, None)
                )  # the use of get prevents a KeyError if an individual field is missing.

        return dr

    def parse_answer(self, answer: str, fields_to_generate: list[str], model: Model) -> dict[FieldName, list[Any]]:
        """ 
        This functions gets a string answer and parses it into an iterable format of [{"field1": value1, "field2": value2}, {...}, ...]
        """
        try:
            # parse json from answer string
            json_answer = get_json_from_answer(answer, model)

            # sanity check validity of parsed json
            assert json_answer != {}, "No output was found!"
            if self.cardinality == Cardinality.ONE_TO_MANY:
                assert "items" in json_answer, '"items" key missing from one-to-many JSON'
                assert (
                    isinstance(json_answer["items"], list) and len(json_answer["items"]) > 0
                ), "No output objects were generated for one-to-many query"
            else:
                assert all([field in json_answer for field in fields_to_generate]), "Not all fields were generated!"

        except Exception as e:
            print(f"Error parsing LLM answer: {e}")
            print(f"\tAnswer: {answer}")
            # msg = str(e)
            # if "line" in msg:
            #     line = int(str(msg).split("line ")[1].split(" ")[0])
            #     print(f"\tAnswer snippet: {answer.splitlines()[line]}")
            return {field_name: [] for field_name in fields_to_generate}

        field_answers = {}
        if self.cardinality == Cardinality.ONE_TO_MANY:
            # json_answer["items"] is a list of dictionaries, each of which contains the generated fields
            for field in fields_to_generate:
                field_answers[field] = []
                for item in json_answer["items"]:
                    try:
                        field_answers[field].append(item[field])
                    except Exception:
                        print(f"Error parsing field {field} in one-to-many answer: {item}")

        else:
            field_answers = {field: [json_answer[field]] for field in fields_to_generate}

        return field_answers

    def _get_candidate_content(self, model: Model, candidate: DataRecord):
        # get text or image content depending on prompt strategy
        if self.image_conversion:
            base64_images = []
            if hasattr(candidate, "contents"):
                # TODO: should address this now; we need a way to infer (or have the programmer declare) what fields contain image content
                base64_images = [
                    base64.b64encode(candidate.contents).decode("utf-8")  
                ]
            elif model in [Model.GPT_4o_V, Model.GPT_4o_MINI_V]:
                for image_file in candidate.image_filepaths:  # TODO: (see note above)
                    image = Image.open(image_file)
                    buffered = BytesIO()
                    image.save(buffered, format=image.format)
                    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    base64_images.append(base64_image)

            # for LLAMA vision model, we must concatenate images into a single image
            elif model in [Model.LLAMA3_V]:
                # TODO: revert after paper submission
                # # load images, get their dimensions, and create new image to fit them horizontally
                # images = [Image.open(image_file) for image_file in candidate.image_filepaths]
                # widths, heights = zip(*(img.size for img in images))
                # total_width, max_height = sum(widths), max(heights)
                # new_image = Image.new(images[0].mode, (total_width, max_height))

                # # construct new image by pasting images side-by-side
                # x_offset = 0
                # for img in images:
                #     new_image.paste(img, (x_offset,0))
                #     x_offset += img.size[0]

                # # crop new image to adhere to max size processed by LLAMA; I'm not sure
                # # what the exact max size allowed by Together is, but 900x900 seems to work
                # crop_height = 900
                # crop_width = 900
                # new_image = new_image.crop((0, 0, crop_width, crop_height))

                # # encode new image in base64
                # buffered = BytesIO()
                # new_image.save(buffered, format=images[0].format)
                # base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # base64_images.append(base64_image)
                
                # NOTE: Together stopped accepting uploaded images, so we now point them to public s3 urls
                listing_idx = int(candidate.listing.split("listing")[-1])
                url = f"https://palimpzest-workloads.s3.amazonaws.com/real-estate-eval-concat-images/img{listing_idx}.png"
                base64_images.append(url)

            content = base64_images
        else:
            content = candidate.as_json_str(include_bytes=False, project_cols=self.depends_on)

        return content

    def _dspy_generate_fields(
        self,
        prompt: str,
        content: str | list[bytes] | None = None,  # either text or image
    ) -> tuple[str, GenerationStats]:
        """
        This functions wraps the call to the generator method to actually perform the field generation.
        Returns an answer which is a string and a query_stats which is a GenerationStats object.
        """
        # generate LLM response and capture statistics
        answer:str
        query_stats:GenerationStats
        try:
            answer, _, query_stats = self.generator.generate(context=content, prompt=prompt)

        except Exception as e:
            print(f"DSPy generation error: {e}")
            return "", GenerationStats()

        return answer, query_stats

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        """ This function is responsible for the LLM conversion process. 
        Different strategies may/should reimplement this function and leave the __call__ function untouched.
        The input is ...
        Outputs:
         - field_outputs: a dictionary where keys are the fields and values are lists of JSON corresponding to the value of the field in that record.
         - query_stats
        """
        raise NotImplementedError("This is an abstract class. Use a subclass instead!")

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # get fields to generate with this convert
        fields_to_generate = self._generate_field_names(candidate, self.input_schema, self.output_schema)

        # execute the convert
        field_answers: dict[str, list]
        field_answers, generation_stats = self.convert(candidate=candidate, fields=fields_to_generate)

        # construct list of dictionaries where each dict. has the (field, value) pairs for each generated field
        # list is indexed per record
        try:
            n_records = max([len(lst) for lst in field_answers.values()])
        except Exception:
            print(f"Error in field answers: {field_answers}. Returning empty records.")
            n_records = 0
            field_answers = {}

        drs = []
        if n_records > 0:
            # build up list of final record dictionaries
            records_json = [{field: None for field in fields_to_generate} for _ in range(n_records)]
            for field_name, answer_list in field_answers.items():
                for idx, output in enumerate(answer_list):
                    records_json[idx][field_name] = output

            # construct list of data records
            drs = [
                self._create_data_record_from_json(json_obj=js, candidate=candidate, cardinality_idx=idx)
                for idx, js in enumerate(records_json)
            ]
        else:
            null_js = {field: None for field in fields_to_generate}
            drs = [self._create_data_record_from_json(json_obj=null_js, candidate=candidate, cardinality_idx=0)]

        # construct and return DataRecordSet
        record_set = self._create_record_set(
            records=drs,
            fields=fields_to_generate,
            generation_stats=generation_stats,
            total_time=time.time() - start_time,
            successful_convert=(n_records > 0),
        )

        return record_set


class LLMConvertConventional(LLMConvert):
    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Update the cost per record and time per record estimates to account for the additional
        LLM calls we incur by executing one query per-field.
        """
        # get naive cost estimates from LLMConvert
        naive_op_cost_estimates = super().naive_cost_estimates(source_op_cost_estimates)

        # re-compute cost per record assuming we use fewer input tokens
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

        # increase estimates of the input and output tokens by the number of fields generated
        # NOTE: this may over-estimate the number of fields that need to be generated
        generate_field_names = []
        for field_name in self.output_schema.field_names():
            if field_name not in self.input_schema.field_names():
                generate_field_names.append(field_name)

        num_fields_to_generate = len(generate_field_names)
        est_num_input_tokens *= num_fields_to_generate
        est_num_output_tokens *= num_fields_to_generate

        if self.image_conversion:
            est_num_input_tokens = 765 / 10  # 1024x1024 image is 765 tokens

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # set refined estimate of time and cost per record
        naive_op_cost_estimates.time_per_record = model_conversion_time_per_record
        naive_op_cost_estimates.time_per_record_lower_bound = naive_op_cost_estimates.time_per_record
        naive_op_cost_estimates.time_per_record_upper_bound = naive_op_cost_estimates.time_per_record
        naive_op_cost_estimates.cost_per_record = model_conversion_usd_per_record
        naive_op_cost_estimates.cost_per_record_lower_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.cost_per_record_upper_bound = naive_op_cost_estimates.cost_per_record

        return naive_op_cost_estimates

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        fields_answers = {}
        fields_stats = {}
        candidate_content = self._get_candidate_content(self.model, candidate)
        for field_name in fields:
            prompt = self._construct_query_prompt(fields_to_generate=[field_name], model=self.model)
            answer, stats = self._dspy_generate_fields(content=candidate_content, prompt=prompt)
            json_answer = self.parse_answer(answer, [field_name], self.model)
            fields_answers.update(json_answer)
            fields_stats[field_name] = stats

        generation_stats = sum(fields_stats.values())
        return fields_answers, generation_stats


class LLMConvertBonded(LLMConvert):

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        prompt = self._construct_query_prompt(fields_to_generate=fields, model=self.model)
        candidate_content = self._get_candidate_content(self.model, candidate)

        # generate all fields in a single query
        answer, generation_stats = self._dspy_generate_fields(content=candidate_content, prompt=prompt)
        json_answers = self.parse_answer(answer, fields, self.model)

        # if there was an error for any field, execute a conventional query on that field
        for field, values in json_answers.items():
            if values == []:
                conventional_op = LLMConvertConventional(**self.get_op_params())

                field_answer, field_stats = conventional_op.convert(candidate, [field])
                json_answers[field] = field_answer[field]
                generation_stats += field_stats

        return json_answers, generation_stats

from __future__ import annotations

from palimpzest import prompts
from palimpzest.constants import *
from palimpzest.corelib import *
from palimpzest.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.generators import CustomGenerator, DSPyGenerator, ImageTextGenerator
from palimpzest.operators import logical, DataRecordsWithStats, PhysicalOperator
from palimpzest.utils import API, getJsonFromAnswer

from typing import Any, Dict, List, Optional, Tuple, Union

import random
import base64
import math
import time

from palimpzest.utils.model_helpers import getVisionModels

# TYPE DEFINITIONS
FieldName = str


class ConvertOp(PhysicalOperator):

    inputSchema = Schema
    outputSchema = Schema

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        desc: Optional[str] = None,
        targetCacheId: Optional[str] = None,
        shouldProfile: bool = False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            shouldProfile=shouldProfile,
        )
        self.cardinality = cardinality
        self.desc = desc
        self.targetCacheId = targetCacheId
        self.heatmap_json_obj = None

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "inputSchema": str(self.inputSchema),
            "outputSchema": str(self.outputSchema),
            "cardinality": self.cardinality.value,
            "desc": str(self.desc),
        }


    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")


class LLMConvert(ConvertOp):
    implemented_op = logical.ConvertScan
    model: Model
    prompt_strategy: PromptStrategy

    @classmethod
    def materializes(self, logical_operator) -> bool:
        if not isinstance(logical_operator, logical.ConvertScan):
            return False
        is_vision_model = self.model in getVisionModels()
        if logical_operator.image_conversion:
            return is_vision_model
        else:
            return not is_vision_model
        # use image model if this is an image conversion


    def __init__(
        self,
        query_strategy: Optional[QueryStrategy] = None,
        image_conversion: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.query_strategy = query_strategy
        self.image_conversion = image_conversion

        # for now, forbid CodeSynthesis on one-to-many cardinality queries
        if self.cardinality == Cardinality.ONE_TO_MANY:
            assert self.query_strategy != QueryStrategy.CODE_GEN_WITH_FALLBACK, "Cannot run code-synthesis on one-to-many operation"
        # TODO find a place where this is being checked by the planner
        if self.outputSchema == ImageFile and self.inputSchema == File or self.image_conversion:
            # TODO : find a more general way by llm provider
            # TODO : which module is responsible of setting PromptStrategy.IMAGE_TO_TEXT?
            if self.model in [Model.GPT_3_5, Model.GPT_4]:
                self.model = Model.GPT_4V
            if self.model == Model.GEMINI_1:
                self.model = Model.GEMINI_1V
            if self.model in [Model.MIXTRAL, Model.LLAMA2]:
                self.model = random.choice([Model.GPT_4V, Model.GEMINI_1V])

            # TODO: in the future remove; for evaluations just use GPT_4V
            self.model = Model.GPT_4V
            self.prompt_strategy = PromptStrategy.IMAGE_TO_TEXT
            self.query_strategy = QueryStrategy.BONDED_WITH_FALLBACK

        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            self.model = None
            self.prompt_strategy = None

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.cardinality == other.cardinality
            and self.image_conversion == other.image_conversion
            and self.prompt_strategy == other.prompt_strategy
            and self.query_strategy == other.query_strategy
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    def __str__(self):
        op = super().__str__()
        op += f"Prompt strategy: {self.prompt_strategy}\n"
        op += f"Query strategy: {self.query_strategy}\n"
        return op
        # model = getattr(self, "model", "")
        # ps = getattr(self, "prompt_strategy", "")
        # qs = getattr(self, "query_strategy", "")
        # return f"{self.__class__.__name__}({str(self.inputSchema):10s}->{str(self.outputSchema):10s}, Model: {model}, Prompt Strategy: {ps}, Query Strategy: {qs})"

    def copy(self):
        return self.__class__(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            query_strategy=self.query_strategy,
            desc=self.desc,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "inputSchema": str(self.inputSchema),
            "outputSchema": str(self.outputSchema),
            "cardinality": self.cardinality,
            "model": self.model.value if getattr(self, "model", None) else None,
            "prompt_strategy": (
                self.prompt_strategy.value if getattr(self, "prompt_strategy", None) else None
            ),
            "query_strategy": (
                self.query_strategy.value if getattr(self, "query_strategy", None) else None
            ),
            "desc": str(self.desc),
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

        if self.query_strategy == QueryStrategy.CONVENTIONAL:
            # NOTE: this may over-estimate the number of fields that need to be generated
            generate_field_names = []
            for field_name in self.outputSchema.fieldNames():
                if field_name not in self.inputSchema.fieldNames(): # and getattr(candidate, field_name, None) is None:
                    generate_field_names.append(field_name)

            num_fields_to_generate = len(generate_field_names)
            est_num_input_tokens *= num_fields_to_generate
            est_num_output_tokens *= num_fields_to_generate

        # TODO REMOVE!
        self.token_budget = 1.
        if self.token_budget is not None:
            est_num_input_tokens = self.token_budget * est_num_input_tokens

        # override for GPT-4V image conversion
        if self.model == Model.GPT_4V:
            # 1024x1024 image is 765 tokens
            # TODO: revert / 10 after running real-estate demo
            est_num_input_tokens = 765 / 10

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        # NOTE: model will only be None for code synthesis, which uses GPT-3.5 as fallback
        model_name = self.model.value if self.model is not None else Model.GPT_3_5.value
        model_conversion_time_per_record = (
            MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens
        ) / self.max_workers

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[model_name]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # TODO: make this better after arxiv; right now codesynth is hard-coded to use GPT-4
        # if we're using code synthesis, assume that model conversion time and cost are low
        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            model_conversion_time_per_record = 1e-5
            model_conversion_usd_per_record = 1e-4  # amortize code synth cost across records

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality == "oneToOne" else NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["MMLU"] / 100.0) * source_op_cost_estimates.quality

        # TODO: make this better after arxiv; right now codesynth is hard-coded to use GPT-4
        # if we're using code synthesis, assume that quality goes down (or view it as E[Quality] = (p=gpt4[code])*1.0 + (p=0.25)*0.0))
        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            quality = quality * (GPT_4_MODEL_CARD["code"] / 100.0)

        if self.token_budget is not None:
            # for now, assume quality is proportional to sqrt(token_budget)
            quality = quality * math.sqrt(math.sqrt(self.token_budget))  

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def _generate_field_names(self, candidate: DataRecord, inputSchema: Schema, outputSchema: Schema) -> List[str]:
        """
        Creates the list of field names that the convert operation needs to generate.
        """
        # construct the list of fields in outputSchema which will need to be generated;
        # specifically, this is the set of fields which are:
        # 1. not declared in the input schema, and
        # 2. not present in the candidate's attributes
        #    a. if the field is present, but its value is None --> we will try to generate it
        fields_to_generate = []
        for field_name in outputSchema.fieldNames():
            if field_name not in inputSchema.fieldNames() and getattr(candidate, field_name, None) is None:
                fields_to_generate.append(field_name)

        return fields_to_generate

    def _construct_query_prompt(
        self,
        fields_to_generate: List[str],
    ) -> str:
        """
        This function constructs the prompt for a bonded query.
        """
        # set defaults
        doc_type = self.outputSchema.className()
        # build string of input fields and their descriptions
        multilineInputFieldDescription = ""
        for field_name in self.inputSchema.fieldNames():
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
            targetOutputDescriptor = prompts.ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL
            appendixInstruction = prompts.ONE_TO_MANY_APPENDIX_INSTRUCTION
        else:
            targetOutputDescriptor = prompts.ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR.format(doc_type=doc_type)
            outputSingleOrPlural = prompts.ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL
            appendixInstruction = prompts.ONE_TO_ONE_APPENDIX_INSTRUCTION

        # construct promptQuestion
        optional_desc = "" if self.desc is None else prompts.OPTIONAL_DESC.format(desc=self.desc)
        if self.prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
            prompt_question = prompts.STRUCTURED_CONVERT_PROMPT
        else:
            prompt_question = prompts.IMAGE_CONVERT_PROMPT

        prompt_question = prompt_question.format(
            targetOutputDescriptor=targetOutputDescriptor,
            input_type = self.inputSchema.className(),
            outputSingleOrPlural = outputSingleOrPlural,
            optionalInputDesc = optionalInputDesc,
            optionalOutputDesc = optionalOutputDesc,
            multilineInputFieldDescription = multilineInputFieldDescription,
            multilineOutputFieldDescription = multilineOutputFieldDescription,
            appendixInstruction = appendixInstruction,
            optional_desc = optional_desc
        )
        # TODO: add this for boolean questions?
        # if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
        #     promptQuestion += "\nRemember, your output MUST be one of TRUE or FALSE."

        return prompt_question

    def _create_record_op_stats_lst(
        self,
        records: List[DataRecord],
        fields: List[str],
        generation_stats: GenerationStats,
        total_time: float,
    ) -> List[RecordOpStats]:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        record_op_stats_lst = []
        per_record_stats = generation_stats / len(records)
        model = getattr(self, "model", None)
        for dr in records:
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=total_time / len(records),
                cost_per_record=per_record_stats.cost_per_record,
                model_name=model.value if model else None,
                answer={field_name: getattr(dr, field_name) for field_name in fields},
                input_fields=self.inputSchema.fieldNames(),
                generated_fields=fields,
                total_input_tokens=per_record_stats.total_input_tokens,
                total_output_tokens=per_record_stats.total_output_tokens,
                total_input_cost=per_record_stats.total_input_cost,
                total_output_cost=per_record_stats.total_output_cost,
                llm_call_duration_secs=per_record_stats.llm_call_duration_secs,
                fn_call_duration_secs=per_record_stats.fn_call_duration_secs,
            )
            record_op_stats_lst.append(record_op_stats)

        return record_op_stats_lst

    def _create_data_record_from_json(
        self,
        jsonObj: Any, 
        candidate: DataRecord,
        cardinality_idx: int = None,
    ) -> DataRecord:
        # initialize data record
        dr = DataRecord(
            self.outputSchema, parent_uuid=candidate._uuid, cardinality_idx=cardinality_idx
        )

        # TODO: This inherits all pre-computed fields in an incremental fashion. The positive / pros of this approach is that it enables incremental schema computation, which tends to feel more natural for the end-user. The downside is it requires us to support an explicit projection to eliminate unwanted input / intermediate computation.
        #
        # first, copy all fields from input schema
        # NOTE: the method is called _getFields instead of getFields to avoid it being picked up as a data record attribute;
        #       in the future we will come up with a less ugly fix -- but for now do not remove the _ even though it's not private
        for field_name in candidate._getFields():
            setattr(dr, field_name, getattr(candidate, field_name, None))

        # get input field names and output field names
        input_fields = self.inputSchema.fieldNames()
        output_fields = self.outputSchema.fieldNames()

        # parse newly generated fields from the generated jsonObj
        for field_name in output_fields:
            if field_name not in input_fields:
                # parse the json object and set the DataRecord's fields with their generated values
                setattr(
                    dr, field_name, jsonObj.get(field_name, None)
                )  # the use of get prevents a KeyError if an individual field is missing.

        return dr

    def parse_answer(
        self, answer: str, fields_to_generate: List[str]
    ) -> List[Dict[FieldName, List]]:
        """ 
        This functions gets a string answer and parses it into an iterable format of [{"field1": value1, "field2": value2}, {...}, ...]
        # """
        try:
            json_answer = getJsonFromAnswer(answer)
            assert json_answer != {}, "No output was found!"
            
            if self.cardinality == Cardinality.ONE_TO_MANY:
                assert (isinstance(json_answer["items"], list) and len(json_answer["items"]) > 0), "No output objects were generated for one-to-many query"               
            else:
                assert all([field in json_answer for field in fields_to_generate]), "Not all fields were generated!"

        except Exception as e:
            print(f"Error parsing LLM answer: {e}")
            print(f"\tAnswer: {answer}")
            breakpoint()
            # msg = str(e)
            # if "line" in msg:
                # line = int(str(msg).split("line ")[1].split(" ")[0])
                # print(f"\tAnswer snippet: {answer.splitlines()[line]}")
            if self.cardinality == Cardinality.ONE_TO_MANY:
                json_answer = {"items":[{field_name: None for field_name in fields_to_generate} for _ in range(1)]}
            else:
                json_answer = {field_name: [] for field_name in fields_to_generate}

        field_answers = {}
        if self.cardinality == Cardinality.ONE_TO_MANY:
            # json_answer["items"] is a list of dictionaries, each of which contains the generated fields
            for field in fields_to_generate:
                field_answers[field] = []
                for item in json_answer["items"]:
                    field_answers[field].append(item[field])
        else:
            field_answers = {
                field: [json_answer[field]] for field in fields_to_generate
            }

        return field_answers

    def _dspy_generate_fields(
        self,
        prompt: str,
        content: Optional[Union[str, List[bytes]]] = None, #either text or image
        verbose: bool = False,
    ) -> Tuple[str, GenerationStats]:
        """ This functions wraps the call to the generator method to actually perform the field generation. Returns an answer which is a string and a query_stats which is a GenerationStats object.
        """
        # create DSPy generator and generate
        doc_schema = str(self.outputSchema)
        doc_type = self.outputSchema.className()

        # generate LLM response and capture statistics
        answer:str
        query_stats:GenerationStats
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            generator = DSPyGenerator(
                self.model.value, self.prompt_strategy, doc_schema, doc_type, verbose
            )
        elif self.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            generator = ImageTextGenerator(self.model.value)
        else:
            raise Exception(f"Prompt strategy not implemented: {self.prompt_strategy}")

        try:
            answer, query_stats = generator.generate(context=content, question=prompt)
        except Exception as e:
            print(f"DSPy generation error: {e}")
            return "", GenerationStats()
        
        return answer, query_stats

    def convert(self, candidate_content: Union[str,List[bytes]] , fields: List[str]) -> Tuple[Dict[str, List], StatsDict]:
        """ This function is responsible for the LLM conversion process. 
        Different strategies may/should reimplement this function and leave the __call__ function untouched.
        The input is ...
        Outputs:
         - field_outputs: a dictionary where keys are the fields and values are lists of JSON corresponding to the value of the field in that record.
         - query_stats
        """
        raise NotImplementedError("This is an abstract class. Use a subclass instead!")

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        start_time = time.time()
        fields_to_generate = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

        # get text or image content depending on prompt strategy
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            content = candidate._asJSONStr(include_bytes=False)
        elif self.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            base64_images = []
            if hasattr(candidate, "contents"):
                # TODO: should address this now; we need a way to infer (or have the programmer declare) what fields contain image content
                base64_images = [
                    base64.b64encode(candidate.contents).decode("utf-8")  
                ]
            else:
                base64_images = [
                    base64.b64encode(image).decode("utf-8")
                    for image in candidate.image_contents  # TODO: (see note above)
                ]
            content = base64_images
        else:
            raise Exception(f"Prompt strategy not implemented: {self.prompt_strategy}")

        field_answers: Dict[str, List]
        field_answers, generation_stats = self.convert(
            fields=fields_to_generate, candidate_content=content
        )

        # construct list of dictionaries where each dict. has the (field, value) pairs for each generated field
        # list is indexed per record
        n_records = max([len(lst) for lst in field_answers.values()])
        records_json = [{field: None for field in fields_to_generate} for _ in range(n_records)]

        for field_name, answer_list in field_answers.items():
            for idx, output in enumerate(answer_list):
                record = records_json[idx]
                record[field_name] = output

        drs = [
            self._create_data_record_from_json(
                jsonObj=js, candidate=candidate, cardinality_idx=idx
            )
            for idx, js in enumerate(records_json)
        ]

        total_time = time.time() - start_time
        record_op_stats_lst = self._create_record_op_stats_lst(
            records=drs,
            fields=fields_to_generate,
            generation_stats=generation_stats,
            total_time=total_time,
        )

        return drs, record_op_stats_lst


class LLMConvertConventional(LLMConvert):

    def convert(
        self, candidate_content: Union[str, List[bytes]], fields: List[str]
    ) -> Tuple[dict[List], GenerationStats]:

        # if self.cardinality == Cardinality.ONE_TO_MANY:
        if False:
            breakpoint()
            # TODO here the problem is: which is the 1:N field that we are splitting the output into?
            # do we need to know this to construct the prompt question ?
            # for now, we will just assume there is only one list in the JSON.
            dct = json.loads(text_content)
            split_attribute = [att for att in dct.keys() if type(dct[att]) == list][0]
            n_splits = len(dct[split_attribute])

            if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # TODO Hacky to nest return and not disrupt the rest of method!!!
                # NOTE: this is a bonded query, but we are treating it as a conventional query
                query_stats = {}
                drs = [] 
                promptQuestion = _construct_query_prompt(td, doc_type, generate_field_names)

                # iterate over the length of the split attribute, and generate a new JSON for each split
                for idx in range(n_splits):
                    if verbose: 
                        print(f"Processing {split_attribute} with index {idx}")
                    new_json = {k:v for k,v in dct.items() if k != split_attribute}
                    new_json[split_attribute] = dct[split_attribute][idx]

                    text_content = json.dumps(new_json)
                    generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                    answer, record_stats = None, None
                    try:
                        answer, _, record_stats = generator.generate(text_content, promptQuestion, plan_idx=td.plan_idx)
                        jsonObj = _get_JSON_from_answer(answer)["items"][0]
                        query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                    except IndexError as e:
                        query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                        print("Could not find any items in the JSON response")
                        continue
                    except json.JSONDecodeError as e:
                        query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                        print(f"Could not decode JSON response: {e}")
                        print(answer)
                        continue
                    except Exception as e:
                        query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                        print(f"Could not decode JSON response: {e}")
                        print(answer)
                        continue

                    dr = _create_data_record_from_json(jsonObj, td, candidate, cardinality_idx=idx)
                    drs.append(dr)

                # TODO how to stat this? I feel that we need a new Stats class for this type of query
                # construct ConventionalQueryStats object
                field_query_stats_lst = [FieldQueryStats(gen_stats=gen_stats, field_name=field_name) for
                                            field_name, gen_stats in query_stats.items()]
                conventional_query_stats = ConventionalQueryStats(
                    field_query_stats_lst=field_query_stats_lst,
                    input_fields=td.inputSchema.fieldNames(),
                    generated_fields=generate_field_names,
                )

                # TODO: debug root cause
                for dr in drs:
                    if not hasattr(dr, 'filename'):
                        setattr(dr, 'filename', candidate.filename)

                return drs, conventional_query_stats

            else:
                raise Exception("Conventional queries cannot execute tasks with cardinality == 'oneToMany'")

        else:
            fields_answers = {}
            fields_stats = {}

            for field_name in fields:
                prompt = self._construct_query_prompt(fields_to_generate=[field_name])
                answer, stats = self._dspy_generate_fields(
                    content=candidate_content,
                    prompt=prompt,
                )
                json_answer = self.parse_answer(answer, [field_name])
                fields_answers.update(json_answer)
                fields_stats[field_name] = stats

            generation_stats = sum(fields_stats.values())
            return fields_answers, generation_stats

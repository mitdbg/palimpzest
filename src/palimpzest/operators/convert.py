from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib import *
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.generators import CustomGenerator, DSPyGenerator, ImageTextGenerator
from palimpzest.operators import logical, DataRecordsWithStats, PhysicalOperator
from palimpzest.utils import API, getJsonFromAnswer

from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import concurrent
import math
import time

# CODE SYNTHESIS PROMPTS
EXAMPLE_PROMPT = """Example{idx}:
{example_inputs}
{example_output}
"""

ADVICEGEN_PROMPT = """You are a helpful programming assistant and an expert {language} programmer. Your job is to provide programming ideas to help me write {language} programs.
For example, if I want to complete a task: "extract the salary number (in USD) from a given employee's document", you can provide me with {n} different ways to do it like:
Idea 1: Use regular expressions to extract the salary number: a number with a dollar sign in front of it. For example, $100,000.
Idea 2: Find the table entry with the salary number.
Idea 3: Use a pre-trained NLP model to extract the salary number.
# 
Now, consider the following {language} programming task that extracts `{output}` ({output_desc}) from given inputs:
{examples_desc}
Please provide me with {n} different ideas to complete this task. Return the ideas only, following the format above.
"""


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

    def __str__(self):
        return f"{self.model}_{self.query_strategy}_{self.token_budget}"

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    # TODO: where does caching go?
    # def __iter__(self) -> IteratorFn:
    #     shouldCache = self.datadir.openCache(self.targetCacheId)

    #     @self.profile(name="convert", shouldProfile=self.shouldProfile)
    #     def iteratorFn():
    #         for nextCandidate in self.source:
    #             resultRecordList = self.__call__(nextCandidate)
    #             if resultRecordList is not None:
    #                 for resultRecord in resultRecordList:
    #                     if resultRecord is not None:
    #                         if shouldCache:
    #                             self.datadir.appendCache(
    #                                 self.targetCacheId, resultRecord
    #                             )
    #                         yield resultRecord
    #         if shouldCache:
    #             self.datadir.closeCache(self.targetCacheId)

    #     return iteratorFn()


class ParallelConvertFromCandidateOp(ConvertOp):
    def __init__(self, streaming, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 32  # TODO hardcoded for now
        self.streaming = streaming

    def __eq__(self, other: PhysicalOperator):
        return super().__eq__(other) and self.streaming == other.streaming

    def copy(self):
        return super().copy(streaming=self.streaming)

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_convert", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks using self.max_workers
            if self.streaming:
                chunksize = self.max_workers
            else:
                chunksize = len(inputs)

            if chunksize == 0:
                return

            for i in range(0, len(inputs), chunksize):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    results = list(
                        executor.map(self.__call__, inputs[i : i + chunksize])
                    )

                    for resultRecordList in results:
                        if resultRecordList is not None:
                            for resultRecord in resultRecordList:
                                if resultRecord is not None:
                                    if shouldCache:
                                        self.datadir.appendCache(
                                            self.targetCacheId, resultRecord
                                        )
                                    yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class LLMConvert(ConvertOp):
    implemented_op = logical.ConvertScan
    model: Model
    prompt_strategy: PromptStrategy

    def __init__(
        self,
        query_strategy: Optional[QueryStrategy] = None,
        token_budget: Optional[float] = None,
        image_conversion: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.query_strategy = query_strategy
        self.token_budget = token_budget
        self.image_conversion = image_conversion

        # for now, forbid CodeSynthesis on one-to-many cardinality queries
        if self.cardinality == Cardinality.ONE_TO_MANY:
            assert self.query_strategy != QueryStrategy.CODE_GEN_WITH_FALLBACK, "Cannot run code-synthesis on one-to-many operation"

        # optimization-specific attributes
        self.heatmap_json_obj = None
        self.field_to_code_ensemble = None
        self.exemplars = None
        self.code_synthesized = False
        self.gpt4_llm = CustomGenerator(model_name=Model.GPT_4.value)

        # use image model if this is an image conversion
        if self.outputSchema == ImageFile and self.inputSchema == File or self.image_conversion:
            # TODO : find a more general way by llm provider
            # TODO : which module is responsible of setting PromptStrategy.IMAGE_TO_TEXT?
            if self.model in [Model.GPT_3_5, Model.GPT_4]:
                self.model = Model.GPT_4V
            if self.model == Model.GEMINI_1:
                self.model = Model.GEMINI_1V
            if self.model in [Model.MIXTRAL, Model.LLAMA2]:
                import random

                self.model = random.choice([Model.GPT_4V, Model.GEMINI_1V])

            # TODO: in the future remove; for evaluations just use GPT_4V
            self.model = Model.GPT_4V
            self.prompt_strategy = PromptStrategy.IMAGE_TO_TEXT
            self.query_strategy = QueryStrategy.BONDED_WITH_FALLBACK
            self.token_budget = 1.0

        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            self.model = None
            self.prompt_strategy = None
            self.token_budget = 1.0

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.cardinality == other.cardinality
            and self.image_conversion == other.image_conversion
            and self.prompt_strategy == other.prompt_strategy
            and self.query_strategy == other.query_strategy
            and self.token_budget == other.token_budget
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    def __str__(self):
        model = self.model.value if self.model is not None else ""
        ps = self.prompt_strategy.value if self.prompt_strategy is not None else ""
        qs = self.query_strategy.value if self.query_strategy is not None else ""

        return f"{self.__class__.__name__}({str(self.outputSchema):10s}, Model: {model}, Prompt Strategy: {ps}, Query Strategy: {qs}, Token Budget: {str(self.token_budget)})"

    def copy(self):
        return self.__class__(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            query_strategy=self.query_strategy,
            token_budget=self.token_budget,
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
            "model": self.model.value if self.model is not None else None,
            "prompt_strategy": (
                self.prompt_strategy.value if self.prompt_strategy is not None else None
            ),
            "query_strategy": (
                self.query_strategy.value if self.query_strategy is not None else None
            ),
            "token_budget": self.token_budget,
            "desc": str(self.desc),
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

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
        generate_field_names = []
        for field_name in outputSchema.fieldNames():
            if field_name not in inputSchema.fieldNames() and getattr(candidate, field_name, None) is None:
                generate_field_names.append(field_name)

        return generate_field_names

    def _construct_query_prompt(
        self,
        doc_type: str,
        generate_field_names: List[str],
        prompt_strategy: Optional[PromptStrategy] = None,
    ) -> str:
        """
        This function constructs the prompt for a bonded query.
        """
        # set defaults
        prompt_strategy = prompt_strategy if prompt_strategy is not None else self.prompt_strategy

        # build string of input fields and their descriptions
        multilineInputFieldDescription = ""
        for field_name in self.inputSchema.fieldNames():
            f = getattr(self.inputSchema, field_name)
            multilineInputFieldDescription += f"INPUT FIELD {field_name}: {f.desc}\n"

        # build string of output fields and their descriptions
        multilineOutputFieldDescription = ""
        for field_name in generate_field_names:
            f = getattr(self.outputSchema, field_name)
            multilineOutputFieldDescription += f"OUTPUT FIELD {field_name}: {f.desc}\n"

        # add input/output schema descriptions (if they have a docstring)
        optionalInputDesc = (
            ""
            if self.inputSchema.__doc__ is None
            else f"Here is a description of the input object: {self.inputSchema.__doc__}."
        )
        optionalOutputDesc = (
            ""
            if self.outputSchema.__doc__ is None
            else f"Here is a description of the output object: {self.outputSchema.__doc__}."
        )

        # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
        targetOutputDescriptor = (
            f"an output JSON object that describes an object of type {doc_type}."
        )
        outputSingleOrPlural = "the output object"
        appendixInstruction = "Be sure to emit a JSON object only"
        if self.cardinality == Cardinality.ONE_TO_MANY:
            targetOutputDescriptor = f"an output array of zero or more JSON objects that describe objects of type {doc_type}."
            outputSingleOrPlural = "the output objects"
            appendixInstruction = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields described above. You must decide the correct number of output objects."

        # construct promptQuestion
        promptQuestion = None
        if prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
            promptQuestion = (
                f"""I would like you to create {targetOutputDescriptor}. 
            You will use the information in an input JSON object that I will provide. The input object has type {self.inputSchema.className()}.
            All of the fields in {outputSingleOrPlural} can be derived using information from the input object.
            {optionalInputDesc}
            {optionalOutputDesc}
            Here is every input field name and a description: 
            {multilineInputFieldDescription}
            Here is every output field name and a description:
            {multilineOutputFieldDescription}
            {appendixInstruction}
            """
                + ""
                if self.desc is None
                else f" Keep in mind that this process is described by this text: {self.desc}."
            )

        else:
            promptQuestion = (
                f"""You are an image analysis bot. Analyze the supplied image(s) and create {targetOutputDescriptor}.
            You will use the information in the image that I will provide. The input image(s) has type {self.inputSchema.className()}.
            All of the fields in {outputSingleOrPlural} can be derived using information from the input image(s).
            {optionalInputDesc}
            {optionalOutputDesc}
            Here is every output field name and a description:
            {multilineOutputFieldDescription}
            {appendixInstruction}
            """
                + ""
                if self.desc is None
                else f" Keep in mind that this process is described by this text: {self.desc}."
            )

        # TODO: add this for boolean questions?
        # if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
        #     promptQuestion += "\nRemember, your output MUST be one of TRUE or FALSE."

        return promptQuestion

    def _create_record_op_stats_lst(
        self,
        records: List[DataRecord],
        fields: List[str],
        query_stats: StatsDict,
    ) -> List[RecordOpStats]:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        record_op_stats_lst = []
        for idx, dr in enumerate(records):
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                model_name=self.model.value,
                input_fields=self.inputSchema.fieldNames(),
                time_per_record= query_stats.get("total_time", 0.0) / len(records),
                cost_per_record= query_stats.get("cost_per_record", 0.0) / len(records),
                generated_fields=fields,
                total_input_tokens= query_stats.get("input_tokens", 0.0) / len(records),
                total_output_tokens=query_stats.get("output_tokens", 0.0) / len(records),
                total_input_cost=query_stats.get("input_cost", 0.0) / len(records),
                total_output_cost=query_stats.get("output_cost", 0.0) / len(records),
                llm_call_duration_secs=query_stats.get("llm_call_duration_secs", 0.0) / len(records),
                fn_call_duration_secs=query_stats.get("fn_call_duration_secs", 0.0) / len(records),
                answer= {field_name: getattr(dr, field_name) for field_name in fields}
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
        for field_name in candidate.getFields():
            setattr(dr, field_name, getattr(candidate, field_name))

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

    def _dspy_generate_fields(
        self,
        generate_field_names: List[str],
        content: Optional[Union[str, List[bytes]]] = None, #either text or image
        model: Optional[Model] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        token_budget: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[List[Dict[FieldName, Any]], StatsDict]:
        # set defaults
        model = model if model is not None else self.model
        prompt_strategy = prompt_strategy if prompt_strategy is not None else self.prompt_strategy
        token_budget = token_budget if token_budget is not None else self.token_budget

        # create DSPy generator and generate
        doc_schema = str(self.outputSchema)
        doc_type = self.outputSchema.className()
        promptQuestion = self._construct_query_prompt(
            doc_type=doc_type,
            generate_field_names=generate_field_names,
            prompt_strategy=prompt_strategy,
        )
        # generate LLM response and capture statistics
        answer, new_heatmap_json_obj, query_stats = None, None, {}
        try:
            if prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # invoke LLM to generate output JSON
                generator = DSPyGenerator(
                    model.value, prompt_strategy, doc_schema, doc_type, verbose
                )
                answer, new_heatmap_json_obj, query_stats = generator.generate(
                    content,
                    promptQuestion,
                    budget=token_budget,
                    heatmap_json_obj=self.heatmap_json_obj,
                )

            elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(model.value)
                answer, query_stats = generator.generate(content, promptQuestion)

            else:
                raise Exception(f"Prompt strategy not implemented: {prompt_strategy}")

        except Exception as e:
            print(f"DSPy generation error: {e}")
            return [{field_name: None for field_name in generate_field_names}], query_stats

        # if using token reduction, this will set the new heatmap (if not, it will just set it to None)
        self.heatmap_json_obj = new_heatmap_json_obj

        # parse the final json objects and standardize the outputs to be lists
        final_json_objects = []
        try:
            # parse JSON object from the answer
            jsonObj = getJsonFromAnswer(answer)

            # parse JSON output
            if self.cardinality == Cardinality.ONE_TO_MANY:
                assert isinstance(jsonObj["items"], list) and len(jsonObj["items"]) > 0, "No output objects were generated for one-to-many query"
                final_json_objects = jsonObj["items"]
            else:
                final_json_objects = [jsonObj]

            # TODO: in the future, do not perform this cleaning step if the field is a ListField
            # if value of field_name is a list; flatten the list
            for json_obj in final_json_objects:
                for field_name in generate_field_names:
                    while type(json_obj[field_name]) == type([]):
                        json_obj[field_name] = json_obj[field_name][0]

        except Exception as e:
            print(f"Error extracting json objects: {str(e)}")
            return [{field_name: None for field_name in generate_field_names}], query_stats

        return final_json_objects, query_stats

    def convert(self, candidate_content: Union[str,List[bytes]] , fields: List[str]) -> List[RecordJSONObjects]:
        """ This function is responsible for the LLM conversion process. 
        Different strategies may/should reimplement this function and leave the __call__ function untouched.
        The input is ...
        The output is a list of record JSON objects where each object is a dictionary with the (field, value) 
        pairs for each generated field."""
        raise NotImplementedError("This is an abstract class. Use a subclass instead!")

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        # capture start time
        start_time = time.time()

        # get the set of fields to generate
        fields_to_generate = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

        # get text or image content depending on prompt strategy
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            content = candidate._asJSONStr(include_bytes=False)
        elif self.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            base64_images = []
            if hasattr(candidate, "contents"):
                base64_images = [
                    base64.b64encode(candidate.contents).decode("utf-8")  # TODO: should address this now; we need a way to infer (or have the programmer declare) what fields contain image content
                ]
            else:
                base64_images = [
                    base64.b64encode(image).decode("utf-8")
                    for image in candidate.image_contents  # TODO: we should address this (see note above)
                ]
            content = base64_images
        else:
            raise Exception(f"Prompt strategy not implemented: {self.prompt_strategy}")

        marshal_time = time.time() - start_time
        field_outputs, query_stats = self.convert(fields=fields_to_generate, candidate_content=content)
        query_stats["total_time"] += marshal_time

        # compute the max number of outputs in any field (for one-to-many cardinality queries,
        # it's possible to have differing output lengths per field)
        field_max_outputs = 1
        if self.cardinality == Cardinality.ONE_TO_MANY:
            for field_name, json_objects in field_outputs.items():
                field_max_outputs = max(field_max_outputs, len(json_objects))

        # extend each field to have the same number of outputs
        for field_name, json_objects in field_outputs.items():
            while len(json_objects) < field_max_outputs:
                json_objects.append({field_name: None})

        # construct list of dictionaries where each dict. has the (field, value) pairs for each generated field
        # list is indexed per record
        n_records = max([len(lst) for lst in field_outputs.values()])
        records_json = [{} for _ in range(n_records)]
        for field in field_outputs:
            for idx, output in enumerate(field_outputs[field]):
                records_json[idx].update(output)

        drs = [
            self._create_data_record_from_json(
                jsonObj=js, candidate=candidate, cardinality_idx=idx
            )
            for idx, js in enumerate(records_json)
        ]

        record_op_stats_lst = self._create_record_op_stats_lst(
            records=drs,
            fields=fields_to_generate,
            query_stats=query_stats,
        )

        return drs, record_op_stats_lst


class LLMConvertConventional(LLMConvert):

    def convert(self, candidate_content: Union[str, List[bytes]], fields: List[str]) -> Tuple[List[RecordJSONObjects], StatsDict]:
        start_time = time.time()
        field_outputs, query_stats = {}, {}
        for field_name in fields:
            json_objects, field_stats = self._dspy_generate_fields([field_name], content=candidate_content)
            for key, value in field_stats.items():
                # TODO maybe a better way to find which stats to aggregate?
                if type(value) == type(''):
                    query_stats[key] = value
                elif type(value) in [type(1), type(1.)]:
                    query_stats[key] = query_stats.get(key, 0) + value
                elif type(value) == type(dict()):
                    for k2, v2 in value.items():
                        query_stats[k2] = query_stats.get(k2,0) + v2 # Should we simply throw the usage away here?
                else:
                    # TODO what to do with list fields like answer_log_probs? 
                    continue 
            field_outputs[field_name] = json_objects

        query_stats["total_time"] = time.time() - start_time
        return field_outputs, query_stats

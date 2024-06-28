from __future__ import annotations

from palimpzest import prompts
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
        return f"{self.model}_{self.query_strategy}"

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
        model = self.model.value if self.model is not None else ""
        ps = self.prompt_strategy.value if self.prompt_strategy is not None else ""
        qs = self.query_strategy.value if self.query_strategy is not None else ""

        return f"{self.__class__.__name__}({str(self.outputSchema):10s}, Model: {model}, Prompt Strategy: {ps}, Query Strategy: {qs})"

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
            "model": self.model.value if self.model is not None else None,
            "prompt_strategy": (
                self.prompt_strategy.value if self.prompt_strategy is not None else None
            ),
            "query_strategy": (
                self.query_strategy.value if self.query_strategy is not None else None
            ),
            "desc": str(self.desc),
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

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
            else prompts.OPTIONAL_OUTPUT_DESC.format(desc=self.inputSchema.__doc__)
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
        generation_stats: StatsDict,
    ) -> List[RecordOpStats]:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        record_op_stats_lst = []
        per_record_stats = generation_stats / len(records)
        for idx, dr in enumerate(records):
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                model_name=self.model.value,
                input_fields=self.inputSchema.fieldNames(),
                generated_fields=fields,
                answer= {field_name: getattr(dr, field_name) for field_name in fields},
                **per_record_stats,
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
        fields_to_generate: List[str],
        prompt: str,
        content: Optional[Union[str, List[bytes]]] = None, #either text or image
        verbose: bool = False,
    ) -> Tuple[List[Dict[FieldName, Any]], StatsDict]:
        """ This functions wraps the call to the generator method to actually perform the field generation.
        It returns a string which contains 
        """
        # create DSPy generator and generate
        doc_schema = str(self.outputSchema)
        doc_type = self.outputSchema.className()
        # generate LLM response and capture statistics
        answer, query_stats = None, {}
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
            return {field_name: None for field_name in fields_to_generate}, query_stats

        try:
            json_answer = getJsonFromAnswer(answer)
        except Exception as e:
            print(f"Error extracting json objects: {str(e)}")
            import pdb; pdb.set_trace()

            return {field_name: None for field_name in fields_to_generate}, query_stats

        return json_answer, query_stats

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

        marshal_time = time.time() - start_time
        fields_answers, fields_stats = self.convert(fields=fields_to_generate, candidate_content=content)

        # parse the final json objects and standardize the outputs to be lists
        field_outputs = {}
        query_stats = {}
        # construct list of dictionaries where each dict. has the (field, value) pairs for each generated field
        # list is indexed per record

        n_records = max([len(lst) for lst in fields_answers.values()])
        
        records_json = [{} for _ in range(n_records)]
        
        for field_name, field_answer in fields_answers.items():
            if self.cardinality == Cardinality.ONE_TO_MANY:
                assert isinstance(field_answer["items"], list) and len(field_answer["items"]) > 0, "No output objects were generated for one-to-many query"
                json_objects = field_answer["items"]
            else:
                json_objects = [field_answer]
            # NOTE: removed cleaning step

            for idx, output in enumerate(json_objects):
                import pdb; pdb.set_trace()
                records_json[idx].update(output)

            # TODO Aggregate field stats, maybe a better way?
            field_stat = fields_stats[field_name]
            for key, value in field_stat.items():
                if type(value) == type(dict()):
                    for k2, v2 in value.items():
                        # Should we simply throw the usage away here?
                        query_stats[k2] = query_stats.get(k2,type(v2)()) + v2 
                else:
                    query_stats[key] = query_stats.get(key, type(value)()) + value

        query_stats["total_time"] = time.time() - start_time

        # extend each record to have the same fields
        for record in records_json:
            record[field_name] = record.get(field_name, None)

        # TODO how does this work here?
        # compute the max number of outputs in any field (for one-to-many cardinality queries,
        # it's possible to have differing output lengths per field)
        field_max_outputs = 1
        if self.cardinality == Cardinality.ONE_TO_MANY:
            for field_name, json_objects in field_outputs.items():
                field_max_outputs = max(field_max_outputs, len(json_objects))


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
        fields_answers = {}
        fields_stats = {}

        for field_name in fields:
            prompt = self._construct_query_prompt(fields_to_generate=[field_name])
            json_answer, field_stats = self._dspy_generate_fields(
                fields_to_generate=[field_name],
                content=candidate_content,
                prompt=prompt,
            )          
            fields_answers[field_name] = json_answer
            fields_stats[field_name] = field_stats

        return fields_answers, fields_stats
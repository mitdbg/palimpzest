from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib import *
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.generators import CustomGenerator, DSPyGenerator, ImageTextGenerator, codeEnsembleExecution
from palimpzest.operators import logical, DataRecordsWithStats, PhysicalOperator
from palimpzest.utils import API, getJsonFromAnswer

from typing import Any, Dict, List, Optional, Tuple

import base64
import concurrent
import math
import time

# TYPE DEFINITIONS
FieldName = str
CodeName = str
Code = str
DataRecordDict = Dict[str, Any]
StatsDict = Dict[str, Any]
Exemplar = Tuple[DataRecordDict, DataRecordDict]
CodeEnsemble = Dict[CodeName, Code]

# CODE SYNTHESIS PROMPTS
EXAMPLE_PROMPT = """Example{idx}:
{example_inputs}
{example_output}
"""

CODEGEN_PROMPT = """You are a helpful programming assistant and an expert {language} programmer. Implement the {language} function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
{examples_desc}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
{advice}
Return the implementation only."""

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

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        query_strategy: Optional[QueryStrategy] = None,
        token_budget: Optional[float] = None,
        image_conversion: bool = False,
        no_cache_across_plans: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.query_strategy = query_strategy
        self.token_budget = token_budget
        self.image_conversion = image_conversion
        self.no_cache_across_plans = no_cache_across_plans

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
        return LLMConvert(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            model=self.model,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            prompt_strategy=self.prompt_strategy,
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

    def _create_empty_query_stats(self) -> StatsDict:
        """
        Creates an empty query stats object w/all the necessary keys.
        """
        return {
            "cost_per_record": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "llm_call_duration_secs": 0.0,
            "fn_call_duration_secs": 0.0,
        }


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
        data_records: List[DataRecord],
        query_stats_lst: List[StatsDict],
    ) -> List[RecordOpStats]:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        record_op_stats_lst = []
        for dr, stats in zip(data_records, query_stats_lst):
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=stats["time_per_record"],
                cost_per_record=stats["cost_per_record"],
                model_name=self.model.value,
                input_fields=self.inputSchema.fieldNames(),
                generated_fields=stats["generated_field_names"],
                total_input_tokens=stats["total_input_tokens"],
                total_output_tokens=stats["total_output_tokens"],
                total_input_cost=stats["total_input_cost"],
                total_output_cost=stats["total_output_cost"],
                llm_call_duration_secs=stats["llm_call_duration_secs"],
                fn_call_duration_secs=stats["fn_call_duration_secs"],
                answer=stats["answer"],
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

        # TODO: This inherits all pre-computed fields in an incremental fashion. The positive / pros
        #       of this approach is that it enables incremental schema computation, which tends to
        #       feel more natural for the end-user. The downside is it requires us to support an
        #       explicit projection to eliminate unwanted input / intermediate computation.
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


    def _extract_data_records_and_stats(
        self,
        candidate: DataRecord,
        start_time: float,
        generate_field_names: List[str],
        final_json_objects: List[Dict[str, Any]],
        query_stats: StatsDict,
    ) -> Tuple[List[DataRecord], List[StatsDict]]:
        # initialize list of output data records and stats
        drs, stats_lst = [], []

        # construct the lists of output data records and stats
        drs = []
        for idx, elt in enumerate(final_json_objects):
            # create output data record
            dr = self._create_data_record_from_json(
                jsonObj=elt,
                candidate=candidate,
                cardinality_idx=idx
            )
            drs.append(dr)

        # create output stats objects by amortizing runtime and cost across all output records
        stats_lst = []
        for dr in drs:
            stats = {
                "generated_fields": generate_field_names,
                "time_per_record": (time.time() - start_time) / len(drs),
                "cost_per_record": query_stats.get("cost_per_record", 0.0) / len(drs),
                "total_input_tokens": query_stats.get("input_tokens", 0.0) / len(drs),
                "total_output_tokens": query_stats.get("output_tokens", 0.0) / len(drs),
                "total_input_cost": query_stats.get("input_cost", 0.0) / len(drs),
                "total_output_cost": query_stats.get("output_cost", 0.0) / len(drs),
                "llm_call_duration_secs": query_stats.get("llm_call_duration_secs", 0.0) / len(drs),
                "fn_call_duration_secs": query_stats.get("fn_call_duration_secs", 0.0) / len(drs),
                "answer": {field_name: getattr(dr, field_name) for field_name in generate_field_names},
            }
            stats_lst.append(stats)

        return drs, stats_lst


    def _dspy_generate_fields(
        self,
        generate_field_names: List[str],
        text_content: Optional[str] = None,
        image_content: Optional[List[bytes]] = None,
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
        answer, new_heatmap_json_obj, query_stats = None, None, self._create_empty_query_stats()
        try:
            if prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # invoke LLM to generate output JSON
                generator = DSPyGenerator(
                    model.value, prompt_strategy, doc_schema, doc_type, verbose
                )
                answer, new_heatmap_json_obj, query_stats = generator.generate(
                    text_content,
                    promptQuestion,
                    budget=token_budget,
                    heatmap_json_obj=self.heatmap_json_obj,
                )

            elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(model.value)
                answer, query_stats = generator.generate(image_content, promptQuestion)

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


    def _fetch_input_output_exemplars(self) -> List[Exemplar]:
        # read the list of exemplars already generated by this operator if present
        if self.exemplars is not None:
            return self.exemplars

        # if we are allowed to cache exemplars across plan executions, check the cache
        if not self.no_cache_across_plans:
            cache = DataDirectory().getCacheService()
            exemplars_cache_id = self.get_op_id()
            exemplars = cache.getCachedData("codeExemplars", exemplars_cache_id)

            # set and return exemplars if it is not empty
            if exemplars is not None and len(exemplars) > 0:
                self.exemplars = exemplars
                return self.exemplars

        # otherwise, if there are no exemplars yet, create an empty list
        self.exemplars = []

        return self.exemplars


    def _synthesizeCondition(
        self,
        exemplars: List[Exemplar],
        strategy: CodeSynthStrategy=CodeSynthStrategy.SINGLE,
        num_exemplars: int=1,               # if strategy == EXAMPLE_ENSEMBLE
        code_regenerate_frequency: int=200, # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
    ) -> bool:
        if strategy == CodeSynthStrategy.NONE:
            return False
        elif strategy == CodeSynthStrategy.SINGLE:
            return not self.code_synthesized and len(exemplars) >= num_exemplars
        elif strategy == CodeSynthStrategy.EXAMPLE_ENSEMBLE:
            if len(exemplars) <= num_exemplars:
                return False
            return not self.code_synthesized
        elif strategy == CodeSynthStrategy.ADVICE_ENSEMBLE:
            return False
        elif strategy == CodeSynthStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
            return len(exemplars) % code_regenerate_frequency == 0
        else:
            raise Exception("not implemented yet")


    def _parse_multiple_outputs(self, text, outputs=['Thought', 'Action']):
        data = {}
        for key in reversed(outputs):
            if key+':' in text:
                remain, value = text.rsplit(key+':', 1)
                data[key.lower()] = value.strip()
                text = remain
            else:
                data[key.lower()] = None
        return data


    def _parse_ideas(self, text, limit=3):
        return self._parse_multiple_outputs(text, outputs=[f'Idea {i}' for i in range(1, limit+1)])


    def _generate_advice(self, prompt):
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        advs = self._parse_ideas(pred)
        return advs, stats


    def _synthesize_code(self, prompt, language='Python'):
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        ordered_keys = [
            f'```{language}',
            f'```{language.lower()}',
            f'```'
        ]
        code = None
        for key in ordered_keys:
            if key in pred:
                code = pred.split(key)[1].split('```')[0].strip()
                break
        return code, stats


    def _code_synth_default(self, api):
        # returns a function with the correct signature which simply returns None
        code = api.api_def() + "  return None\n"
        stats = self._create_empty_query_stats()
        return code, stats


    def _code_synth_single(self, api: API, output_field_name: str, exemplars: List[Exemplar]=list(), advice: str=None, language='Python'):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[0][field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[1][output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'advice': f"Hint: {advice}" if advice else "",
        }
        prompt = CODEGEN_PROMPT.format(**context)
        print("PROMPT")
        print("-------")
        print(f"{prompt}")
        code, stats = self._synthesize_code(prompt, language=language)
        print("-------")
        print("SYNTHESIZED CODE")
        print("---------------")
        print(f"{code}")

        return code, stats


    def _synthesize_advice(self, api: API, output_field_name: str, exemplars: List[Dict[DataRecord, DataRecord]]=list(), language='Python', n_advices=4):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[0][field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[1][output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'n': n_advices,
        }
        prompt = ADVICEGEN_PROMPT.format(**context)
        advs, stats = self._generate_advice(prompt)
        return advs, stats


    def _synthesize_code_ensemble(
        self,
        api: API,
        output_field_name: str,
        exemplars: List[Exemplar]=list(),
        strategy: CodeSynthStrategy=CodeSynthStrategy.SINGLE,
        code_ensemble_num: int=1,       # if strategy != SINGLE
        num_exemplars: int=1,           # if strategy != EXAMPLE_ENSEMBLE
    ) -> Tuple[Dict[CodeName, Code], StatsDict]:
        code_ensemble = dict()
        if strategy == CodeSynthStrategy.NONE:
            # create an ensemble with one function which returns None
            code, code_synth_stats = self._code_synth_default(api)
            code_name = f"{api.name}_v0"
            code_ensemble[code_name] = code
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.SINGLE:
            # use exemplars to create an ensemble with a single synthesized function
            code, code_synth_stats = self._code_synth_single(api, output_field_name, exemplars=exemplars[:num_exemplars])
            code_name = f"{api.name}_v0"
            code_ensemble[code_name] = code
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.EXAMPLE_ENSEMBLE:
            # creates an ensemble of `code_ensemble_num` synthesized functions; each of
            # which uses a different exemplar (modulo the # of exemplars) for its synthesis
            code_synth_stats = self._create_empty_query_stats()
            for i in range(code_ensemble_num):
                code_name = f"{api.name}_v{i}"
                exemplar = exemplars[i % len(exemplars)]
                code, stats = self._code_synth_single(api, output_field_name, exemplars=[exemplar])
                code_ensemble[code_name] = code
                for key in code_synth_stats.keys():
                    code_synth_stats[key] += stats[key]
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.ADVICE_ENSEMBLE:
            # a more advanced approach in which advice is first solicited, and then
            # provided as context when synthesizing the code ensemble
            code_synth_stats = self._create_empty_query_stats()

            # solicit advice
            advices, adv_stats = self._synthesize_advice(api, output_field_name, exemplars=exemplars[:num_exemplars], n_advices=code_ensemble_num)
            for key in code_synth_stats.keys():
                code_synth_stats[key] += adv_stats[key]

            # synthesize code ensemble
            for i, adv in enumerate(advices):
                code_name = f"{api.name}_v{i}"
                code, stats = self._code_synth_single(api, output_field_name, exemplars=exemplars[:num_exemplars], advice=adv)
                code_ensemble[code_name] = code
                for key in code_synth_stats.keys():
                    code_synth_stats[key] += stats[key]
            return code_ensemble, code_synth_stats

        else:
            raise Exception("not implemented yet")


    def _fetch_or_synthesize_code_ensemble(
        self,
        generate_field_names: List[str],
        candidate_dict: DataRecordDict,
        synthesize: bool = False,
        exemplars: List[Exemplar] = [],
    ) -> Optional[Tuple[Dict[FieldName, CodeEnsemble], StatsDict]]:
        # if we're not (re-)generating the code; try reading cached code
        if not synthesize:
            # read the dictionary of ensembles already synthesized by this operator if present
            if self.field_to_code_ensemble is not None:
                return self.field_to_code_ensemble, {}

            # if we are allowed to cache synthesized code across plan executions, check the cache
            if not self.no_cache_across_plans:
                field_to_code_ensemble = {}
                cache = DataDirectory().getCacheService()
                for field_name in generate_field_names:
                    code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
                    code_ensemble = cache.getCachedData("codeEnsembles", code_ensemble_cache_id)
                    if code_ensemble is not None:
                        field_to_code_ensemble[field_name] = code_ensemble

                # set and return field_to_code_ensemble if all fields are present and have code
                if all([field_to_code_ensemble.get(field_name, None) is not None for field_name in generate_field_names]):
                    self.field_to_code_ensemble = field_to_code_ensemble
                    return self.field_to_code_ensemble, {}

            # if we're not synthesizing new code ensemble(s) and there is nothing to fetch, return None
            return None

        # initialize stats to be collected for each field's code sythesis
        total_code_synth_stats = self._create_empty_query_stats()

        # synthesize the per-field code ensembles
        field_to_code_ensemble = {}
        for field_name in generate_field_names:
            # create api instance
            api = API.from_input_output_schemas(
                inputSchema=self.inputSchema,
                outputSchema=self.outputSchema,
                field_name=field_name,
                input_fields=candidate_dict.keys()
            )

            # synthesize the code ensemble
            code_ensemble, code_synth_stats = self._synthesize_code_ensemble(api, field_name, exemplars)

            # update stats
            for key in total_code_synth_stats.keys():
                total_code_synth_stats[key] += code_synth_stats[key]

            # add synthesized code ensemble to field_to_code_ensemble
            field_to_code_ensemble[field_name] = code_ensemble

            # add code ensemble to the cache
            if not self.no_cache_across_plans:
                cache = DataDirectory().getCacheService()
                code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
                cache.putCachedData("codeEnsembles", code_ensemble_cache_id, code_ensemble)

            # TODO: if verbose
            for code_name, code in code_ensemble.items():
                print(f"CODE NAME: {code_name}")
                print("-----------------------")
                print(code)

        # set field_to_code_ensemble and code_synthesized to True
        self.field_to_code_ensemble = field_to_code_ensemble
        self.code_synthesized = True

        return field_to_code_ensemble, total_code_synth_stats


    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        # capture start time
        start_time = time.time()

        if self.query_strategy == QueryStrategy.BONDED_WITH_FALLBACK:
            # get the set of fields to generate
            generate_field_names = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

            # get text or image content depending on prompt strategy
            dspy_generate_fields_kwargs = {}
            if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                dspy_generate_fields_kwargs = {"text_content": candidate._asJSONStr(include_bytes=False)}

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
                dspy_generate_fields_kwargs = {"image_content": base64_images}

            else:
                raise Exception(f"Prompt strategy not implemented: {self.prompt_strategy}")

            # generate all fields in a single query
            final_json_objects, query_stats = self._dspy_generate_fields(generate_field_names, **dspy_generate_fields_kwargs)

            # if there was an error, execute a conventional query
            if all([v is None for v in final_json_objects[0].values()]):
                # generate each field one at a time
                field_outputs = {}
                for field_name in generate_field_names:
                    json_objects, field_stats = self._dspy_generate_fields([field_name], **dspy_generate_fields_kwargs)

                    # update query_stats
                    for key, value in field_stats.items():
                        query_stats[key] += value

                    # update field_outputs
                    field_outputs[field_name] = json_objects

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
                final_json_objects = []
                for idx in range(field_max_outputs):
                    output_fields_dict = {}
                    for field_name, json_objects in field_outputs.items():
                        output_fields_dict[field_name] = json_objects[field_name][idx]

                    final_json_objects.append(output_fields_dict)

            # construct the set of output data records and record_op_stats
            drs, query_stats_lst = self._extract_data_records_and_stats(candidate, start_time, generate_field_names, final_json_objects, query_stats)

            # compute the record_op_stats for each data record and return
            record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

            return drs, record_op_stats_lst

        elif self.query_strategy == QueryStrategy.CONVENTIONAL:
            # get the set of fields to generate
            generate_field_names = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

            # get text or image content depending on prompt strategy
            dspy_generate_fields_kwargs = {}
            if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                dspy_generate_fields_kwargs = {"text_content": candidate._asJSONStr(include_bytes=False)}

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
                dspy_generate_fields_kwargs = {"image_content": base64_images}

            else:
                raise Exception(f"Prompt strategy not implemented: {self.prompt_strategy}")

            # generate each field one at a time
            field_outputs, query_stats = {}, self._create_empty_query_stats()
            for field_name in generate_field_names:
                json_objects, field_stats = self._dspy_generate_fields([field_name], **dspy_generate_fields_kwargs)

                # update query_stats
                for key, value in field_stats.items():
                    query_stats[key] += value

                # update field_outputs
                field_outputs[field_name] = json_objects

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
            final_json_objects = []
            for idx in range(field_max_outputs):
                output_fields_dict = {}
                for field_name, json_objects in field_outputs.items():
                    output_fields_dict[field_name] = json_objects[field_name][idx]

                final_json_objects.append(output_fields_dict)

            # construct the set of output data records and record_op_stats
            drs, query_stats_lst = self._extract_data_records_and_stats(candidate, start_time, generate_field_names, final_json_objects, query_stats)

            # compute the record_op_stats for each data record and return
            record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

            return drs, record_op_stats_lst

        elif self.query_strategy == QueryStrategy.CODE_GEN_WITH_FALLBACK:
            # get the set of fields to generate
            generate_field_names = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

            # get any existing (input, output) example pairs for each field
            exemplars = self._fetch_input_output_exemplars()

            # check whether we need to (re-)generate our code ensembles
            synthesize = self._synthesizeCondition(exemplars=exemplars)

            # convert the data record to a dictionary of field --> value
            # NOTE: the following is how we used to compute the candidate_dict;
            #       now that I am disallowing code synthesis for one-to-many queries,
            #       I don't think we need to invoke the _asJSONStr() method, which
            #       helped format the tabular data in the "rows" column for Medical Schema Matching.
            #       In the longer term, we should come up with a proper solution to make _asDict()
            #       properly format data which relies on the schema's _asJSONStr method.
            #
            #   candidate_dict_str = candidate._asJSONStr(include_bytes=False, include_data_cols=False)
            #   candidate_dict = json.loads(candidate_dict_str)
            #   candidate_dict = {k: v for k, v in candidate_dict.items() if v != "<bytes>"}
            candidate_dict = candidate._asDict(include_bytes=False)

            # fetch or synthesize code ensemble for code synthesis
            synthesize_output = self._fetch_or_synthesize_code_ensemble(generate_field_names, candidate_dict, synthesize, exemplars)

            # if we have yet to synthesize code (perhaps b/c we are waiting for more exemplars),
            # use GPT-4 to perform the convert (and generate high-quality exemplars) using a bonded query
            if synthesize_output is None:
                text_content = json.loads(candidate_dict)
                final_json_objects, query_stats = self._dspy_generate_fields(
                    generate_field_names,
                    text_content=text_content,
                    model=Model.GPT_4,  # TODO: assert GPT-4 is available; maybe fall back to another model otherwise
                    prompt_strategy=PromptStrategy.DSPY_COT_QA,
                )

                # construct the set of output data records and record_op_stats
                drs, query_stats_lst = self._extract_data_records_and_stats(candidate, start_time, generate_field_names, final_json_objects, query_stats)

                # compute the record_op_stats for each data record and return
                record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

                # NOTE: this now includes bytes input fields which will show up as: `field_name = "<bytes>"`;
                #       keep an eye out for a regression in code synth performance and revert if necessary
                # update operator's set of exemplars
                exemplars = [dr._asDict(include_bytes=False) for dr in drs] # TODO: need to extend candidate to same length and zip
                self.exemplars.extend(exemplars)

                # if we are allowed to cache exemplars across plan executions, add exemplars to cache
                if not self.no_cache_across_plans:
                    cache = DataDirectory().getCacheService()
                    exemplars_cache_id = self.get_op_id()
                    cache.putCachedData(f"codeExemplars", exemplars_cache_id, exemplars)

                return drs, record_op_stats_lst

            # extract output from call to synthesize ensemble
            field_to_code_ensemble, total_code_synth_stats = synthesize_output

            # add total_code_synth_stats to query_stats
            query_stats = self._create_empty_query_stats()
            for key in total_code_synth_stats:
                query_stats[key] += total_code_synth_stats

            # if we have synthesized code run it on each field
            field_outputs = {}
            for field_name in generate_field_names:
                # create api instance for executing python code
                api = API.from_input_output_schemas(
                    inputSchema=self.inputSchema,
                    outputSchema=self.outputSchema,
                    field_name=field_name,
                    input_fields=candidate_dict.keys()
                )
                code_ensemble = field_to_code_ensemble[field_name]
                answer, exec_stats = codeEnsembleExecution(api, code_ensemble, candidate_dict)

                if answer is not None:
                    field_outputs[field_name] = answer
                    for key, value in exec_stats.items():
                        query_stats[key] += value
                else:
                    # if there is a failure, run a conventional query
                    print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                    text_content = json.loads(candidate_dict)
                    final_json_objects, field_stats = self._dspy_generate_fields(
                        [field_name],
                        text_content=text_content,
                        model=Model.GPT_3_5,
                        prompt_strategy=PromptStrategy.DSPY_COT_QA,
                    )

                    # include code execution time in field_stats
                    if "fn_call_duration_secs" not in field_stats:
                        field_stats["fn_call_duration_secs"] = 0.0
                    field_stats["fn_call_duration_secs"] += exec_stats["fn_call_duration_secs"]

                    # update query_stats
                    for key, value in field_stats.items():
                        query_stats[key] += value

                    # NOTE: we disallow code synth for one-to-many queries, so there will only be
                    #       one element in final_json_objects
                    # update field_outputs
                    field_outputs[field_name] = final_json_objects[0][field_name]

            # construct the set of output data records and record_op_stats
            drs, query_stats_lst = self._extract_data_records_and_stats(candidate, start_time, generate_field_names, [field_outputs], query_stats)

            # compute the record_op_stats for each data record and return
            record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

            return drs, record_op_stats_lst

        else:
            raise ValueError(f"Unimplemented QueryStrategy: {self.query_strategy.value}")

from palimpzest.datamanager import DataDirectory
from palimpzest.constants import PromptStrategy, CodeGenStrategy, Model
from palimpzest.elements import DataRecord
from palimpzest.generators import (
    DSPyGenerator,
    ImageTextGenerator,
    codeEnsembleGeneration,
    codeEnsembleExecution,
    reGenerationCondition,
)
from palimpzest.dataclasses import *
from palimpzest.utils import API, getJsonFromAnswer

from typing import Any, Dict, List, Tuple

import base64
import json
import re


def _construct_query_prompt(
    doc_type: str,
    inputSchema,
    outputSchema,
    cardinality: str,
    prompt_strategy: PromptStrategy,
    conversionDesc: str,
    generate_field_names: List[str],
    is_conventional: bool = False,
) -> str:
    """
    This function constructs the prompt for a bonded query.
    """
    # build string of input fields and their descriptions
    multilineInputFieldDescription = ""
    for field_name in inputSchema.fieldNames():
        f = getattr(inputSchema, field_name)
        multilineInputFieldDescription += f"INPUT FIELD {field_name}: {f.desc}\n"

    # build string of output fields and their descriptions
    multilineOutputFieldDescription = ""
    for field_name in generate_field_names:
        f = getattr(outputSchema, field_name)
        multilineOutputFieldDescription += f"OUTPUT FIELD {field_name}: {f.desc}\n"

    # add input/output schema descriptions (if they have a docstring)
    optionalInputDesc = (
        ""
        if inputSchema.__doc__ is None
        else f"Here is a description of the input object: {inputSchema.__doc__}."
    )
    optionalOutputDesc = (
        ""
        if outputSchema.__doc__ is None
        else f"Here is a description of the output object: {outputSchema.__doc__}."
    )

    # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
    targetOutputDescriptor = (
        f"an output JSON object that describes an object of type {doc_type}."
    )
    outputSingleOrPlural = "the output object"
    appendixInstruction = "Be sure to emit a JSON object only"
    if cardinality == "oneToMany":
        targetOutputDescriptor = f"an output array of zero or more JSON objects that describe objects of type {doc_type}."
        outputSingleOrPlural = "the output objects"
        appendixInstruction = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields described above. You must decide the correct number of output objects."

    # if this is a conventional query, focus only on generating output field
    if is_conventional:
        targetOutputDescriptor = f'an output JSON object with a single key "{generate_field_names[0]}" whose value is specified in the input object.'

    # construct promptQuestion
    promptQuestion = None
    if prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
        promptQuestion = (
            f"""I would like you to create {targetOutputDescriptor}. 
        You will use the information in an input JSON object that I will provide. The input object has type {inputSchema.className()}.
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
            if conversionDesc is None
            else f" Keep in mind that this process is described by this text: {conversionDesc}."
        )

    else:
        promptQuestion = (
            f"""You are an image analysis bot. Analyze the supplied image(s) and create {targetOutputDescriptor} {doc_type}.
        You will use the information in the image that I will provide. The input image(s) has type {inputSchema.className()}.
        All of the fields in {outputSingleOrPlural} can be derived using information from the input image(s).
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}
        {appendixInstruction}
        """
            + ""
            if conversionDesc is None
            else f" Keep in mind that this process is described by this text: {conversionDesc}."
        )

    # TODO: add this for boolean questions?
    # if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
    #     promptQuestion += "\nRemember, your output MUST be one of TRUE or FALSE."

    return promptQuestion


def _create_data_record_from_json(
    jsonObj: Any, 
    inputSchema,
    outputSchema,
    candidate: DataRecord, cardinality_idx: int = None
) -> DataRecord:
    # initialize data record
    dr = DataRecord(
        outputSchema, parent_uuid=candidate._uuid, cardinality_idx=cardinality_idx
    )

    # get input field names and output field names
    input_fields = inputSchema.fieldNames()
    output_fields = outputSchema.fieldNames()

    # first, copy all fields from input schema
    for field_name in input_fields:
        setattr(dr, field_name, getattr(candidate, field_name))

    # parse newly generated fields from the generated jsonObj
    for field_name in output_fields:
        if field_name not in input_fields:
            # parse the json object and set the DataRecord's fields with their generated values
            setattr(
                dr, field_name, jsonObj.get(field_name, None)
            )  # the use of get prevents a KeyError if an individual field is missing. TODO: is this behavior desired?

    return dr


def runBondedQuery(
    candidate: DataRecord,
    inputSchema,
    outputSchema,
    cardinality: str,
    prompt_strategy: PromptStrategy,
    model: Model,
    token_budget: float,
    conversionDesc: str,
    heatmap_json_obj: Dict[str, Any],
    verbose: bool = False
) -> Tuple[List[DataRecord], RecordOpStats, str]:
    """
    Run a bonded query, in which all new fields in the outputSchema are generated simultaneously
    in a single LLM call. This is in contrast to a conventional query, in which each output field
    is generated using its own LLM call.

    At the moment, tasks with cardinality == "oneToMany" can only be executed using bonded queries.

    This is not a theoretical limitation of conventional queries, but there are some practical
    difficulties with guaranteeing that each field has the same number of outputs generated
    in each separate LLM invocation.
    """
    # initialize list of output data records
    drs = []

    # construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in outputSchema.fieldNames():
        if field_name not in inputSchema.fieldNames():
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate._asJSON(include_bytes=False)
    doc_schema = str(outputSchema)
    doc_type = outputSchema.className()

    # construct prompt question
    promptQuestion = _construct_query_prompt(
        doc_type=doc_type,
        inputSchema=inputSchema,
        outputSchema=outputSchema,
        cardinality=cardinality,
        prompt_strategy=prompt_strategy,
        conversionDesc=None,
        generate_field_names=generate_field_names,
    )

    # generate LLM response and capture statistics
    answer, new_heatmap_json_obj, bonded_query_stats = None, None, None
    try:
        if prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # invoke LLM to generate output JSON
            generator = DSPyGenerator(
                model.value, prompt_strategy, doc_schema, doc_type, verbose
            )
            answer, new_heatmap_json_obj, gen_stats = generator.generate(
                text_content,
                promptQuestion,
                budget=token_budget,
                heatmap_json_obj=heatmap_json_obj,
            )

            # construct BondedQueryStats object
            bonded_query_stats = RecordOpStats(
                record_uuid=candidate._uuid,
                record_parent_uuid=candidate._parent_uuid,
                op_id="LLMConvert_eeaca",
                op_name="LLMConvert",
                op_time=gen_stats['llm_call_duration_secs'],
                op_cost=gen_stats['op_cost'],
                record_state= gen_stats,
                model_name=model.value,
                input_fields_str=inputSchema.fieldNames(),
                generated_fields_str=generate_field_names,
                total_input_tokens=gen_stats['input_tokens'],
                total_output_tokens=gen_stats['output_tokens'],
                total_input_cost=gen_stats['input_cost'],
                total_output_cost=gen_stats['output_cost'],
                answer=answer
            )

        elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            # TODO: this is very hacky; need to come up w/more general solution for multimodal schemas
            # b64 decode of candidate.contents or candidate.image_contents
            base64_images = []
            if hasattr(candidate, "contents"):
                base64_images = [base64.b64encode(candidate.contents).decode("utf-8")]
            else:
                base64_images = [
                    base64.b64encode(image).decode("utf-8")
                    for image in candidate.image_contents
                ]

            # invoke LLM to generate output JSON
            generator = ImageTextGenerator(model.value)
            answer, gen_stats = generator.generate(base64_images, promptQuestion)

            # construct BondedQueryStats object
            bonded_query_stats = RecordOpStats(
                record_uuid=candidate._uuid,
                record_parent_uuid=candidate._parent_uuid,
                op_id="bonded_query_123",
                op_name="bonded_query",
                op_time=gen_stats['op_time'],
                op_cost=gen_stats['op_cost'],
                record_stats= gen_stats,
                model_name=model.value,
                input_fields_str=inputSchema.fieldNames(),
                generated_fields_str=generate_field_names,
                total_input_tokens=gen_stats['input_tokens'],
                total_output_tokens=gen_stats['output_tokens'],
                total_input_cost=gen_stats['input_cost'],
                total_output_cost=gen_stats['output_cost'],
                answer=answer
            )

        # TODO
        elif prompt_strategy == PromptStrategy.ZERO_SHOT:
            raise Exception("not implemented yet")

        # TODO
        elif prompt_strategy == PromptStrategy.FEW_SHOT:
            raise Exception("not implemented yet")
        else:
            raise Exception(f"Prompt strategy not implemented: {prompt_strategy}")
    except Exception as e:
        print(f"Bonded query processing error: {e}")
        return None, new_heatmap_json_obj, bonded_query_stats, str(e)

    try:
        # parse JSON object from the answer
        jsonObj = getJsonFromAnswer(answer)

        # parse JSON output and construct data records
        if cardinality == "oneToMany":
            if len(jsonObj["items"]) == 0:
                raise Exception(
                    "No output objects were generated with bonded query - trying with conventional query..."
                )
            for idx, elt in enumerate(jsonObj["items"]):
                dr = _create_data_record_from_json(
                    jsonObj=elt,
                    inputSchema=inputSchema,
                    outputSchema=outputSchema,
                    candidate=candidate,
                    cardinality_idx=idx
                )
                drs.append(dr)
        else:
            dr = _create_data_record_from_json(
                    jsonObj=jsonObj,
                    inputSchema=inputSchema,
                    outputSchema=outputSchema,
                    candidate=candidate,
            )
            drs = [dr]
    except Exception as e:
        print(f"Parsing answer error: {e}")
        return None, new_heatmap_json_obj, bonded_query_stats, str(e)


    # # TODO: debug root cause
    # for dr in drs:
    #     if not hasattr(dr, 'filename'):
    #         setattr(dr, 'filename', candidate.filename)

    return drs, new_heatmap_json_obj, bonded_query_stats, None


# NOTE: temporary to have running code. Refactor this out into strategies
def runConventionalQuery(
    candidate: DataRecord,
    inputSchema,
    outputSchema,
    cardinality: str,
    prompt_strategy: PromptStrategy,
    model: Model,
    token_budget: float,
    conversionDesc: str,
    verbose: bool = False
) -> Tuple[DataRecord, RecordOpStats]:
    """
    Run a conventional query, in which each output field is generated using its own LLM call.

    At the moment, conventional queries cannot execute tasks with cardinality == "oneToMany".
    """
    # initialize output data record
    dr = DataRecord(outputSchema, parent_uuid=candidate._uuid)

    # copy fields from the candidate (input) record if they already exist
    # and construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in outputSchema.fieldNames():
        if field_name in inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        elif hasattr(candidate, field_name) and (
            getattr(candidate, field_name) is not None
        ):
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate._asJSON(include_bytes=False)
    doc_schema = str(outputSchema)
    doc_type = outputSchema.className()

    if cardinality == "oneToMany":
        # TODO here the problem is: which is the 1:N field that we are splitting the output into?
        # do we need to know this to construct the prompt question ?
        # for now, we will just assume there is only one list in the JSON.
        dct = json.loads(text_content)
        split_attribute = [att for att in dct.keys() if type(dct[att]) == list][0]
        n_splits = len(dct[split_attribute])

        query_stats = dict()
        if prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # TODO Hacky to nest return and not disrupt the rest of method!!!
            # NOTE: this is a bonded query, but we are treating it as a conventional query
            promptQuestion = _construct_query_prompt(
                doc_type=doc_type,
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                cardinality=cardinality,
                prompt_strategy=prompt_strategy,
                conversionDesc=conversionDesc,
                generate_field_names=generate_field_names)
        
        field_stats = None
        try:
            field_stats = None
            if prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # print(f"FALL BACK FIELD: {field_name}")
                # print("---------------")
                # invoke LLM to generate output JSON
                generator = DSPyGenerator(
                    model.value, prompt_strategy, doc_schema, doc_type, verbose
                )
                answer, field_stats = generator.generate(
                    text_content,
                    promptQuestion,
                    budget=token_budget,
                )

            elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
                # TODO: this is very hacky; need to come up w/more general solution for multimodal schemas
                # b64 decode of candidate.contents or candidate.image_contents
                base64_images = []
                if hasattr(candidate, "contents"):
                    base64_images = [
                        base64.b64encode(candidate.contents).decode("utf-8")
                    ]
                else:
                    base64_images = [
                        base64.b64encode(image).decode("utf-8")
                        for image in candidate.image_contents
                    ]

                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(model.value)
                answer, field_stats = generator.generate(base64_images, promptQuestion)

            # TODO
            elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                raise Exception("not implemented yet")

            # TODO
            elif prompt_strategy == PromptStrategy.FEW_SHOT:
                raise Exception("not implemented yet")

            # update query_stats
            query_stats[f"{field_name}"] = field_stats

            # extract result from JSON and set the DataRecord's field with its generated value
            jsonObj = getJsonFromAnswer(answer)
            setattr(dr, field_name, jsonObj[field_name])

        except Exception as e:
            print(f"Conventional field processing error: {e}")
            setattr(dr, field_name, None)
            query_stats[f"{field_name}"] = field_stats
    else:
        query_stats = dict()
        for field_name in generate_field_names:
            # construct prompt question
            promptQuestion = _construct_query_prompt(
                doc_type=doc_type,
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                cardinality=cardinality,
                prompt_strategy=prompt_strategy,
                conversionDesc=conversionDesc,
                generate_field_names=[field_name]
            )

            try:
                field_stats = None
                if prompt_strategy == PromptStrategy.DSPY_COT_QA:
                    # invoke LLM to generate output JSON
                    generator = DSPyGenerator(model.value, prompt_strategy, doc_schema, doc_type, verbose)
                    answer, field_stats = generator.generate(text_content, promptQuestion)

                elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:                               
                    # b64 decode of candidate.contents
                    image_b64 = base64.b64encode(candidate.contents).decode('utf-8')
                    # invoke LLM to generate output JSON
                    generator = ImageTextGenerator(model.value)
                    answer, field_stats = generator.generate(image_b64, promptQuestion)

                else:
                    raise Exception("not implemented yet")

                # set the DataRecord's field with its generated value
                setattr(dr, field_name, answer)
                # update query_stats
                query_stats[f"{field_name}"] = field_stats

            except Exception as e:
                print(f"Conventional field processing error: {e}")
                setattr(dr, field_name, None)
                query_stats[f"{field_name}"] = None

    op_time = sum([gen_stats['op_time'] for gen_stats in query_stats.values()])
    op_cost = sum([gen_stats['op_cost'] for gen_stats in query_stats.values()])

    conventional_query_stats = RecordOpStats(
        record_uuid=candidate._uuid,
        record_parent_uuid=candidate._parent_uuid,
        op_id="conventional_query_123",
        op_name="conventional__query",
        op_time=op_time,
        op_cost=op_cost,
        model_name=model.value,
        input_fields_str=inputSchema.fieldNames(),
        generated_fields_str=generate_field_names,
        total_input_tokens=sum([q['input_tokens'] for q in query_stats.values()]),
        total_output_tokens=sum([q['output_tokens'] for q in query_stats.values()]),
        total_input_cost=sum([q['input_cost'] for q in query_stats.values()]),
        total_output_cost=sum([q['output_cost'] for q in query_stats.values()]),
        answer=answer

    )

    # # TODO: debug root cause
    # if not hasattr(dr, 'filename'):
    #     setattr(dr, 'filename', candidate.filename)

    return dr, conventional_query_stats


def runCodeGenQuery(
    candidate: DataRecord, 
    inputSchema,
    outputSchema,
    cardinality: str,
    prompt_strategy: PromptStrategy,
    model: Model,
    op_id: str,
    plan_idx: int,
    conversionDesc: str,
    token_budget: float,
    verbose: bool = False) -> Tuple[DataRecord, RecordOpStats]:
    """
    I think this would roughly map to the internals of _makeCodeGenTypeConversionFn() in your branch.
    Similar to the functions above, I moved most of the details of generating responses
    """
    # initialize output data record
    dr = DataRecord(outputSchema, parent_uuid=candidate._uuid)

    # copy fields from the candidate (input) record if they already exist
    # and construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in outputSchema.fieldNames():
        if field_name in inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    if cardinality == "oneToMany":
        # TODO here the problem is: which is the 1:N field that we are splitting the output into?
        # do we need to know this to construct the prompt question ?
        # for now, we will just assume there is only one list in the JSON.
        dct = candidate._asJSON(include_bytes=False, include_data_cols=False)
        dct = json.loads(dct)
        split_attribute = [att for att in dct.keys() if type(dct[att]) == list][0]
        n_splits = len(dct[split_attribute])

        # TODO Hacky to nest return and not disrupt the rest of method!!!
        # NOTE: this is a bonded query, but we are treating it as a conventional query
        drs = []
        full_code_gen_stats, conv_query_stats = RecordOpStats(), {}
        for idx in range(n_splits):
            # initialize output data record
            dr = DataRecord(
                outputSchema, parent_uuid=candidate._uuid, cardinality_idx=idx
            )

            cache = DataDirectory().getCacheService()
            for field_name in generate_field_names:
                code_ensemble_id = "_".join([op_id, field_name])
                cached_code_ensemble_info = cache.getCachedData(
                    f"codeEnsemble{plan_idx}", code_ensemble_id
                )

                gen_stats = None
                if cached_code_ensemble_info is not None:
                    code_ensemble, _ = cached_code_ensemble_info
                    examples = cache.getCachedData(
                        f"codeSamples{plan_idx}", code_ensemble_id
                    )
                else:
                    code_ensemble, examples = dict(), list()

                if verbose:
                    print(f"Processing {split_attribute} with index {idx}")
                new_json = {k: v for k, v in dct.items() if k != split_attribute}
                new_json[split_attribute] = dct[split_attribute][idx]

                # examples.append(new_json)
                candidate_dicts = []
                for _idx in range(n_splits):
                    candidate_dict = dict(new_json)
                    candidate_dict[split_attribute] = dct[split_attribute][_idx]
                    candidate_dicts.append(candidate_dict)

                # print(type(candidate_dicts))
                # print(candidate_dicts)
                examples.extend(candidate_dicts)
                cache.putCachedData(
                    f"codeSamples{plan_idx}", code_ensemble_id, examples
                )
                api = API.from_inout_schema(
                    inputSchema=inputSchema, outputSchema=outputSchema, field_name=field_name, input_fields=candidate_dict.keys()
                )
                if len(code_ensemble) == 0 or reGenerationCondition(
                    api, examples=examples
                ):
                    code_ensemble, gen_stats = codeEnsembleGeneration(
                        api, examples=examples, code_num_examples=n_splits
                    )
                    cache.putCachedData(
                        f"codeEnsemble{plan_idx}",
                        code_ensemble_id,
                        (code_ensemble, gen_stats),
                    )

                for code_name, code in code_ensemble.items():
                    print(f"CODE NAME: {code_name}")
                    print("-----------------------")
                    print(code)

                answer, exec_stats = codeEnsembleExecution(api, code_ensemble, new_json)
                full_code_gen_stats.code_gen_stats[f"{field_name}_{idx}"] = gen_stats
                full_code_gen_stats.code_exec_stats[f"{field_name}_{idx}"] = exec_stats

                if answer is None:
                    print(
                        f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}"
                    )
                    # construct prompt question
                    doc_schema = str(outputSchema)
                    doc_type = outputSchema.className()
                    promptQuestion = _construct_query_prompt(
                        doc_type=doc_type,
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        cardinality=cardinality,
                        prompt_strategy=prompt_strategy,
                        conversionDesc=conversionDesc,
                        generate_field_names=[field_name],
                        is_conventional=False
                    )
                    field_stats = None
                    try:
                        # print(f"FALL BACK FIELD: {field_name}")
                        # print("---------------")
                        # invoke LLM to generate output JSON
                        text_content = json.dumps(new_json)
                        generator = DSPyGenerator(
                            Model.GPT_3_5.value,
                            prompt_strategy,
                            doc_schema,
                            doc_type,
                            verbose,
                        )
                        answer, field_stats = generator.generate(
                            text_content,
                            promptQuestion,
                            budget=token_budget,
                        )

                        # update conv_query_stats
                        conv_query_stats[f"{field_name}_{idx}_fallback"] = field_stats

                        # extract result from JSON and set the DataRecord's field with its generated value
                        jsonObj = getJsonFromAnswer(answer)
                        answer = jsonObj[field_name]
                    except:
                        # update conv_query_stats
                        conv_query_stats[f"{field_name}_{idx}_fallback"] = field_stats
                        answer = None

                print(f"SETTING {field_name} to be {answer}")
                while type(answer) == type([]):
                    answer = answer[0]
                setattr(dr, field_name, answer)

            # # TODO: last minute hack for Biofabric; for some reason some records are not setting a filename
            # # I will need to debug this more thoroughly in the future, but for now this is an easy fix
            # if not hasattr(dr, 'filename'):
            #     setattr(dr, 'filename', candidate.filename)

            drs.append(dr)

        # construct ConventionalQueryStats object
        op_time = sum([gen_stats['op_time'] for gen_stats in conv_query_stats.values()])
        op_cost = sum([gen_stats['op_cosr'] for gen_stats in conv_query_stats.values()])

        conventional_query_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            op_id="codegen_query_123", # TODO
            op_name="codegen_query",
            op_time=op_time,
            op_cost=op_cost,
            record_stats= conv_query_stats,
            # TODO update RecordOpStats
        )

        # TODO: debug root cause
        for dr in drs:
            if not hasattr(dr, "filename"):
                setattr(dr, "filename", candidate.filename)

        return drs, full_code_gen_stats, conventional_query_stats

    else:
        #TODO here fill real stats
        full_code_gen_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            op_id="codegen_query_123",
            op_name="codegen_query",
            op_time=0,
            op_cost=0,
            record_state= {},
        )
        cache = DataDirectory().getCacheService()
        for field_name in generate_field_names:
            code_ensemble_id = "_".join([op_id, field_name])
            cached_code_ensemble_info = cache.getCachedData(
                f"codeEnsemble{plan_idx}", code_ensemble_id
            )
            if cached_code_ensemble_info is not None:
                code_ensemble, _ = cached_code_ensemble_info
                gen_stats = RecordOpStats()
                examples = cache.getCachedData(
                    f"codeSamples{plan_idx}", code_ensemble_id
                )
            else:
                code_ensemble, gen_stats, examples = dict(), None, list()

            # remove bytes data from candidate
            candidate_dict = candidate._asJSON(
                include_bytes=False, include_data_cols=False
            )
            candidate_dict = json.loads(candidate_dict)
            candidate_dict = {k: v for k, v in candidate_dict.items() if v != "<bytes>"}

            examples.append(candidate_dict)
            cache.putCachedData(f"codeSamples{plan_idx}", code_ensemble_id, examples)
            api = API.from_inout_schema(
                inputSchema=inputSchema, outputSchema=outputSchema, field_name=field_name, input_fields=candidate_dict.keys()
            )
            if len(code_ensemble) == 0 or reGenerationCondition(api, examples=examples):
                code_ensemble, gen_stats = codeEnsembleGeneration(
                    api, examples=examples
                )
                cache.putCachedData(
                    f"codeEnsemble{plan_idx}",
                    code_ensemble_id,
                    (code_ensemble, gen_stats),
                )

            for code_name, code in code_ensemble.items():
                print(f"CODE NAME: {code_name}")
                print("-----------------------")
                print(code)

            answer, exec_stats = codeEnsembleExecution(
                api, code_ensemble, candidate_dict
            )
            # TODO refactor stats to be compatible with new RecordOpStats
            # full_code_gen_stats.record_state[field_name] = gen_stats
            # full_code_gen_stats.code_exec_stats[field_name] = exec_stats

            if answer is None:
                print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                # construct prompt question
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()
                promptQuestion = _construct_query_prompt(
                    promptQuestion = _construct_query_prompt(
                    doc_type=doc_type,
                    inputSchema=inputSchema,
                    outputSchema=outputSchema,
                    cardinality=cardinality,
                    prompt_strategy=prompt_strategy,
                    conversionDesc=None,
                    generate_field_names=[field_name],
                    is_conventional=False
                    )
                    )
                field_stats = None
                try:
                    # print(f"FALL BACK FIELD: {field_name}")
                    # print("---------------")
                    # invoke LLM to generate output JSON
                    text_content = json.loads(candidate_dict)
                    generator = DSPyGenerator(
                        Model.GPT_3_5.value,
                        prompt_strategy,
                        doc_schema,
                        doc_type,
                        verbose,
                    )
                    answer, field_stats = generator.generate(
                        text_content,
                        promptQuestion,
                        budget=token_budget,
                    )

                    # update stats
                    conv_query_stats[f"{field_name}_fallback"] = field_stats

                    # extract result from JSON and set the DataRecord's field with its generated value
                    jsonObj = getJsonFromAnswer(answer)
                    answer = jsonObj[field_name]
                except:
                    # update stats
                    conv_query_stats[f"{field_name}_fallback"] = field_stats
                    answer = None

            print(f"SETTING {field_name} to be {answer}")
            setattr(dr, field_name, answer)

        # # TODO: last minute hack for Biofabric; for some reason some records are not setting a filename
        # # I will need to debug this more thoroughly in the future, but for now this is an easy fix
        # if not hasattr(dr, 'filename'):
        #     setattr(dr, 'filename', candidate.filename)

        # construct ConventionalQueryStats object
        conventional_query_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            op_id="conventional_query_123",
            op_name="conventional_query",
            op_time=sum([gen_stats['op_time'] for gen_stats in conv_query_stats.values()]),
            op_cost=sum([gen_stats['op_cost'] for gen_stats in conv_query_stats.values()]),
            record_stats= conv_query_stats,
        )

        # TODO: debug root cause
        if not hasattr(dr, "filename"):
            setattr(dr, "filename", candidate.filename)

        return dr, full_code_gen_stats, conventional_query_stats

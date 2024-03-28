from palimpzest.constants import PromptStrategy
from palimpzest.elements import DataRecord
from palimpzest.generators import DSPyGenerator, ImageTextGenerator
from palimpzest.profiler import Stats
from palimpzest.solver import TaskDescriptor

from typing import Any, Dict, List, Tuple

import base64
import json


def _construct_bonded_query_prompt(td: TaskDescriptor, doc_type: str, generate_field_names: List[str]) -> str:
    """
    This function constructs the prompt for a bonded query.
    """
    # build string of input fields and their descriptions
    multilineInputFieldDescription = ""
    for field_name in td.inputSchema.fieldNames():
        f = getattr(td.inputSchema, field_name)
        multilineInputFieldDescription += f"INPUT FIELD {field_name}: {f.desc}\n"

    # build string of output fields and their descriptions
    multilineOutputFieldDescription = ""
    for field_name in generate_field_names:
        f = getattr(td.outputSchema, field_name)
        multilineOutputFieldDescription += f"OUTPUT FIELD {field_name}: {f.desc}\n"

    # add input/output schema descriptions (if they have a docstring)
    optionalInputDesc = "" if td.inputSchema.__doc__ is None else f"Here is a description of the input object: {td.inputSchema.__doc__}."
    optionalOutputDesc = "" if td.outputSchema.__doc__ is None else f"Here is a description of the output object: {td.outputSchema.__doc__}."

    # construct promptQuestion
    promptQuestion = None
    if td.prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
        promptQuestion = f"""I would like you to create an output JSON object that describes an object of type {doc_type}. 
        You will use the information in an input JSON object that I will provide. The input object has type {td.inputSchema.className()}.
        All of the fields in the output object can be derived using information from the input object.
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every input field name and a description: 
        {multilineInputFieldDescription}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}.
        Be sure to emit a JSON object only.
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}."                
    
    else:
        promptQuestion = f"""You are an image analysis bot. Analyze the supplied image and create an output JSON object that describes an object of type {doc_type}. 
        You will use the information in the image that I will provide. The input image has type {td.inputSchema.className()}.
        All of the fields in the output object can be derived using information from the input image.
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}.
        Be sure to emit a JSON object only.
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}." 

    return promptQuestion


def _get_JSON_from_answer(answer: str) -> Dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    if not answer.strip().startswith('{'):
        # Find the start index of the actual JSON string
        # assuming the prefix is followed by the JSON object/array
        start_index = answer.find('{') if '{' in answer else answer.find('[')
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]

    if not answer.strip().endswith('}'):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind('}') if '}' in answer else answer.rfind(']')
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[:end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace("\_", "_")

    return json.loads(answer)


def runBondedQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[DataRecord, Stats, str]:
    # initialize output data record
    dr = DataRecord(td.outputSchema)
    dr.parent_uuid = candidate.uuid

    # copy fields from the candidate (input) record if they already exist
    # and construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in td.outputSchema.fieldNames():
        if field_name in td.inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate.asTextJSON()
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    # construct prompt question
    promptQuestion = _construct_bonded_query_prompt(td, doc_type, generate_field_names)

    # generate LLM response and capture statistics
    answer, query_stats = None, None
    try:
        if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # invoke LLM to generate output JSON
            generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
            answer, query_stats = generator.generate(text_content, promptQuestion)

            # add input and *computed* output fields to query_stats object
            query_stats["in_fields"] = td.inputSchema.fieldNames()
            query_stats["out_fields"] = generate_field_names

        elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            # b64 decode of candidate.contents
            image_b64 = base64.b64encode(candidate.contents).decode('utf-8')

            # invoke LLM to generate output JSON
            generator = ImageTextGenerator(td.model.value)
            answer, query_stats = generator.generate(image_b64, promptQuestion)

            # add input and *computed* output fields to query_stats object
            query_stats["in_fields"] = td.inputSchema.fieldNames()
            query_stats["out_fields"] = generate_field_names

        # TODO
        elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
            raise Exception("not implemented yet")

        # TODO
        elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
            raise Exception("not implemented yet")

        # parse JSON object from the answer
        jsonObj = _get_JSON_from_answer(answer)

        # set the DataRecord's fields with their generated values 
        for field_name in generate_field_names:
            setattr(dr, field_name, jsonObj[field_name])

    except Exception as e:
        return None, query_stats, str(e)

    return dr, query_stats, None


def runConventionalQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[DataRecord, Stats]:
    # initialize output data record
    dr = DataRecord(td.outputSchema)
    dr.parent_uuid = candidate.uuid

    # copy fields from the candidate (input) record if they already exist
    # and construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in td.outputSchema.fieldNames():
        if field_name in td.inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate.asTextJSON()
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    # iterate over fields and generate their values using an LLM
    query_stats = {}
    for field_name in generate_field_names:
        f = getattr(td.outputSchema, field_name)
        promptQuestion = f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if td.conversionDesc is None else f" Keep in mind that this output is described by this text: {td.conversionDesc}."
        try:
            field_stats = None
            if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # invoke LLM to generate output JSON
                generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                answer, field_stats = generator.generate(text_content, promptQuestion)

            elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:                               
                # b64 decode of candidate.contents
                image_b64 = base64.b64encode(candidate.contents).decode('utf-8')

                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(td.model.value)
                answer, field_stats = generator.generate(image_b64, promptQuestion)

            # TODO
            elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
                raise Exception("not implemented yet")

            # TODO
            elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
                raise Exception("not implemented yet")

            # set the DataRecord's field with its generated value
            setattr(dr, field_name, answer)

            # update query_stats
            query_stats[f"{field_name}"] = field_stats

        except Exception as e:
            print(f"Conventional field processing error: {e}")
            setattr(dr, field_name, None)
            query_stats[f"{field_name}"] = None

    return dr, query_stats

""" This file contains prompts used by Palimpzest 
Whenever they are called, they can be parameterize with the str.format() method using the parameter names that are in brackets.
For now, this is an easy decoupling. In the future, we maybe want a more sophisticated approach like a PromptBuilder.
"""

from enum import Enum

### CONVERT PROMPTS ### 
INPUT_FIELD = "{field_name}: {field_desc}\n"
OUTPUT_FIELD = "{field_name}: {field_desc}\n"

OPTIONAL_INPUT_DESC = "Here is a description of the input object: {desc}."
OPTIONAL_OUTPUT_DESC = "Here is a description of the output object: {desc}."

OPTIONAL_DESC = "Keep in mind that this process is described by this text: {desc}."

### ONE TO ONE ###
ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR = "an output JSON object that describes an object of type {doc_type}."
ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL = "the output object"
ONE_TO_ONE_APPENDIX_INSTRUCTION = "Be sure to emit a JSON object only. The dictionary should only have the output fields: {fields}.\n\nFor example:\n{fields_example_dict}"

# TODO: add JSON dict example to ONE_TO_MANY_APPENDIX_INSTRUCTION
### ONE_TO_MANY ###
ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR = "an output array of zero or more JSON objects that describe objects of type {doc_type}."
ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL = "the output objects"
ONE_TO_MANY_APPENDIX_INSTRUCTION = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields {fields}. You must decide the correct number of output objects."

STRUCTURED_CONVERT_PROMPT = """I would like you to create {targetOutputDescriptor}
You will use the information in an input JSON object that I will provide. The input object has type {input_type}.
All of the fields in {outputSingleOrPlural} can be derived using information from the input object.
{optionalInputDesc}
{optionalOutputDesc}
Here is every input field name and a description: 
{multilineInputFieldDescription}
Here is every output field name and a description:
{multilineOutputFieldDescription}
{appendixInstruction}
{optional_desc}"""

IMAGE_CONVERT_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and create {targetOutputDescriptor}.
You will use the information in the image that I will provide. The input image(s) has type {input_type}.
All of the fields in {outputSingleOrPlural} can be derived using information from the input image(s).
{optionalInputDesc}
{optionalOutputDesc}
Here is every output field name and a description:
{multilineOutputFieldDescription}
{appendixInstruction}
{optional_desc}"""

IMAGE_FILTER_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and:
- Output TRUE if the given image satisfies the filter condition
- Output FALSE if the given image does not satisfy the condition

Your answer must be TRUE or FALSE.

FILTER CONDITION: {filter_condition}

ANSWER: """

### MIXTURE-OF-AGENTS PROMPTS ###
MOA_ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR = "an output that describes an object of type {doc_type}."
MOA_ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR = "an output that describes one or more objects of type {doc_type}."

MOA_STRUCTURED_CONVERT_PROMPT = """I would like you to create {targetOutputDescriptor}
You will use the information in an input JSON object that I will provide. The input object has type {input_type}.
All of the fields in {outputSingleOrPlural} can be derived using information from the input object.
{optionalInputDesc}
{optionalOutputDesc}
Here is every input field name and a description: 
{multilineInputFieldDescription}
Here is the field name and a description of every field which should be present in your output:
{multilineOutputFieldDescription}
Your output should be a paragraph or two describing what you believe should be the keys and values of the output object of type {doc_type}. Be sure to cite information from the Context as evidence of why your output is correct. Do not hallucinate evidence, and if you are uncertain about any parts of the output -- say so."""

MOA_IMAGE_CONVERT_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and create {targetOutputDescriptor}.
You will use the information in the image that I will provide. The input image(s) has type {input_type}.
All of the fields in {outputSingleOrPlural} can be derived using information from the input image(s).
{optionalInputDesc}
{optionalOutputDesc}
Here is the field name and a description of every field which should be present in your output:
{multilineOutputFieldDescription}.
Your output should be a paragraph or two describing what you believe should be the keys and values of the output object of type {doc_type}.
**Include text snippets from the Context as evidence of why your output is correct.**
Do not hallucinate evidence, and if you are uncertain about any parts of the output -- say so."""

MOA_AGGREGATOR_CONVERT_PROMPT = """I would like you to create {targetOutputDescriptor}
You will use the information in the provided model responses to synthesize your response.
All of the fields in {outputSingleOrPlural} can be derived using information from the responses.
{optionalOutputDesc}
Here is the field name and a description of every field which should be present in your output:
{multilineOutputFieldDescription}
{appendixInstruction}
{optional_desc}
"""

### CODE SYNTHESIS PROMPTS ###
EXAMPLE_PROMPT = """Example{idx}:
Example Input
-------------
{example_inputs}

Example Output
--------------
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


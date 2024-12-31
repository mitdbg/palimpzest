"""This file contains prompts used by Palimpzest
Whenever they are called, they can be parameterized with the str.format() method using the parameter names that are in brackets.
For now, this is an easy decoupling. In the future, we maybe want a more sophisticated approach like a PromptBuilder.
"""

### FORMATTING INSTRUCTIONS ###
ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified output fields."
ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON list of dictionaries. The list may contain one or more dictionaries, and each dictionary should only have the specified output fields."

### DEVELOPER / SYSTEM PROMPTS ###
COT_BOOL_SYSTEM_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE.

An example is shown below:
---
CONTEXT:
{
  "text": "The quick brown fox jumps over the lazy dog."
}

INPUT FIELDS:
- text: a short passage of text

FILTER CONDITION: the text mentions an animal

Let's think step by step in order to answer the question.

REASONING: the text mentions the words "fox" and "dog" which are animals, therefore the answer is TRUE.

ANSWER: TRUE
"""

COT_BOOL_IMAGE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to answer a TRUE / FALSE question.
You will be presented with the image(s) and a filter condition. You may also have some textual inputs. Output TRUE if the input(s) satisfy the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE.

An example is shown below (the image will be provided in a subsequent message, suppose it is an image of a dog playing with a cat):
---
CONTEXT:
{
  "image": <bytes>,
  "filename": "img001.jpg"
}

INPUT FIELDS:
- image: an image of a scene
- filename: the filename of the image

FILTER CONDITION: there's an animal in this image

Let's think step by step in order to answer the question.

REASONING: the image shows a dog and a cat playing, both of which are animals, therefore the answer is TRUE.

ANSWER: TRUE
"""

COT_QA_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

Let's think step by step in order to answer the question.

REASONING: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.

ANSWER:
{
  "name": "Augusta Ada King",
  "birth_year": 1815
}
---

"""

COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field. All of the fields in the output can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

Let's think step by step in order to answer the question.

ANSWER: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.
---

"""

COT_MOA_AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. The context will contain one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}

MODEL RESPONSE 1: the text mentions the scientist's full name "Augusta Ada King, Countess of Lovelace" and states she was an English mathematician who worked on Babbage's Analytical Engine.

MODEL RESPONSE 2: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

Let's think step by step in order to answer the question.

REASONING: Looking at both model responses, they agree that the scientist's formal name is "Augusta Ada King". Model Response 2 correctly extracts the birth year from the birthday field as 1815. The responses are consistent and provide sufficient evidence for these values.

ANSWER:
{
  "name": "Augusta Ada King",
  "birth_year": 1815
}
---

"""

COT_QA_IMAGE_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each output field. All of the fields in the output JSON object can be derived using information from the input(s).

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below (the image will be provided in a subsequent message, suppose it is an image of a dog playing with a cat):
---
CONTEXT:
{
  "image": <bytes>,
  "filename": "img001.jpg"
}

INPUT FIELDS:
- image: an image of a scene
- filename: the filename of the image

OUTPUT FIELDS:
- dog_in_image: true if a dog is in the image and false otherwise
- person_in_image: true if a person is in the image and false otherwise

Let's think step by step in order to answer the question.

REASONING: The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image.

ANSWER:
{
  "dog_in_image": true,
  "person_in_image": false
}
---

"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
COT_BOOL_USER_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE.
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

FILTER CONDITION: {filter_condition}

Let's think step by step in order to answer the question.

REASONING: """

COT_BOOL_IMAGE_USER_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to answer a TRUE / FALSE question.
You will be presented with the image(s) and a filter condition. You may also have some textual inputs. Output TRUE if the input(s) satisfy the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE.
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

FILTER CONDITION: {filter_condition}

Let's think step by step in order to answer the question.

REASONING: """

COT_QA_BASE_USER_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step by step in order to answer the question.

REASONING: """

COT_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field. All of the fields in the output can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step by step in order to answer the question.

REASONING: """

COT_MOA_AGG_BASE_USER_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. The context will contain one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

{model_responses}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step by step in order to answer the question.

REASONING: """

COT_QA_IMAGE_BASE_USER_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each output field. All of the fields in the output JSON object can be derived using information from the input(s).

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step by step in order to answer the question.

REASONING: """

### CONVERT PROMPTS ###
INPUT_FIELD = "{field_name}: {field_desc}\n"
OUTPUT_FIELD = "{field_name}: {field_desc}\n"

OPTIONAL_INPUT_DESC = "Here is a description of the input object: {desc}."
OPTIONAL_OUTPUT_DESC = "Here is a description of the output object: {desc}."

OPTIONAL_DESC = "Keep in mind that this process is described by this text: {desc}."
LLAMA_INSTRUCTION = "Keep your answer brief and to the point. Do not repeat yourself endlessly."

### ONE TO ONE ###
ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR = "an output JSON object that describes an object of type {doc_type}."
ONE_TO_ONE_OUTPUT_SINGLE_OR_PLURAL = "the output object"
ONE_TO_ONE_APPENDIX_INSTRUCTION = "Be sure to emit a JSON object only. The dictionary should only have the output fields: {fields}.\n\nFor example:\n{fields_example_dict}"

# TODO: add JSON dict example to ONE_TO_MANY_APPENDIX_INSTRUCTION
### ONE_TO_MANY ###
ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR = (
    "an output array of zero or more JSON objects that describe objects of type {doc_type}."
)
ONE_TO_MANY_OUTPUT_SINGLE_OR_PLURAL = "the output objects"
ONE_TO_MANY_APPENDIX_INSTRUCTION = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields {fields}. You must decide the correct number of output objects."

STRUCTURED_CONVERT_PROMPT = """I would like you to create {target_output_descriptor}
You will use the information in an input JSON object that I will provide. The input object has type {input_type}.
All of the fields in {output_single_or_plural} can be derived using information from the input object.
{optional_input_desc}
{optional_output_desc}
Here is every input field name and a description: 
{multiline_input_field_description}
Here is every output field name and a description:
{multiline_output_field_description}
{appendix_instruction}
{optional_desc}
{model_instruction}"""

IMAGE_CONVERT_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and create {target_output_descriptor}.
You will use the information in the image that I will provide. The input image(s) has type {input_type}.
All of the fields in {output_single_or_plural} can be derived using information from the input image(s).
{optional_input_desc}
{optional_output_desc}
Here is every output field name and a description:
{multiline_output_field_description}
{appendix_instruction}
{optional_desc}
{model_instruction}"""

IMAGE_FILTER_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and:
- Output TRUE if the given image satisfies the filter condition
- Output FALSE if the given image does not satisfy the condition

Your answer must be TRUE or FALSE.

FILTER CONDITION: {filter_condition}

ANSWER: """

### MIXTURE-OF-AGENTS PROMPTS ###
MOA_ONE_TO_ONE_TARGET_OUTPUT_DESCRIPTOR = "an output that describes an object of type {doc_type}."
MOA_ONE_TO_MANY_TARGET_OUTPUT_DESCRIPTOR = "an output that describes one or more objects of type {doc_type}."

MOA_STRUCTURED_CONVERT_PROMPT = """I would like you to create {target_output_descriptor}
You will use the information in an input JSON object that I will provide. The input object has type {input_type}.
All of the fields in {output_single_or_plural} can be derived using information from the input object.
{optional_input_desc}
{optional_output_desc}
Here is every input field name and a description: 
{multiline_input_field_description}
Here is the field name and a description of every field which should be present in your output:
{multiline_output_field_description}
Your output should be a paragraph or two describing what you believe should be the keys and values of the output object of type {doc_type}. Be sure to cite information from the Context as evidence of why your output is correct. Do not hallucinate evidence, and if you are uncertain about any parts of the output -- say so."""

MOA_IMAGE_CONVERT_PROMPT = """You are an image analysis bot. Analyze the supplied image(s) and create {target_output_descriptor}.
You will use the information in the image that I will provide. The input image(s) has type {input_type}.
All of the fields in {output_single_or_plural} can be derived using information from the input image(s).
{optional_input_desc}
{optional_output_desc}
Here is the field name and a description of every field which should be present in your output:
{multiline_output_field_description}.
Your output should be a paragraph or two describing what you believe should be the keys and values of the output object of type {doc_type}.
**Include text snippets from the Context as evidence of why your output is correct.**
Do not hallucinate evidence, and if you are uncertain about any parts of the output -- say so.
{model_instruction}"""

MOA_AGGREGATOR_CONVERT_PROMPT = """I would like you to create {target_output_descriptor}
You will use the information in the provided model responses to synthesize your response.
All of the fields in {output_single_or_plural} can be derived using information from the responses.
{optional_output_desc}
Here is the field name and a description of every field which should be present in your output:
{multiline_output_field_description}
{appendix_instruction}
{optional_desc}
{model_instruction}
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

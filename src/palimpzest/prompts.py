"""This file contains prompts used by Palimpzest
Whenever they are called, they can be parameterized with the str.format() method using the parameter names that are in brackets.
For now, this is an easy decoupling. In the future, we maybe want a more sophisticated approach like a PromptBuilder.
"""

### FORMATTING INSTRUCTIONS ###
ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified output fields."
ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON list of dictionaries. The list may contain one or more dictionaries, and each dictionary should only have the specified output fields."

### REASONING INSTRUCTION FOR IMAGE PROMPTS ###
IMAGE_REASONING_SUFFIX = """Let's think step by step in order to answer the question.

REASONING: """
IMAGE_ANSWER_SUFFIX = """Let's think step by step in order to answer the question.

ANSWER: """

### DEVELOPER / SYSTEM PROMPTS ###
COT_BOOL_SYSTEM_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "text": "The quick brown fox jumps over the lazy dog."
}}

INPUT FIELDS:
- text: a short passage of text

FILTER CONDITION: the text mentions an animal

Let's think step by step in order to answer the question.

REASONING: the text mentions the words "fox" and "dog" which are animals, therefore the answer is TRUE.

ANSWER: TRUE
---
"""

COT_BOOL_IMAGE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to answer a TRUE / FALSE question.
You will be presented with the image(s) and a filter condition. You may also have some textual inputs. Output TRUE if the input(s) satisfy the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}

INPUT FIELDS:
- image: an image of a scene
- photographer: the photographer of the image

FILTER CONDITION: there's an animal in this image

<image content provided here; assume in this example the image shows a dog and a cat playing>

Let's think step by step in order to answer the question.

REASONING: the image shows a dog and a cat playing, both of which are animals, therefore the answer is TRUE.

ANSWER: TRUE
---
"""

COT_QA_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}}

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

Let's think step by step in order to answer the question.

REASONING: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.

ANSWER:
{{
  "name": "Augusta Ada King",
  "birth_year": 1815
}}
---
"""

COT_QA_IMAGE_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each output field. All of the fields in the output JSON object can be derived using information from the input(s).

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}

INPUT FIELDS:
- image: an image of a scene
- photographer: the photographer of the image

OUTPUT FIELDS:
- dog_in_image: true if a dog is in the image and false otherwise
- person_in_image: true if a person is in the image and false otherwise

<image content provided here; assume in this example the image shows a dog and a cat playing>

Let's think step by step in order to answer the question.

REASONING: The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image.

ANSWER:
{{
  "dog_in_image": true,
  "person_in_image": false
}}
---
"""

COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}}

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

COT_MOA_PROPOSER_IMAGE_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce an answer to a question.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the input(s) as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}

INPUT FIELDS:
- image: an image of a scene
- photographer: the photographer of the image

OUTPUT FIELDS:
- dog_in_image: true if a dog is in the image and false otherwise
- person_in_image: true if a person is in the image and false otherwise

<image content provided here; assume in this example the image shows a dog and a cat playing>

Let's think step by step in order to answer the question.

ANSWER: The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image.
---
"""

COT_MOA_AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
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
{{
  "name": "Augusta Ada King",
  "birth_year": 1815
}}
---
"""


### USER / INSTANCE-SPECIFIC PROMPTS ###
COT_BOOL_USER_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
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

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

FILTER CONDITION: {filter_condition}

"""

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

"""

COT_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

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

COT_MOA_PROPOSER_IMAGE_BASE_USER_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce an answer to a question.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the input(s) as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

"""

COT_MOA_AGG_BASE_USER_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---
---
{model_responses}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step by step in order to answer the question.

REASONING: """


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

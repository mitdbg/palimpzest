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

An example is shown below (the image will be provided in a subsequent message, suppose it is an image of a dog playing with a cat):
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

COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field. All of the fields in the output can be derived using information from the context.

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

COT_MOA_AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with a context and a set of output fields to generate. The context will contain one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}}

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

COT_QA_IMAGE_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object.
You will be presented with the image(s) and a set of output fields to generate. You may also have some textual inputs. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each output field. All of the fields in the output JSON object can be derived using information from the input(s).

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below (the image will be provided in a subsequent message, suppose it is an image of a dog playing with a cat):
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

Let's think step by step in order to answer the question.

REASONING: The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image.

ANSWER:
{{
  "dog_in_image": true,
  "person_in_image": false
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

COT_QA_BASE_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the JSON object generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Does the JSON object adhere to the required format? Highlight any structural issues.
2. Are the values of the output fields accurate based on the provided context? If any fields are incorrect or missing, provide specific examples.
3. Are there any logical errors in reasoning used to derive the output? Provide detailed feedback on potential mistakes.

Finish your critique with actionable recommendations for improving the JSON object.
"""

COT_QA_BASE_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original JSON object generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the original JSON object to address the critique. Ensure the refined JSON:
1. Adheres to the required format specified in the prompt.
2. Correctly derives all values for the output fields based on the provided context.
3. Resolves any logical errors identified in the critique.

Return the refined JSON object as your final answer.
"""

COT_BOOL_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the answer generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Does the answer adhere to the required TRUE or FALSE format?
2. Is the reasoning provided logically sound and well-supported by the context?
3. Are there any errors in applying the filter condition to the given context?

Finish your critique with actionable recommendations for improving the model's reasoning and answer format.
"""

COT_BOOL_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original answer generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the answer based on the critique. Ensure that:
1. The answer adheres to the required TRUE or FALSE format.
2. The reasoning is logically sound and supported by the given context.
3. The filter condition is correctly applied.

Return the improved answer.
"""

COT_BOOL_IMAGE_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the answer generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Does the answer adhere to the required TRUE or FALSE format?
2. Is the reasoning well-supported by the provided image(s) and/or text?
3. Are there any logical errors in applying the filter condition to the given context?

Finish your critique with actionable recommendations for improving the model's reasoning and answer format.
"""

COT_BOOL_IMAGE_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original answer generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the answer based on the critique. Ensure that:
1. The answer adheres to the required TRUE or FALSE format.
2. The reasoning correctly considers the image(s) and/or text provided in the context.
3. The filter condition is properly applied.

Return the improved answer.
"""

COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the response generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Is the response well-structured and does it clearly explain each output field?
2. Are all claims supported by the provided context? Identify any instances of hallucination or missing evidence.
3. Does the response cite specific parts of the context when making claims?

Finish your critique with actionable recommendations for improving the response.
"""

COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original response generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the response based on the critique. Ensure that:
1. The response is well-structured and clearly explains each output field.
2. All claims are directly supported by the provided context.
3. The response explicitly cites relevant parts of the context.

Return the improved response.
"""

COT_MOA_AGG_BASE_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the synthesized JSON object generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Does the JSON object adhere to the required format?
2. Does the synthesis appropriately combine responses from multiple models, resolving conflicts where necessary?
3. Are there any biases, inaccuracies, or missing information in the final output?

Finish your critique with actionable recommendations for improving the synthesized response.
"""

COT_MOA_AGG_BASE_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original synthesized JSON object generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the synthesized JSON object based on the critique. Ensure that:
1. The JSON object adheres to the required format.
2. The synthesis properly reconciles different model responses, making informed decisions on conflicts.
3. The final output is accurate, unbiased, and complete.

Return the improved JSON object.
"""

COT_QA_IMAGE_BASE_SYSTEM_PROMPT_CRITIQUE = """You are a helpful assistant tasked with reviewing the output of a model based on a given prompt.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the JSON object generated by the model:

OUTPUT:
{original_output}

Your task is to critique the output based on the following:
1. Does the JSON object adhere to the required format?
2. Are the values of the output fields accurate based on the provided image(s) and/or text?
3. Are there any logical errors in the model's reasoning when extracting information from images and text?

Finish your critique with actionable recommendations for improving the JSON object.
"""

COT_QA_IMAGE_BASE_SYSTEM_PROMPT_REFINEMENT = """You are a helpful assistant tasked with refining the output of a model based on a critique.
Below is the original user prompt used to generate the output:

USER PROMPT:
{user_prompt}

Here is the original JSON object generated by the model:

ORIGINAL OUTPUT:
{original_output}

Here is the critique of the output:

CRITIQUE:
{critique_output}

Your task is to refine the original JSON object to address the critique. Ensure the refined JSON:
1. Adheres to the required format specified in the prompt.
2. Correctly derives all values for the output fields based on the provided image(s) and/or text.
3. Resolves any logical errors identified in the critique.

Return the refined JSON object as your final answer.
"""


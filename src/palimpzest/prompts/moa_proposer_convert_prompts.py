"""This file contains prompts for MixtureOfAgentsConvert operations on text inputs."""

### BASE PROMPTS ###
COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELDS:
{example_output_fields}

CONTEXT:
{example_context}
{image_disclaimer}
Let's think step-by-step in order to answer the question.

ANSWER: {example_answer}
---
"""

COT_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.
---
INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

CONTEXT:
{context}
<<image-placeholder>>
Let's think step-by-step in order to answer the question.

ANSWER: """


### TEMPLATE INPUTS ###
COT_MOA_PROPOSER_JOB_INSTRUCTION = "produce an answer to a question"
COT_MOA_PROPOSER_IMAGE_JOB_INSTRUCTION = "analyze input image(s) and/or text in order to produce an answer to a question"

COT_MOA_PROPOSER_EXAMPLE_INPUT_FIELDS = """- text: a text passage describing a scientist
- birthday: the scientist's birthday"""
COT_MOA_PROPOSER_IMAGE_EXAMPLE_INPUT_FIELDS = """- image: an image of a scene
- photographer: the photographer of the image"""

COT_MOA_PROPOSER_EXAMPLE_OUTPUT_FIELDS = """- name: the name of the scientist
- birth_year: the year the scientist was born"""
COT_MOA_PROPOSER_IMAGE_EXAMPLE_OUTPUT_FIELDS = """- dog_in_image: true if a dog is in the image and false otherwise
- person_in_image: true if a person is in the image and false otherwise"""

COT_MOA_PROPOSER_EXAMPLE_CONTEXT = """{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}}"""
COT_MOA_PROPOSER_IMAGE_EXAMPLE_CONTEXT = """{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}"""

COT_MOA_PROPOSER_IMAGE_DISCLAIMER = """
<image content provided here; assume in this example the image shows a dog and a cat playing>
"""

COT_MOA_PROPOSER_EXAMPLE_ANSWER = """the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815."""
COT_MOA_PROPOSER_IMAGE_EXAMPLE_ANSWER = """The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image."""

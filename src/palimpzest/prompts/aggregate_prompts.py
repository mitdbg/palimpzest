"""This file contains prompts for convert operations."""

### BASE PROMPTS ###
COT_AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELD:
{example_output_fields}

CONTEXT:
{example_context}{image_disclaimer}{audio_disclaimer}

AGGREGATION INSTRUCTION: {example_agg_instruction}

Let's think step-by-step in order to answer the question.

REASONING: {example_reasoning}

ANSWER:
{example_answer}
---
"""

COT_AGG_NO_REASONING_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELD:
{example_output_fields}

CONTEXT:
{example_context}{image_disclaimer}{audio_disclaimer}

AGGREGATION INSTRUCTION: {example_agg_instruction}

ANSWER:
{example_answer}
---
"""


COT_AGG_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.
{desc_section}
{output_format_instruction} Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELD:
{output_fields_desc}

CONTEXT:
{context}<<image-placeholder>><<audio-placeholder>>

AGGREGATION INSTRUCTION: {agg_instruction}

Let's think step-by-step in order to answer the question.

REASONING: """

COT_AGG_NO_REASONING_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.
{desc_section}
{output_format_instruction} Finish your response with a newline character followed by ---
---
INPUT FIELD:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

CONTEXT:
{context}<<image-placeholder>><<audio-placeholder>>

AGGREGATION INSTRUCTION: {agg_instruction}

ANSWER: """

### TEMPLATE INPUTS ###
COT_AGG_JOB_INSTRUCTION = """perform an aggregation to generate a JSON object"""
COT_AGG_IMAGE_JOB_INSTRUCTION = """analyze input image(s) and/or text in order to perform an aggregation and produce a JSON object"""
COT_AGG_AUDIO_JOB_INSTRUCTION = """analyze input audio and/or text in order to perform an aggregation and produce a JSON object"""

COT_AGG_EXAMPLE_INPUT_FIELDS = """- text: a text passage describing a scientist
- birthday: the scientist's birthday"""
COT_AGG_IMAGE_EXAMPLE_INPUT_FIELDS = """- image: an image of a scene
- photographer: the photographer of the image"""
COT_AGG_AUDIO_EXAMPLE_INPUT_FIELDS = """- recording: an audio recording of a newscast
- speaker: the name of the speaker in the recording"""

COT_AGG_EXAMPLE_OUTPUT_FIELDS = """- num_names_starting_with_letter_a: the number of scientists whose first name starts with the letter 'A'
"""
COT_AGG_IMAGE_EXAMPLE_OUTPUT_FIELDS = """- num_dogs: the number of dogs in all images
"""
COT_AGG_AUDIO_EXAMPLE_OUTPUT_FIELDS = """- main_topic: the main topic discussed in the newscasts
"""

COT_AGG_EXAMPLE_CONTEXT = """{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}
{
  "text": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for its influence on the philosophy of science.",
  "birthday": "March 14, 1879"
}
{
  "text": "Niels Henrik David Bohr was a Danish physicist who made foundational contributions to understanding atomic structure and quantum theory, for which he received the Nobel Prize in Physics in 1922. Bohr was also a philosopher and a promoter of scientific research.",
  "birthday": "October 7, 1885"
}"""
COT_AGG_IMAGE_EXAMPLE_CONTEXT = """{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}
{
  "image": <bytes>,
  "photographer": "PhotoPro42"
}"""
COT_AGG_AUDIO_EXAMPLE_CONTEXT = """{
  "recording": <bytes>,
  "speaker": "Walter Cronkite"
}
{
  "recording": <bytes>,
  "speaker": "Anderson Cooper"
}"""

COT_AGG_IMAGE_DISCLAIMER = """
\n<image content provided here; assume in this example the first image shows two dogs and the second image shows one dog>
"""
COT_AGG_AUDIO_DISCLAIMER = """
\n<audio content provided here; assume in this example that both recordings are primarily about the Cuban Missile Crisis>
"""

COT_AGG_EXAMPLE_AGG_INSTRUCTION = "count the number of scientists whose first name starts with the letter 'A'"
COT_AGG_IMAGE_EXAMPLE_AGG_INSTRUCTION = "count the total number of dogs in all images"
COT_AGG_AUDIO_EXAMPLE_AGG_INSTRUCTION = "determine the main topic discussed in the newscasts"

COT_AGG_EXAMPLE_REASONING = """The text passages mention three scientists: "Augusta Ada King, Countess of Lovelace", "Albert Einstein", and "Niels Henrik David Bohr". Among these, two scientists have first names starting with the letter 'A': "Augusta" and "Albert". Therefore, the number of scientists whose first name starts with the letter 'A' is 2."""
COT_AGG_IMAGE_EXAMPLE_REASONING = """The first image shows two dogs, and the second image shows one dog. Therefore, the total number of dogs in all images is 3."""
COT_AGG_AUDIO_EXAMPLE_REASONING = """Both recordings primarily discuss the Cuban Missile Crisis. Therefore, the main topic discussed in the newscasts is the Cuban Missile Crisis."""

COT_AGG_EXAMPLE_ANSWER = """{
  "num_names_starting_with_letter_a": 2
}"""
COT_AGG_IMAGE_EXAMPLE_ANSWER = """{
  "num_dogs": 3
}"""
COT_AGG_AUDIO_EXAMPLE_ANSWER = """{
  "main_topic": "Cuban Missile Crisis"
}"""

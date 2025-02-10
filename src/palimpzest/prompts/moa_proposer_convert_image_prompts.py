"""This file contains prompts for Mixture-of-Agents convert operations on image inputs."""

### SYSTEM PROMPTS ###
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

Let's think step-by-step in order to answer the question.

ANSWER: The image shows a dog playing with a cat, so there is a dog in the image. There is no person in the image.
---
"""

# TODO?: add refine and critique?

### USER / INSTANCE-SPECIFIC PROMPTS ###
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

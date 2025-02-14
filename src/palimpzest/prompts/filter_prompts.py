"""This file contains prompts for filter operations."""

### BASE PROMPTS ###
COT_BOOL_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

CONTEXT:
{example_context}
{image_disclaimer}
FILTER CONDITION: {example_filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: {example_reasoning}

ANSWER: TRUE
---
"""

COT_BOOL_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

CONTEXT:
{context}
<<image-placeholder>>
FILTER CONDITION: {filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: """


### TEMPLATE INPUTS ###
COT_BOOL_JOB_INSTRUCTION = """answer a TRUE / FALSE question"""
COT_BOOL_IMAGE_JOB_INSTRUCTION = """analyze input image(s) and/or text in order to answer a TRUE / FALSE question"""

COT_BOOL_EXAMPLE_INPUT_FIELDS = """- text: a short passage of text"""
COT_BOOL_IMAGE_EXAMPLE_INPUT_FIELDS = """- image: an image of a scene
- photographer: the photographer of the image"""

COT_BOOL_EXAMPLE_CONTEXT = """{{
  "text": "The quick brown fox jumps over the lazy dog."
}}"""
COT_BOOL_IMAGE_EXAMPLE_CONTEXT = """{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}"""

COT_BOOL_EXAMPLE_FILTER_CONDITION = "the text mentions an animal"
COT_BOOL_IMAGE_EXAMPLE_FILTER_CONDITION = "there's an animal in this image"

COT_BOOL_IMAGE_DISCLAIMER = """
<image content provided here; assume in this example the image shows a dog and a cat playing>
"""

COT_BOOL_EXAMPLE_REASONING = """the text mentions the words "fox" and "dog" which are animals, therefore the answer is TRUE."""
COT_BOOL_IMAGE_EXAMPLE_REASONING = """the image shows a dog and a cat playing, both of which are animals, therefore the answer is TRUE."""

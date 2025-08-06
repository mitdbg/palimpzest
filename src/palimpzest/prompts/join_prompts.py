"""This file contains prompts for join operations."""

### BASE PROMPTS ###
COT_JOIN_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
LEFT INPUT FIELDS:
{example_input_fields}

LEFT CONTEXT:
{example_context}
{image_disclaimer}

RIGHT INPUT FIELDS:
{right_example_input_fields}

RIGHT CONTEXT:
{right_example_context}
{right_image_disclaimer}

JOIN CONDITION: {example_join_condition}

Let's think step-by-step in order to evaluate the join condition.

REASONING: {example_reasoning}

ANSWER: TRUE
---
"""

COT_JOIN_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
LEFT INPUT FIELDS:
{input_fields_desc}

LEFT CONTEXT:
{context}
<<image-placeholder>>

RIGHT INPUT FIELDS:
{right_input_fields_desc}

RIGHT CONTEXT:
{right_context}
<<image-placeholder>>

JOIN CONDITION: {join_condition}

Let's think step-by-step in order to evaluate the join condition.

REASONING: """


### TEMPLATE INPUTS ###
COT_JOIN_JOB_INSTRUCTION = """determine whether two data records satisfy a join condition"""
COT_JOIN_IMAGE_JOB_INSTRUCTION = """analyze input image(s) and/or text in order to determine whether two data records satisfy a join condition"""

COT_JOIN_EXAMPLE_INPUT_FIELDS = """- text: a short passage of text"""
COT_JOIN_IMAGE_EXAMPLE_INPUT_FIELDS = """- image: an image of a scene
- photographer: the photographer of the image"""

COT_JOIN_RIGHT_EXAMPLE_INPUT_FIELDS = """- contents: the contents of a text file"""
COT_JOIN_IMAGE_RIGHT_EXAMPLE_INPUT_FIELDS = """- image: an image of a scene
- photographer: the photographer of the image"""

COT_JOIN_EXAMPLE_CONTEXT = """{{
  "text": "The quick brown fox jumps over the lazy dog."
}}"""
COT_JOIN_IMAGE_EXAMPLE_CONTEXT = """{{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}}"""

COT_JOIN_RIGHT_EXAMPLE_CONTEXT = """{{
  "contents": "Foxes are wild animals which primarily hunt small mammals like rabbits and rodents."
}}"""
COT_JOIN_IMAGE_RIGHT_EXAMPLE_CONTEXT = """{{
  "image": <bytes>,
  "filename": "img123.png"
}}"""

COT_JOIN_EXAMPLE_JOIN_CONDITION = "each record mentions the same animal"
COT_JOIN_IMAGE_EXAMPLE_JOIN_CONDITION = "the images are of the same subject"

COT_JOIN_IMAGE_DISCLAIMER = """
<image content provided here; assume in this example the image shows a horse in a field>
"""
COT_JOIN_RIGHT_IMAGE_DISCLAIMER = """
<image content provided here; assume in this example the image shows a horse in its stable>
"""

COT_JOIN_EXAMPLE_REASONING = """both passages mention a fox, which is the same animal, therefore the answer is TRUE."""
COT_JOIN_IMAGE_EXAMPLE_REASONING = """both images show a horse, which appears to be the main subject of each image, therefore the answer is TRUE."""

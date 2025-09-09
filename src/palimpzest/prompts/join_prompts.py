"""This file contains prompts for join operations."""

### BASE PROMPTS ###
JOIN_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
LEFT INPUT FIELDS:
{example_input_fields}

LEFT CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

RIGHT INPUT FIELDS:
{right_example_input_fields}

RIGHT CONTEXT:
{{{right_example_context}}}{right_image_disclaimer}{right_audio_disclaimer}

JOIN CONDITION: {example_join_condition}

Let's think step-by-step in order to evaluate the join condition.

REASONING: {example_reasoning}

ANSWER: TRUE
---
"""

JOIN_NO_REASONING_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
LEFT INPUT FIELDS:
{example_input_fields}

LEFT CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

RIGHT INPUT FIELDS:
{right_example_input_fields}

RIGHT CONTEXT:
{{{right_example_context}}}{right_image_disclaimer}{right_audio_disclaimer}

JOIN CONDITION: {example_join_condition}

ANSWER: TRUE
---
"""

JOIN_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.
{desc_section}
Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
LEFT INPUT FIELDS:
{input_fields_desc}

LEFT CONTEXT:
{context}<<image-audio-placeholder>>

RIGHT INPUT FIELDS:
{right_input_fields_desc}

RIGHT CONTEXT:
{right_context}<<right-image-audio-placeholder>>

JOIN CONDITION: {join_condition}

Let's think step-by-step in order to evaluate the join condition.

REASONING: """

JOIN_NO_REASONING_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with two data records and a join condition. Output TRUE if the two data records satisfy the join condition, and FALSE otherwise.
{desc_section}
Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
LEFT INPUT FIELDS:
{input_fields_desc}

LEFT CONTEXT:
{context}<<image-audio-placeholder>>

RIGHT INPUT FIELDS:
{right_input_fields_desc}

RIGHT CONTEXT:
{right_context}<<right-image-audio-placeholder>>

JOIN CONDITION: {join_condition}

ANSWER: """

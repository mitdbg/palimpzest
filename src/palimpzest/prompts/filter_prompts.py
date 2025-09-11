"""This file contains prompts for filter operations."""

### BASE PROMPTS ###
FILTER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

FILTER CONDITION: {example_filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: {example_reasoning}

ANSWER: TRUE
---
"""

FILTER_NO_REASONING_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

FILTER CONDITION: {example_filter_condition}

ANSWER: TRUE
---
"""

FILTER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.
{desc_section}
Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

FILTER CONDITION: {filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: """

FILTER_NO_REASONING_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.
{desc_section}
Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

FILTER CONDITION: {filter_condition}

ANSWER: """

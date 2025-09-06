"""This file contains prompts for MixtureOfAgentsConvert operations."""

### SYSTEM PROMPTS ###
MAP_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a set of output fields to generate. Your task is to generate a detailed and succinct analysis describing what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELDS:
{example_output_fields}

CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

Let's think step-by-step in order to answer the question.

ANSWER: {example_answer}
---
"""

FILTER_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Your task is to generate a detailed and succinct analysis describing whether you believe the input satisfies the filter condition.
Be sure to cite information from the context as evidence of why your determination is correct. Do not hallucinate evidence.

You will be provided with a description of each input field.

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

CONTEXT:
{{{example_context}}}{image_disclaimer}{audio_disclaimer}

FILTER CONDITION: {example_filter_condition}

Let's think step-by-step in order to answer the question.

ANSWER: {example_answer}
---
"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
MAP_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a set of output fields to generate. Your task is to generate a detailed and succinct analysis describing what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.
{desc_section}
You will be provided with a description of each input field and each output field.
---
INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

Let's think step-by-step in order to answer the question.

ANSWER: """

FILTER_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and a filter condition. Your task is to generate a detailed and succinct analysis describing whether you believe the input satisfies the filter condition.
Be sure to cite information from the context as evidence of why your determination is correct. Do not hallucinate evidence.
{desc_section}
You will be provided with a description of each input field.

An example is shown below:
---
INPUT FIELDS:
{input_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

FILTER CONDITION: {filter_condition}

Let's think step-by-step in order to answer the question.

ANSWER: """

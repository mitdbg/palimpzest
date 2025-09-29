"""This file contains prompts for aggregation operations."""

### BASE PROMPTS ###
AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELDS:
{example_output_fields}

CONTEXT:
{{{example_context}}}
{{{second_example_context}}}
{{{third_example_context}}}{image_disclaimer}{audio_disclaimer}

AGGREGATION INSTRUCTION: {example_agg_instruction}

Let's think step-by-step in order to answer the question.

REASONING: {example_reasoning}

ANSWER:
{{{example_answer}}}
---
"""

AGG_NO_REASONING_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
INPUT FIELDS:
{example_input_fields}

OUTPUT FIELDS:
{example_output_fields}

CONTEXT:
{{{example_context}}}
{{{second_example_context}}}
{{{third_example_context}}}{image_disclaimer}{audio_disclaimer}

AGGREGATION INSTRUCTION: {example_agg_instruction}

ANSWER:
{{{example_answer}}}
---
"""


AGG_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.
{desc_section}
{output_format_instruction} Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

AGGREGATION INSTRUCTION: {agg_instruction}

Let's think step-by-step in order to answer the question.

REASONING: """

AGG_NO_REASONING_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
You will be presented with a context and an output field to generate. Your task is to generate a JSON object which aggregates the input and fills in the output field with the correct value.
You will be provided with a description of each input field and each output field. The field in the output JSON object can be derived using information from the context.
{desc_section}
{output_format_instruction} Finish your response with a newline character followed by ---
---
INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

CONTEXT:
{context}<<image-audio-placeholder>>

AGGREGATION INSTRUCTION: {agg_instruction}

ANSWER: """

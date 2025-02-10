"""This file contains prompts for MixtureOfAgentsConvert operations on text inputs."""

### SYSTEM PROMPTS ###
COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

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

Let's think step-by-step in order to answer the question.

ANSWER: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.
---
"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
COT_MOA_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to produce an answer to a question.
You will be presented with a context and a set of output fields to generate. Your task is to generate a paragraph or two which describes what you believe is the correct value for each output field.
Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.

You will be provided with a description of each input field and each output field.

{output_format_instruction} Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step-by-step in order to answer the question.

REASONING: """

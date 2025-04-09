"""This file contains prompts for SplitConvert operations on text inputs."""

### BASE PROMPTS ###
COT_SPLIT_PROPOSER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
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

Let's think step-by-step in order to answer the question.

ANSWER: {example_answer}
---
"""

COT_SPLIT_PROPOSER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to {job_instruction}.
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

Let's think step-by-step in order to answer the question.

ANSWER: """


### TEMPLATE INPUTS ###
SPLIT_PROPOSER_JOB_INSTRUCTION = "produce an answer to a question"
SPLIT_PROPOSER_EXAMPLE_INPUT_FIELDS = """- text: a text passage describing scientists"""
SPLIT_PROPOSER_EXAMPLE_OUTPUT_FIELDS = """- name: the list of names for each scientist mentioned in the text
- field_of_study: a list with the field of study for each scientist"""
SPLIT_PROPOSER_EXAMPLE_CONTEXT = """{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, born December 10, 1815 was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation."
}}"""
SPLIT_PROPOSER_EXAMPLE_ANSWER = """the text passage mentions the scientists "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and "Charles Babbage", both of whom were mathematicians. Therefore, the name output should be ["Augusta Ada King", "Charles Babbage"] and the field_of_study output should be ["Mathematician", "Mathematician"]."""

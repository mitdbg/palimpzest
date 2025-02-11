"""This file contains prompts for Mixture-of-Agents aggregator operations."""

### SYSTEM PROMPTS ###
COT_MOA_AGG_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
MODEL RESPONSE 1: the text mentions the scientist's full name "Augusta Ada King, Countess of Lovelace" and states she was an English mathematician who worked on Babbage's Analytical Engine.

MODEL RESPONSE 2: the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815.

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

Let's think step-by-step in order to answer the question.

REASONING: Looking at both model responses, they agree that the scientist's formal name is "Augusta Ada King". Model Response 2 correctly extracts the birth year from the birthday field as 1815. The responses are consistent and provide sufficient evidence for these values.

ANSWER:
{{
  "name": "Augusta Ada King",
  "birth_year": 1815
}}
---
"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
COT_MOA_AGG_BASE_USER_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---
---
{model_responses}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step-by-step in order to answer the question.

REASONING: """

"""This file contains prompts for SplitConvert aggregator operations."""

### SYSTEM PROMPTS ###
MAP_SPLIT_MERGER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models operating on chunks of an input. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased, incorrect, or contain duplicates.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---

An example is shown below:
---
CHUNK 1 OUTPUT: the text mentions the scientists "Augusta Ada King, Countess of Lovelace" and "Charles Babbage". It states that King was an English mathematician who worked on Babbage's Analytical Engine.

CHUNK 2 OUTPUT: the text passage mentions the scientist "Charles Babbage", who was a mathematician. Therefore, the name output should be ["Charles Babbage"] and the field_of_study output should be ["Mathematician"].

INPUT FIELDS:
- text: a text passage describing scientists

OUTPUT FIELDS:
- name: the list of names for each scientist mentioned in the text
- field_of_study: a list with the field of study for each scientist

Let's think step-by-step in order to answer the question.

REASONING: Looking at both chunk outputs, they specify that the scientists' formal names are "Augusta Ada King" and "Charles Babbage". Chunk Output 2 indicates that Charles Babbage was a Mathematician and Chunk Output 1 says that Augusta Ada King was an English mathematician. Therefore, the name output should be ["Augusta Ada King", "Charles Babbage"] and the field_of_study output should be ["Mathematician", "Mathematician"].

ANSWER:
{{
  "name": ["Augusta Ada King", "Charles Babbage"],
  "field_of_study": ["Mathematician", "Mathematician"]
}}
---
"""

FILTER_SPLIT_MERGER_BASE_SYSTEM_PROMPT = """You are a helpful assistant whose job is to answer a TRUE/FALSE question.
You will be presented with one or more outputs produced by a set of models operating on chunks of an input. Your task is to synthesize these responses into a single TRUE/FALSE answer.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased, incorrect, or contain duplicates.

You will be provided with a description of each input field and the filter condition.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
CHUNK 1 OUTPUT: The context describes Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, who is widely recognized as a foundational figure in computer science. Therefore, the answer is TRUE.

CHUNK 2 OUTPUT: Based on the context provided, Ada Lovelace is indeed a foundational computer scientist, therefore the answer is TRUE.

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday
- image: an image of the scientist
- recording: an audio recording of a newscast about the scientist's contributions to their field

FILTER CONDITION: The subject of the input is a foundational computer scientist.

Let's think step-by-step in order to answer the question.

REASONING: Looking at both chunk outputs, they agree that the subject is a foundational computer scientist. Both outputs provide consistent evidence supporting this conclusion.

ANSWER: TRUE
---
"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
MAP_SPLIT_MERGER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to generate a JSON object.
You will be presented with one or more outputs produced by a set of models. Your task is to synthesize these responses into a single, high-quality JSON object which fills in the output fields with the correct values.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased, incorrect, or contain duplicates.

You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the model responses.

{output_format_instruction} Finish your response with a newline character followed by ---
---
{chunk_outputs}

INPUT FIELDS:
{input_fields_desc}

OUTPUT FIELDS:
{output_fields_desc}

Let's think step-by-step in order to answer the question.

REASONING: """

FILTER_SPLIT_MERGER_BASE_USER_PROMPT = """You are a helpful assistant whose job is to answer a TRUE/FALSE question.
You will be presented with one or more outputs produced by a set of models operating on chunks of an input. Your task is to synthesize these responses into a single TRUE/FALSE answer.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased, incorrect, or contain duplicates.

You will be provided with a description of each input field and the filter condition.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
{chunk_outputs}

INPUT FIELDS:
{input_fields_desc}

FILTER CONDITION: {filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: """

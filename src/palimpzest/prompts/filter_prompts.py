"""This file contains prompts for filter operations on text inputs."""

### SYSTEM PROMPTS ###
COT_BOOL_SYSTEM_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---

An example is shown below:
---
CONTEXT:
{{
  "text": "The quick brown fox jumps over the lazy dog."
}}

INPUT FIELDS:
- text: a short passage of text

FILTER CONDITION: the text mentions an animal

Let's think step-by-step in order to answer the question.

REASONING: the text mentions the words "fox" and "dog" which are animals, therefore the answer is TRUE.

ANSWER: TRUE
---
"""

### USER / INSTANCE-SPECIFIC PROMPTS ###
COT_BOOL_USER_PROMPT = """You are a helpful assistant whose job is to answer a TRUE / FALSE question.
You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.

Remember, your answer must be TRUE or FALSE. Finish your response with a newline character followed by ---
---
CONTEXT:
{context}

INPUT FIELDS:
{input_fields_desc}

FILTER CONDITION: {filter_condition}

Let's think step-by-step in order to answer the question.

REASONING: """

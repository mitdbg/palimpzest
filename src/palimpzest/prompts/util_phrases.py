"""This file contains utility phrases which are templated into many of our prompts."""

### FORMATTING INSTRUCTIONS ###
ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified output fields."
ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON list of dictionaries. The list may contain one or more dictionaries, and each dictionary should only have the specified output fields."

### REASONING INSTRUCTION FOR IMAGE PROMPTS ###
COT_REASONING_INSTRUCTION = """Let's think step-by-step in order to answer the question.

REASONING: """

COT_ANSWER_INSTRUCTION = """Let's think step-by-step in order to answer the question.

ANSWER: """

"""This file contains strings which may be templated into our prompt templates."""

### FORMATTING INSTRUCTIONS ###
ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified output fields."
ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON list of dictionaries. The list may contain one or more dictionaries, and each dictionary should only have the specified output fields."

### REASONING INSTRUCTION FOR IMAGE PROMPTS ###
IMAGE_REASONING_SUFFIX = """Let's think step-by-step in order to answer the question.

REASONING: """

IMAGE_ANSWER_SUFFIX = """Let's think step-by-step in order to answer the question.

ANSWER: """

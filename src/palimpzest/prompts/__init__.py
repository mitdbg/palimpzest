from palimpzest.prompts.code_synthesis_prompts import ADVICEGEN_PROMPT, CODEGEN_PROMPT, EXAMPLE_PROMPT
from palimpzest.prompts.prompt_factory import PromptFactory
from palimpzest.prompts.util_phrases import (
    COT_ANSWER_INSTRUCTION,
    COT_REASONING_INSTRUCTION,
    ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION,
    ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION,
)

__all__ = [
    # code synthesis prompts
    "ADVICEGEN_PROMPT",
    "CODEGEN_PROMPT",
    "EXAMPLE_PROMPT",
    # prompt factory
    "PromptFactory",
    # util phrases
    "COT_ANSWER_INSTRUCTION",
    "COT_REASONING_INSTRUCTION",
    "ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION",
    "ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION",
]

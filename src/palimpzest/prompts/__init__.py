from palimpzest.prompts.code_synthesis_prompts import ADVICEGEN_PROMPT, CODEGEN_PROMPT, EXAMPLE_PROMPT
from palimpzest.prompts.convert_image_prompts import (
    COT_QA_IMAGE_BASE_SYSTEM_PROMPT,
    COT_QA_IMAGE_BASE_USER_PROMPT,
)
from palimpzest.prompts.format_strings import (
    IMAGE_ANSWER_SUFFIX,
    IMAGE_REASONING_SUFFIX,
    ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION,
    ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION,
)

__all__ = [
    # code synthesis prompts
    "ADVICEGEN_PROMPT",
    "CODEGEN_PROMPT",
    "EXAMPLE_PROMPT",
    # convert image prompts
    "COT_QA_IMAGE_BASE_SYSTEM_PROMPT",
    "COT_QA_IMAGE_BASE_USER_PROMPT",
    # format strings
    "IMAGE_ANSWER_SUFFIX",
    "IMAGE_REASONING_SUFFIX",
    "ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION",
    "ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION",
]

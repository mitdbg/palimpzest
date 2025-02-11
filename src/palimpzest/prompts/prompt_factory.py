"""This file contains factory methods which return templated prompts for the given input(s)."""
from palimpzest.constants import Cardinality, PromptStrategy
from palimpzest.prompts.convert_prompts import (
    COT_QA_BASE_SYSTEM_PROMPT,
    COT_QA_BASE_USER_PROMPT,
    COT_QA_EXAMPLE_ANSWER,
    COT_QA_EXAMPLE_CONTEXT,
    COT_QA_EXAMPLE_INPUT_FIELDS,
    COT_QA_EXAMPLE_OUTPUT_FIELDS,
    COT_QA_EXAMPLE_REASONING,
    COT_QA_IMAGE_EXAMPLE_ANSWER,
    COT_QA_IMAGE_EXAMPLE_CONTEXT,
    COT_QA_IMAGE_EXAMPLE_INPUT_FIELDS,
    COT_QA_IMAGE_EXAMPLE_OUTPUT_FIELDS,
    COT_QA_IMAGE_EXAMPLE_REASONING,
    COT_QA_IMAGE_JOB_INSTRUCTION,
    COT_QA_JOB_INSTRUCTION,
)


class PromptFactory:
    """Factory class for generating prompts for the Generator given the input(s)."""

    def __init__(self, prompt_strategy: PromptStrategy, is_image_conversion: bool, cardinality: Cardinality) -> None:
        self.prompt_strategy = prompt_strategy
        self.is_image_conversion = is_image_conversion
        self.cardinality = cardinality

    def get_system_prompt(self) -> str:
        pass

    def get_user_prompt(self) -> str:
        pass

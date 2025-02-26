"""This file contains factory methods which return template prompts and return messages for chat payloads."""

import base64
import json
from string import Formatter

from palimpzest.constants import (
    MIXTRAL_LLAMA_CONTEXT_TOKENS_LIMIT,
    TOKENS_PER_CHARACTER,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BytesField, ImageBase64Field, ImageFilepathField, ImageURLField
from palimpzest.core.lib.schemas import Schema
from palimpzest.prompts.convert_prompts import (
    COT_QA_BASE_SYSTEM_PROMPT,
    COT_QA_BASE_USER_PROMPT,
    COT_QA_EXAMPLE_ANSWER,
    COT_QA_EXAMPLE_CONTEXT,
    COT_QA_EXAMPLE_INPUT_FIELDS,
    COT_QA_EXAMPLE_OUTPUT_FIELDS,
    COT_QA_EXAMPLE_REASONING,
    COT_QA_IMAGE_DISCLAIMER,
    COT_QA_IMAGE_EXAMPLE_ANSWER,
    COT_QA_IMAGE_EXAMPLE_CONTEXT,
    COT_QA_IMAGE_EXAMPLE_INPUT_FIELDS,
    COT_QA_IMAGE_EXAMPLE_OUTPUT_FIELDS,
    COT_QA_IMAGE_EXAMPLE_REASONING,
    COT_QA_IMAGE_JOB_INSTRUCTION,
    COT_QA_JOB_INSTRUCTION,
)
from palimpzest.prompts.critique_and_refine_convert_prompts import (
    BASE_CRITIQUE_PROMPT,
    BASE_REFINEMENT_PROMPT,
    COT_QA_CRITIQUE_CRITERIA,
    COT_QA_CRITIQUE_FINISH_INSTRUCTION,
    COT_QA_IMAGE_CRITIQUE_CRITERIA,
    COT_QA_IMAGE_REFINEMENT_CRITERIA,
    COT_QA_REFINEMENT_CRITERIA,
    COT_QA_REFINEMENT_FINISH_INSTRUCTION,
)
from palimpzest.prompts.filter_prompts import (
    COT_BOOL_BASE_SYSTEM_PROMPT,
    COT_BOOL_BASE_USER_PROMPT,
    COT_BOOL_EXAMPLE_CONTEXT,
    COT_BOOL_EXAMPLE_FILTER_CONDITION,
    COT_BOOL_EXAMPLE_INPUT_FIELDS,
    COT_BOOL_EXAMPLE_REASONING,
    COT_BOOL_IMAGE_DISCLAIMER,
    COT_BOOL_IMAGE_EXAMPLE_CONTEXT,
    COT_BOOL_IMAGE_EXAMPLE_FILTER_CONDITION,
    COT_BOOL_IMAGE_EXAMPLE_INPUT_FIELDS,
    COT_BOOL_IMAGE_EXAMPLE_REASONING,
    COT_BOOL_IMAGE_JOB_INSTRUCTION,
    COT_BOOL_JOB_INSTRUCTION,
)
from palimpzest.prompts.moa_aggregator_convert_prompts import (
    COT_MOA_AGG_BASE_SYSTEM_PROMPT,
    COT_MOA_AGG_BASE_USER_PROMPT,
)
from palimpzest.prompts.moa_proposer_convert_prompts import (
    COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
    COT_MOA_PROPOSER_BASE_USER_PROMPT,
    COT_MOA_PROPOSER_EXAMPLE_ANSWER,
    COT_MOA_PROPOSER_EXAMPLE_CONTEXT,
    COT_MOA_PROPOSER_EXAMPLE_INPUT_FIELDS,
    COT_MOA_PROPOSER_EXAMPLE_OUTPUT_FIELDS,
    COT_MOA_PROPOSER_IMAGE_DISCLAIMER,
    COT_MOA_PROPOSER_IMAGE_EXAMPLE_ANSWER,
    COT_MOA_PROPOSER_IMAGE_EXAMPLE_CONTEXT,
    COT_MOA_PROPOSER_IMAGE_EXAMPLE_INPUT_FIELDS,
    COT_MOA_PROPOSER_IMAGE_EXAMPLE_OUTPUT_FIELDS,
    COT_MOA_PROPOSER_IMAGE_JOB_INSTRUCTION,
    COT_MOA_PROPOSER_JOB_INSTRUCTION,
)
from palimpzest.prompts.util_phrases import (
    ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION,
    ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION,
)


class PromptFactory:
    """Factory class for generating prompts for the Generator given the input(s)."""

    BASE_SYSTEM_PROMPT_MAP = {
        PromptStrategy.COT_BOOL: COT_BOOL_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_QA: COT_QA_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_QA_CRITIC: None,
        PromptStrategy.COT_QA_REFINE: None,
        PromptStrategy.COT_QA_IMAGE: COT_QA_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_QA_IMAGE_CRITIC: None,
        PromptStrategy.COT_QA_IMAGE_REFINE: None,
        PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.COT_MOA_AGG: COT_MOA_AGG_BASE_SYSTEM_PROMPT,
    }
    BASE_USER_PROMPT_MAP = {
        PromptStrategy.COT_BOOL: COT_BOOL_BASE_USER_PROMPT,
        PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_BASE_USER_PROMPT,
        PromptStrategy.COT_QA: COT_QA_BASE_USER_PROMPT,
        PromptStrategy.COT_QA_CRITIC: BASE_CRITIQUE_PROMPT,
        PromptStrategy.COT_QA_REFINE: BASE_REFINEMENT_PROMPT,
        PromptStrategy.COT_QA_IMAGE: COT_QA_BASE_USER_PROMPT,
        PromptStrategy.COT_QA_IMAGE_CRITIC: BASE_CRITIQUE_PROMPT,
        PromptStrategy.COT_QA_IMAGE_REFINE: BASE_REFINEMENT_PROMPT,
        PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.COT_MOA_AGG: COT_MOA_AGG_BASE_USER_PROMPT,
    }

    def __init__(self, prompt_strategy: PromptStrategy, model: Model, cardinality: Cardinality) -> None:
        self.prompt_strategy = prompt_strategy
        self.model = model
        self.cardinality = cardinality

    def _get_context(self, candidate: DataRecord, input_fields: list[str]) -> str:
        """
        Returns the context for the prompt.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The input fields.

        Returns:
            str: The context.
        """
        # get context from input record (project_cols will be None if not provided in kwargs)
        context: dict = candidate.to_dict(include_bytes=False, project_cols=input_fields)

        # TODO: MOVE THIS LOGIC INTO A CHUNKING / CONTEXT MANAGEMENT CLASS
        #   - this class should be able to:
        #      - handle the context length of different models (i.e. self.model should be an input)
        #      - handle images
        #      - handle the issue with `original_messages` (ask Matt if this is not clear)
        # TODO: this does not work for image prompts
        # TODO: this ignores the size of the `orignal_messages` in critique and refine prompts
        # cut down on context based on window length
        if self.model in [Model.LLAMA3, Model.MIXTRAL]:
            total_context_len = len(json.dumps(context, indent=2))

            # sort fields by length and progressively strip from the longest field until it is short enough;
            # NOTE: MIXTRAL_LLAMA_CONTEXT_TOKENS_LIMIT is a rough estimate which leaves room for the rest of the prompt text
            while total_context_len * TOKENS_PER_CHARACTER > MIXTRAL_LLAMA_CONTEXT_TOKENS_LIMIT:
                # sort fields by length
                field_lengths = [(field, len(value) if value is not None else 0) for field, value in context.items()]
                sorted_fields = sorted(field_lengths, key=lambda item: item[1], reverse=True)

                # get field with longest context
                longest_field_name, longest_field_length = sorted_fields[0]

                # trim the field
                context_factor = MIXTRAL_LLAMA_CONTEXT_TOKENS_LIMIT / (total_context_len * TOKENS_PER_CHARACTER)
                keep_frac_idx = int(longest_field_length * context_factor)
                context[longest_field_name] = context[longest_field_name][:keep_frac_idx]

                # update total context length
                total_context_len = len(json.dumps(context, indent=2))

        return json.dumps(context, indent=2)

    def _get_input_fields(self, candidate: DataRecord, **kwargs) -> list[str]:
        """
        The list of input fields to be templated into the prompt(s).
        If the user provides a list of "project_cols" in kwargs, then this list will be returned.
        Otherwise, this function returns the list of all field names in the candidate record.

        Args:
            candidate (DataRecord): The input record.
            kwargs: The keyword arguments provided by the user.

        Returns:
            list[str]: The list of input field names.
        """
        return kwargs.get("project_cols", candidate.get_field_names())

    def _get_input_fields_desc(self, candidate: DataRecord, input_fields: list[str]) -> str:
        """
        Returns a multi-line description of each input field for the prompt.

        Args:
            input_fields (list[str]): The input fields.

        Returns:
            str: The input fields description.
        """
        input_fields_desc = ""
        for field_name in input_fields:
            input_fields_desc += f"- {field_name}: {candidate.get_field_type(field_name)._desc}\n"

        return input_fields_desc[:-1]

    def _get_output_fields_desc(self, output_fields: list[str], **kwargs) -> str:
        """
        Returns a multi-line description of each output field for the prompt.

        Args:
            output_fields (list[str]): The output fields.
            kwargs: The keyword arguments provided by the user.

        Returns:
            str: The output fields description.
        """
        output_fields_desc = ""
        output_schema: Schema = kwargs.get("output_schema")
        if self.prompt_strategy.is_convert_prompt():
            assert output_schema is not None, "Output schema must be provided for convert prompts."

            field_desc_map = output_schema.field_desc_map()
            for field_name in output_fields:
                output_fields_desc += f"- {field_name}: {field_desc_map[field_name]}\n"

        # strip the last newline characters from the field descriptions and return
        return output_fields_desc[:-1]

    def _get_filter_condition(self, **kwargs) -> str | None:
        """
        Returns the filter condition for the filter operation.

        Returns:
            str | None: The filter condition (if applicable).
        """
        filter_condition = kwargs.get("filter_condition")
        if self.prompt_strategy.is_bool_prompt():
            assert filter_condition is not None, "Filter condition must be provided for filter operations."

        return filter_condition

    def _get_original_output(self, **kwargs) -> str | None:
        """
        Returns the original output from a previous model generation for the critique and refinement operations.

        Args:
            kwargs: The keyword arguments provided by the user.

        Returns:
            str | None: The original output.
        """
        original_output = kwargs.get("original_output")
        if self.prompt_strategy.is_critic_prompt() or self.prompt_strategy.is_refine_prompt():
            assert original_output is not None, (
                "Original output must be provided for critique and refinement operations."
            )

        return original_output

    def _get_critique_output(self, **kwargs) -> str | None:
        """
        Returns the critique output for the refinement operation.

        Args:
            kwargs: The keyword arguments provided by the user.

        Returns:
            str | None: The critique output.
        """
        critique_output = kwargs.get("critique_output")
        if self.prompt_strategy.is_refine_prompt():
            assert critique_output is not None, "Critique output must be provided for refinement operations."

        return critique_output

    def _get_model_responses(self, **kwargs) -> str | None:
        """
        Returns the model responses for the mixture-of-agents aggregation operation.

        Args:
            kwargs: The keyword arguments provided by the user.

        Returns:
            str | None: The model responses.
        """
        model_responses = None
        if self.prompt_strategy.is_moa_aggregator_prompt():
            model_responses = ""
            for idx, model_response in enumerate(kwargs.get("model_responses")):
                model_responses += f"MODEL RESPONSE {idx + 1}: {model_response}\n"

        return model_responses

    def _get_output_format_instruction(self) -> str:
        """
        Returns the output format instruction based on the cardinality.

        Returns:
            str: The output format instruction.
        """
        return (
            ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION
            if self.cardinality == Cardinality.ONE_TO_ONE
            else ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION
        )

    def _get_job_instruction(self) -> str | None:
        """
        Returns the job instruction based on the prompt strategy.

        Returns:
            str | None: The job instruction (if applicable).
        """
        prompt_strategy_to_job_instruction = {
            PromptStrategy.COT_BOOL: COT_BOOL_JOB_INSTRUCTION,
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_JOB_INSTRUCTION,
            PromptStrategy.COT_QA: COT_QA_JOB_INSTRUCTION,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_JOB_INSTRUCTION,
            PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_JOB_INSTRUCTION,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_JOB_INSTRUCTION,
        }
        return prompt_strategy_to_job_instruction.get(self.prompt_strategy)

    def _get_critique_criteria(self) -> str | None:
        """
        Returns the critique criteria for the critique operation.

        Returns:
            str | None: The critique criteria (if applicable).
        """
        critique_criteria = None
        if self.prompt_strategy.is_critic_prompt():
            critique_criteria = (
                COT_QA_IMAGE_CRITIQUE_CRITERIA if self.prompt_strategy.is_image_prompt() else COT_QA_CRITIQUE_CRITERIA
            )

        return critique_criteria

    def _get_refinement_criteria(self) -> str | None:
        """
        Returns the refinement criteria for the refinement operation.

        Returns:
            str | None: The refinement criteria (if applicable).
        """
        refinement_criteria = None
        if self.prompt_strategy.is_refine_prompt():
            refinement_criteria = (
                COT_QA_IMAGE_REFINEMENT_CRITERIA
                if self.prompt_strategy.is_image_prompt()
                else COT_QA_REFINEMENT_CRITERIA
            )

        return refinement_criteria

    def _get_finish_instruction(self) -> str | None:
        """
        Returns the finish instruction for the critique and refinement operations.

        Returns:
            str | None: The finish instruction (if applicable).
        """
        finish_instruction = None
        if self.prompt_strategy.is_critic_prompt():
            finish_instruction = COT_QA_CRITIQUE_FINISH_INSTRUCTION
        elif self.prompt_strategy.is_refine_prompt():
            finish_instruction = COT_QA_REFINEMENT_FINISH_INSTRUCTION

        return finish_instruction

    def _get_example_input_fields(self) -> str | None:
        """
        Returns the example input fields for the prompt.

        Returns:
            str | None: The example input fields (if applicable).
        """
        prompt_strategy_to_example_input_fields = {
            PromptStrategy.COT_BOOL: COT_BOOL_EXAMPLE_INPUT_FIELDS,
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_EXAMPLE_INPUT_FIELDS,
            PromptStrategy.COT_QA: COT_QA_EXAMPLE_INPUT_FIELDS,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_EXAMPLE_INPUT_FIELDS,
            PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_EXAMPLE_INPUT_FIELDS,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_EXAMPLE_INPUT_FIELDS,
        }

        return prompt_strategy_to_example_input_fields.get(self.prompt_strategy)

    def _get_example_output_fields(self) -> str | None:
        """
        Returns the example output fields for the prompt.

        Returns:
            str | None: The example output fields (if applicable).
        """
        prompt_strategy_to_example_output_fields = {
            PromptStrategy.COT_QA: COT_QA_EXAMPLE_OUTPUT_FIELDS,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_EXAMPLE_OUTPUT_FIELDS,
            PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_EXAMPLE_OUTPUT_FIELDS,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_EXAMPLE_OUTPUT_FIELDS,
        }

        return prompt_strategy_to_example_output_fields.get(self.prompt_strategy)

    def _get_example_context(self) -> str | None:
        """
        Returns the example context for the prompt.

        Returns:
            str | None: The example context (if applicable).
        """
        prompt_strategy_to_example_context = {
            PromptStrategy.COT_BOOL: COT_BOOL_EXAMPLE_CONTEXT,
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_EXAMPLE_CONTEXT,
            PromptStrategy.COT_QA: COT_QA_EXAMPLE_CONTEXT,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_EXAMPLE_CONTEXT,
            PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_EXAMPLE_CONTEXT,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_EXAMPLE_CONTEXT,
        }

        return prompt_strategy_to_example_context.get(self.prompt_strategy)

    def _get_image_disclaimer(self) -> str:
        """
        Returns the image disclaimer for the prompt. The disclaimer must be an empty string
        for text prompts.

        Returns:
            str: The image disclaimer. If this is a text prompt then it is an empty string.
        """
        prompt_strategy_to_image_disclaimer = {
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_DISCLAIMER,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_DISCLAIMER,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_DISCLAIMER,
        }

        return prompt_strategy_to_image_disclaimer.get(self.prompt_strategy, "")

    def _get_example_filter_condition(self) -> str | None:
        """
        Returns the example filter condition for the prompt.

        Returns:
            str | None: The example filter condition (if applicable).
        """
        prompt_strategy_to_example_filter_condition = {
            PromptStrategy.COT_BOOL: COT_BOOL_EXAMPLE_FILTER_CONDITION,
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_EXAMPLE_FILTER_CONDITION,
        }

        return prompt_strategy_to_example_filter_condition.get(self.prompt_strategy)

    def _get_example_reasoning(self) -> str | None:
        """
        Returns the example reasoning for the prompt.

        Returns:
            str | None: The example reasoning (if applicable).
        """
        prompt_strategy_to_example_reasoning = {
            PromptStrategy.COT_BOOL: COT_BOOL_EXAMPLE_REASONING,
            PromptStrategy.COT_BOOL_IMAGE: COT_BOOL_IMAGE_EXAMPLE_REASONING,
            PromptStrategy.COT_QA: COT_QA_EXAMPLE_REASONING,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_EXAMPLE_REASONING,
        }

        return prompt_strategy_to_example_reasoning.get(self.prompt_strategy)

    def _get_example_answer(self) -> str | None:
        """
        Returns the example answer for the prompt.

        Returns:
            str | None: The example answer (if applicable).
        """
        prompt_strategy_to_example_answer = {
            PromptStrategy.COT_QA: COT_QA_EXAMPLE_ANSWER,
            PromptStrategy.COT_QA_IMAGE: COT_QA_IMAGE_EXAMPLE_ANSWER,
            PromptStrategy.COT_MOA_PROPOSER: COT_MOA_PROPOSER_EXAMPLE_ANSWER,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE: COT_MOA_PROPOSER_IMAGE_EXAMPLE_ANSWER,
        }

        return prompt_strategy_to_example_answer.get(self.prompt_strategy)

    def _get_all_format_kwargs(
        self, candidate: DataRecord, input_fields: list[str], output_fields: list[str], **kwargs
    ) -> dict:
        """
        Returns a dictionary containing all the format kwargs for templating the prompts.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The input fields.
            output_fields (list[str]): The output fields.
            kwargs: The keyword arguments provided by the user.

        Returns:
            dict: The dictionary containing all the format kwargs.
        """
        # get format kwargs which depend on the input data
        input_format_kwargs = {
            "context": self._get_context(candidate, input_fields),
            "input_fields_desc": self._get_input_fields_desc(candidate, input_fields),
            "output_fields_desc": self._get_output_fields_desc(output_fields, **kwargs),
            "filter_condition": self._get_filter_condition(**kwargs),
            "original_output": self._get_original_output(**kwargs),
            "critique_output": self._get_critique_output(**kwargs),
            "model_responses": self._get_model_responses(**kwargs),
        }

        # get format kwargs which depend on the prompt strategy
        prompt_strategy_format_kwargs = {
            "output_format_instruction": self._get_output_format_instruction(),
            "job_instruction": self._get_job_instruction(),
            "critique_criteria": self._get_critique_criteria(),
            "refinement_criteria": self._get_refinement_criteria(),
            "finish_instruction": self._get_finish_instruction(),
            "example_input_fields": self._get_example_input_fields(),
            "example_output_fields": self._get_example_output_fields(),
            "example_context": self._get_example_context(),
            "image_disclaimer": self._get_image_disclaimer(),
            "example_filter_condition": self._get_example_filter_condition(),
            "example_reasoning": self._get_example_reasoning(),
            "example_answer": self._get_example_answer(),
        }

        # return all format kwargs
        return {**input_format_kwargs, **prompt_strategy_format_kwargs}

    def _create_image_messages(self, candidate: DataRecord, input_fields: list[str]) -> list[dict]:
        """
        Parses the candidate record and returns the image messages for the chat payload.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The list of input fields.

        Returns:
            list[dict]: The image messages for the chat payload.
        """
        # create a message for each image in an input field with an image (or list of image) type
        image_messages = []
        for field_name in input_fields:
            field_value = candidate[field_name]
            field_type = candidate.get_field_type(field_name)

            # image filepath (or list of image filepaths)
            if isinstance(field_type, ImageFilepathField):
                with open(field_value, "rb") as f:
                    base64_image_str = base64.b64encode(f.read()).decode("utf-8")
                image_messages.append(
                    {"role": "user", "type": "image", "content": f"data:image/jpeg;base64,{base64_image_str}"}
                )

            elif hasattr(field_type, "element_type") and issubclass(field_type.element_type, ImageFilepathField):
                for image_filepath in field_value:
                    with open(image_filepath, "rb") as f:
                        base64_image_str = base64.b64encode(f.read()).decode("utf-8")
                    image_messages.append(
                        {"role": "user", "type": "image", "content": f"data:image/jpeg;base64,{base64_image_str}"}
                    )

            # image url (or list of image urls)
            elif isinstance(field_type, ImageURLField):
                image_messages.append({"role": "user", "type": "image", "content": field_value})

            elif hasattr(field_type, "element_type") and issubclass(field_type.element_type, ImageURLField):
                for image_url in field_value:
                    image_messages.append({"role": "user", "type": "image", "content": image_url})

            # pre-encoded images (or list of pre-encoded images)
            elif isinstance(field_type, ImageBase64Field):
                base64_image_str = field_value.decode("utf-8")
                image_messages.append(
                    {"role": "user", "type": "image", "content": f"data:image/jpeg;base64,{base64_image_str}"}
                )

            elif hasattr(field_type, "element_type") and issubclass(field_type.element_type, ImageBase64Field):
                for base64_image in field_value:
                    base64_image_str = base64_image.decode("utf-8")
                    image_messages.append(
                        {"role": "user", "type": "image", "content": f"data:image/jpeg;base64,{base64_image_str}"}
                    )

        return image_messages

    def _get_system_prompt(self, **format_kwargs) -> str | None:
        """
        Returns the fully templated system prompt for the given prompt strategy.
        Returns None if the prompt strategy does not use a system prompt.

        Returns:
            str | None: The fully templated system prompt (or None if the prompt strategy
                does not use a system prompt).
        """
        base_prompt: str = self.BASE_SYSTEM_PROMPT_MAP.get(self.prompt_strategy)

        # for critic and refine prompt strategies, we do not use a base prompt
        if base_prompt is None:
            return base_prompt

        return base_prompt.format(**format_kwargs)

    def _get_user_messages(self, candidate: DataRecord, input_fields: list[str], **kwargs) -> str:
        """
        Returns a list of messages for the chat payload based on the prompt strategy.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The input fields.
            output_fields (list[str]): The output fields.
            kwargs: The formatting kwargs and some keyword arguments provided by the user.

        Returns:
            Tuple[str, str | None]: The fully templated start and end of the user prompt.
                The second element will be None for text prompts.
        """
        # get the base prompt template
        base_prompt = self.BASE_USER_PROMPT_MAP.get(self.prompt_strategy)

        # get any image messages for the chat payload (will be an empty list if this is not an image prompt)
        image_messages = (
            self._create_image_messages(candidate, input_fields) if self.prompt_strategy.is_image_prompt() else []
        )

        # get any original messages for critique and refinement operations
        original_messages = kwargs.get("original_messages")
        if self.prompt_strategy.is_critic_prompt() or self.prompt_strategy.is_refine_prompt():
            assert original_messages is not None, (
                "Original messages must be provided for critique and refinement operations."
            )

        # construct the user messages based on the prompt strategy
        user_messages = []
        if self.prompt_strategy.is_critic_prompt() or self.prompt_strategy.is_refine_prompt():
            # NOTE: if this critic / refinement prompt is processing images, those images will
            #       be part of the `original_messages` and will show up in the final chat payload
            base_prompt_start, base_prompt_end = base_prompt.split("<<original-prompt-placeholder>>\n")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(original_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        elif self.prompt_strategy.is_image_prompt():
            base_prompt_start, base_prompt_end = base_prompt.split("<<image-placeholder>>\n")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(image_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        else:
            base_prompt = base_prompt.replace("<<image-placeholder>>", "")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt.format(**kwargs)})

        return user_messages

    def _process_custom_user_prompt(self, candidate: DataRecord, input_fields: list[str], **kwargs) -> list[dict]:
        """
        Processes a custom user prompt provided by the user.

        Args:
            candidate (DataRecord): The input record.
            kwargs: The keyword arguments provided by the user.

        Returns:
            list[dict]: The messages for the chat payload.
        """
        # get the user prompt
        user_prompt: str = kwargs["prompt"]

        # sanity check that we have all the inputs for the user's prompt template
        prompt_field_names = [fname for _, fname, _, _ in Formatter().parse(user_prompt) if fname]
        fields_check = all([field in input_fields for field in prompt_field_names])
        if not fields_check:
            if sorted(candidate.get_field_names()) != (input_fields):
                err_msg = (
                    f"Prompt string has fields which are not in input fields.\n"
                    f"Prompt fields: {prompt_field_names}\n"
                    f"Computed fields: {candidate.get_field_names()}\n"
                    f"Input fields: {input_fields}\n"
                    f"Be careful that you are not projecting out computed fields. "
                    f"If you use `depends_on` in your program, make sure it includes the fields you need."
                )
            else:
                err_msg = (
                    f"Prompt string has fields which are not in input fields.\n"
                    f"Prompt fields: {prompt_field_names}\n"
                    f"Input fields: {input_fields}\n"
                )
            assert fields_check, err_msg

        # build set of format kwargs
        format_kwargs = {
            field_name: "<bytes>"
            if isinstance(candidate.get_field_type(field_name), BytesField)
            else candidate[field_name]
            for field_name in input_fields
        }

        # split prompt on <<image-placeholder>> if it exists
        if "<<image-placeholder>>" in user_prompt:
            raise NotImplementedError("Image prompts are not yet supported.")

        prompt_sections = user_prompt.split("<<image-placeholder>>")
        messages = [{"role": "user", "type": "text", "content": prompt_sections[0].format(**format_kwargs)}]

        # NOTE: this currently assumes that the user can only provide a single <<image-placeholder>>
        if len(prompt_sections) > 1:
            image_messages = self._create_image_messages(candidate, input_fields)
            messages.extend(image_messages)
            messages.append({"role": "user", "type": "text", "content": prompt_sections[1].format(**format_kwargs)})

        return messages

    def create_messages(self, candidate: DataRecord, output_fields: list[str], **kwargs) -> list[dict]:
        """
        Creates the messages for the chat payload based on the prompt strategy.

        Each message will be a dictionary with the following format:
        {
            "role": "user" | "system",
            "type": "text" | "image",
            "content": str
        }

        Args:
            candidate (DataRecord): The input record.
            output_fields (list[str]): The output fields.
            kwargs: The keyword arguments provided by the user.

        Returns:
            list[dict]: The messages for the chat payload.
        """
        # compute the set of input fields
        input_fields = self._get_input_fields(candidate, **kwargs)

        # if the user provides a prompt, we process that prompt into messages and return them
        if "prompt" in kwargs:
            messages = []
            if "system_prompt" in kwargs:
                messages.append({"role": "system", "type": "text", "content": kwargs["system_prompt"]})
            messages.extend(self._process_custom_user_prompt(candidate, input_fields, **kwargs))
            return messages

        # initialize messages
        messages = []

        # compute the full dictionary of format kwargs and add to kwargs
        format_kwargs = self._get_all_format_kwargs(candidate, input_fields, output_fields, **kwargs)
        kwargs = {**kwargs, **format_kwargs}

        # generate system message (if applicable)
        system_prompt = self._get_system_prompt(**kwargs)
        if system_prompt is not None:
            messages.append({"role": "system", "type": "text", "content": system_prompt})

        # generate user messages and add to messages
        user_messages = self._get_user_messages(candidate, input_fields, **kwargs)
        messages.extend(user_messages)

        return messages

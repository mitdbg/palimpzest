"""This file contains factory methods which return template prompts and return messages for chat payloads."""

import base64
import json
from typing import Any

from pydantic import BaseModel

from palimpzest.constants import (
    LLAMA_CONTEXT_TOKENS_LIMIT,
    TOKENS_PER_CHARACTER,
    Cardinality,
    Modality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import (
    AUDIO_FIELD_TYPES,
    IMAGE_FIELD_TYPES,
    AudioBase64,
    AudioFilepath,
    ImageBase64,
    ImageFilepath,
    ImageURL,
)
from palimpzest.prompts.aggregate_prompts import (
    AGG_BASE_SYSTEM_PROMPT,
    AGG_BASE_USER_PROMPT,
    AGG_NO_REASONING_BASE_SYSTEM_PROMPT,
    AGG_NO_REASONING_BASE_USER_PROMPT,
)
from palimpzest.prompts.convert_prompts import (
    MAP_BASE_SYSTEM_PROMPT,
    MAP_BASE_USER_PROMPT,
    MAP_NO_REASONING_BASE_SYSTEM_PROMPT,
    MAP_NO_REASONING_BASE_USER_PROMPT,
)
from palimpzest.prompts.critique_and_refine_prompts import (
    BASE_CRITIQUE_PROMPT,
    BASE_REFINEMENT_PROMPT,
    FILTER_CRITIQUE_CRITERIA,
    FILTER_CRITIQUE_FINISH_INSTRUCTION,
    FILTER_REFINEMENT_CRITERIA,
    FILTER_REFINEMENT_FINISH_INSTRUCTION,
    MAP_CRITIQUE_CRITERIA,
    MAP_CRITIQUE_FINISH_INSTRUCTION,
    MAP_REFINEMENT_CRITERIA,
    MAP_REFINEMENT_FINISH_INSTRUCTION,
)
from palimpzest.prompts.filter_prompts import (
    FILTER_BASE_SYSTEM_PROMPT,
    FILTER_BASE_USER_PROMPT,
    FILTER_NO_REASONING_BASE_SYSTEM_PROMPT,
    FILTER_NO_REASONING_BASE_USER_PROMPT,
)
from palimpzest.prompts.join_prompts import (
    JOIN_BASE_SYSTEM_PROMPT,
    JOIN_BASE_USER_PROMPT,
    JOIN_NO_REASONING_BASE_SYSTEM_PROMPT,
    JOIN_NO_REASONING_BASE_USER_PROMPT,
)
from palimpzest.prompts.moa_aggregator_prompts import (
    FILTER_MOA_AGG_BASE_SYSTEM_PROMPT,
    FILTER_MOA_AGG_BASE_USER_PROMPT,
    MAP_MOA_AGG_BASE_SYSTEM_PROMPT,
    MAP_MOA_AGG_BASE_USER_PROMPT,
)
from palimpzest.prompts.moa_proposer_prompts import (
    FILTER_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
    FILTER_MOA_PROPOSER_BASE_USER_PROMPT,
    MAP_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
    MAP_MOA_PROPOSER_BASE_USER_PROMPT,
)
from palimpzest.prompts.split_merge_prompts import (
    FILTER_SPLIT_MERGER_BASE_SYSTEM_PROMPT,
    FILTER_SPLIT_MERGER_BASE_USER_PROMPT,
    MAP_SPLIT_MERGER_BASE_SYSTEM_PROMPT,
    MAP_SPLIT_MERGER_BASE_USER_PROMPT,
)
from palimpzest.prompts.split_proposer_prompts import (
    FILTER_SPLIT_PROPOSER_BASE_SYSTEM_PROMPT,
    FILTER_SPLIT_PROPOSER_BASE_USER_PROMPT,
    MAP_SPLIT_PROPOSER_BASE_SYSTEM_PROMPT,
    MAP_SPLIT_PROPOSER_BASE_USER_PROMPT,
)
from palimpzest.prompts.utils import (
    AGG_AUDIO_DISCLAIMER,
    AGG_EXAMPLE_ANSWER,
    AGG_EXAMPLE_OUTPUT_FIELDS,
    AGG_EXAMPLE_REASONING,
    AGG_IMAGE_DISCLAIMER,
    AGG_JOB_INSTRUCTION,
    AUDIO_DISCLAIMER,
    AUDIO_EXAMPLE_ANSWER,
    AUDIO_EXAMPLE_CONTEXT,
    AUDIO_EXAMPLE_INPUT_FIELDS,
    AUDIO_EXAMPLE_OUTPUT_FIELDS,
    AUDIO_EXAMPLE_REASONING,
    AUDIO_SENTENCE_EXAMPLE_ANSWER,
    DESC_SECTION,
    EXAMPLE_AGG_INSTRUCTION,
    EXAMPLE_FILTER_CONDITION,
    EXAMPLE_JOIN_CONDITION,
    FILTER_EXAMPLE_REASONING,
    FILTER_JOB_INSTRUCTION,
    IMAGE_DISCLAIMER,
    IMAGE_EXAMPLE_ANSWER,
    IMAGE_EXAMPLE_CONTEXT,
    IMAGE_EXAMPLE_INPUT_FIELDS,
    IMAGE_EXAMPLE_OUTPUT_FIELDS,
    IMAGE_EXAMPLE_REASONING,
    IMAGE_SENTENCE_EXAMPLE_ANSWER,
    JOIN_EXAMPLE_REASONING,
    JOIN_JOB_INSTRUCTION,
    MAP_JOB_INSTRUCTION,
    ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION,
    ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION,
    PROPOSER_JOB_INSTRUCTION,
    RIGHT_AUDIO_DISCLAIMER,
    RIGHT_AUDIO_EXAMPLE_CONTEXT,
    RIGHT_AUDIO_EXAMPLE_INPUT_FIELDS,
    RIGHT_IMAGE_DISCLAIMER,
    RIGHT_IMAGE_EXAMPLE_CONTEXT,
    RIGHT_IMAGE_EXAMPLE_INPUT_FIELDS,
    RIGHT_TEXT_EXAMPLE_CONTEXT,
    RIGHT_TEXT_EXAMPLE_INPUT_FIELDS,
    SECOND_AUDIO_EXAMPLE_CONTEXT,
    SECOND_IMAGE_EXAMPLE_CONTEXT,
    SECOND_TEXT_EXAMPLE_CONTEXT,
    TEXT_EXAMPLE_ANSWER,
    TEXT_EXAMPLE_CONTEXT,
    TEXT_EXAMPLE_INPUT_FIELDS,
    TEXT_EXAMPLE_OUTPUT_FIELDS,
    TEXT_EXAMPLE_REASONING,
    TEXT_SENTENCE_EXAMPLE_ANSWER,
    THIRD_AUDIO_EXAMPLE_CONTEXT,
    THIRD_IMAGE_EXAMPLE_CONTEXT,
    THIRD_TEXT_EXAMPLE_CONTEXT,
)


class PromptFactory:
    """Factory class for generating prompts for the Generator given the input(s)."""

    BASE_SYSTEM_PROMPT_MAP = {
        # agg user prompts
        PromptStrategy.AGG: AGG_BASE_SYSTEM_PROMPT,
        PromptStrategy.AGG_NO_REASONING: AGG_NO_REASONING_BASE_SYSTEM_PROMPT,

        # filter system prompts
        PromptStrategy.FILTER: FILTER_BASE_SYSTEM_PROMPT,
        PromptStrategy.FILTER_NO_REASONING: FILTER_NO_REASONING_BASE_SYSTEM_PROMPT,
        PromptStrategy.FILTER_CRITIC: None,
        PromptStrategy.FILTER_REFINE: None,
        PromptStrategy.FILTER_MOA_PROPOSER: FILTER_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.FILTER_MOA_AGG: FILTER_MOA_AGG_BASE_SYSTEM_PROMPT,
        PromptStrategy.FILTER_SPLIT_PROPOSER: FILTER_SPLIT_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.FILTER_SPLIT_MERGER: FILTER_SPLIT_MERGER_BASE_SYSTEM_PROMPT,

        # join system prompts
        PromptStrategy.JOIN: JOIN_BASE_SYSTEM_PROMPT,
        PromptStrategy.JOIN_NO_REASONING: JOIN_NO_REASONING_BASE_SYSTEM_PROMPT,

        # map system prompts
        PromptStrategy.MAP: MAP_BASE_SYSTEM_PROMPT,
        PromptStrategy.MAP_NO_REASONING: MAP_NO_REASONING_BASE_SYSTEM_PROMPT,
        PromptStrategy.MAP_CRITIC: None,
        PromptStrategy.MAP_REFINE: None,
        PromptStrategy.MAP_MOA_PROPOSER: MAP_MOA_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.MAP_MOA_AGG: MAP_MOA_AGG_BASE_SYSTEM_PROMPT,
        PromptStrategy.MAP_SPLIT_PROPOSER: MAP_SPLIT_PROPOSER_BASE_SYSTEM_PROMPT,
        PromptStrategy.MAP_SPLIT_MERGER: MAP_SPLIT_MERGER_BASE_SYSTEM_PROMPT,
    }
    BASE_USER_PROMPT_MAP = {
        # agg user prompts
        PromptStrategy.AGG: AGG_BASE_USER_PROMPT,
        PromptStrategy.AGG_NO_REASONING: AGG_NO_REASONING_BASE_USER_PROMPT,

        # filter user prompts
        PromptStrategy.FILTER: FILTER_BASE_USER_PROMPT,
        PromptStrategy.FILTER_NO_REASONING: FILTER_NO_REASONING_BASE_USER_PROMPT,
        PromptStrategy.FILTER_CRITIC: BASE_CRITIQUE_PROMPT,
        PromptStrategy.FILTER_REFINE: BASE_REFINEMENT_PROMPT,
        PromptStrategy.FILTER_MOA_PROPOSER: FILTER_MOA_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.FILTER_MOA_AGG: FILTER_MOA_AGG_BASE_USER_PROMPT,
        PromptStrategy.FILTER_SPLIT_PROPOSER: FILTER_SPLIT_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.FILTER_SPLIT_MERGER: FILTER_SPLIT_MERGER_BASE_USER_PROMPT,

        # join user prompts
        PromptStrategy.JOIN: JOIN_BASE_USER_PROMPT,
        PromptStrategy.JOIN_NO_REASONING: JOIN_NO_REASONING_BASE_USER_PROMPT,

        # map user prompts
        PromptStrategy.MAP: MAP_BASE_USER_PROMPT,
        PromptStrategy.MAP_NO_REASONING: MAP_NO_REASONING_BASE_USER_PROMPT,
        PromptStrategy.MAP_CRITIC: BASE_CRITIQUE_PROMPT,
        PromptStrategy.MAP_REFINE: BASE_REFINEMENT_PROMPT,
        PromptStrategy.MAP_MOA_PROPOSER: MAP_MOA_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.MAP_MOA_AGG: MAP_MOA_AGG_BASE_USER_PROMPT,
        PromptStrategy.MAP_SPLIT_PROPOSER: MAP_SPLIT_PROPOSER_BASE_USER_PROMPT,
        PromptStrategy.MAP_SPLIT_MERGER: MAP_SPLIT_MERGER_BASE_USER_PROMPT,
    }

    def __init__(self, prompt_strategy: PromptStrategy, model: Model, cardinality: Cardinality, desc: str | None = None) -> None:
        self.prompt_strategy = prompt_strategy
        self.model = model
        self.cardinality = cardinality
        self.desc = desc

    def _get_context(self, candidate: DataRecord | list[DataRecord], input_fields: list[str]) -> str:
        """
        Returns the context for the prompt.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The input fields.

        Returns:
            str: The context.
        """
        # TODO: remove mask_filepaths=True after SemBench evaluation
        # get context from input record (project_cols will be None if not provided in kwargs)
        if isinstance(candidate, list):
            context: list[dict] = [record.to_dict(include_bytes=False, project_cols=input_fields, mask_filepaths=True) for record in candidate]
        else:
            context: dict = candidate.to_dict(include_bytes=False, project_cols=input_fields, mask_filepaths=True)

        # TODO: MOVE THIS LOGIC INTO A CHUNKING / CONTEXT MANAGEMENT CLASS
        #   - this class should be able to:
        #      - handle the context length of different models (i.e. self.model should be an input)
        #      - handle images
        #      - handle the issue with `original_messages` (ask Matt if this is not clear)
        # TODO: this does not work for image prompts
        # TODO: this ignores the size of the `orignal_messages` in critique and refine prompts
        # NOTE: llama models are disallowed for aggregation so we can assume context is a dict here
        # cut down on context based on window length
        if self.model.is_llama_model():
            assert isinstance(context, dict), "Llama models are not allowed for aggregation operations."
            total_context_len = len(json.dumps(context, indent=2))

            # sort fields by length and progressively strip from the longest field until it is short enough;
            # NOTE: LLAMA_CONTEXT_TOKENS_LIMIT is a rough estimate which leaves room for the rest of the prompt text
            while total_context_len * TOKENS_PER_CHARACTER > LLAMA_CONTEXT_TOKENS_LIMIT:
                # sort fields by length
                field_lengths = [(field, len(value) if value is not None else 0) for field, value in context.items()]
                sorted_fields = sorted(field_lengths, key=lambda item: item[1], reverse=True)

                # get field with longest context
                longest_field_name, longest_field_length = sorted_fields[0]

                # trim the field
                context_factor = LLAMA_CONTEXT_TOKENS_LIMIT / (total_context_len * TOKENS_PER_CHARACTER)
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
        # NOTE: joins will include left and right input fields in project_cols, so we have to check
        #       if the field is in the candidate record
        input_fields = kwargs.get("project_cols", candidate.get_field_names())
        input_fields = [field for field in input_fields if field in candidate.get_field_names()]
        return input_fields

    def _get_input_modalities(self, candidate: DataRecord, input_fields: list[str]) -> set[Modality]:
        """
        The list of input modalities for the given input fields.

        Args:
            candidate (DataRecord): The input record.
            input_fields (list[str]): The input fields.

        Returns:
            set[Modality]: The list of input modalities.
        """
        input_modalities = []
        for field_name in input_fields:
            field_type = candidate.get_field_type(field_name)
            if field_type.annotation in IMAGE_FIELD_TYPES:
                input_modalities.append(Modality.IMAGE)
            elif field_type.annotation in AUDIO_FIELD_TYPES:
                input_modalities.append(Modality.AUDIO)
            else:
                input_modalities.append(Modality.TEXT)

        return set(input_modalities)

    def _get_modalities_str(self, input_modalities: set[Modality]) -> str:
        """
        Returns a format string to reflect the input modalities.

        Args:
            input_modalities (set[Modality]): The input modalities.

        Returns:
            str: The string to reflect the input modalities.
        """
        if input_modalities == {Modality.TEXT}:
            return "text"
        elif input_modalities == {Modality.IMAGE}:
            return "image(s)"
        elif input_modalities == {Modality.AUDIO}:
            return "audio"
        elif input_modalities == {Modality.TEXT, Modality.IMAGE}:
            return "text and/or image(s)"
        elif input_modalities == {Modality.TEXT, Modality.AUDIO}:
            return "text and/or audio"
        elif input_modalities == {Modality.IMAGE, Modality.AUDIO}:
            return "image(s) and/or audio"
        elif input_modalities == {Modality.TEXT, Modality.IMAGE, Modality.AUDIO}:
            return "text, image(s), and/or audio"

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
            input_fields_desc += f"- {field_name}: {candidate.get_field_type(field_name).description}\n"

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
        output_schema: type[BaseModel] = kwargs.get("output_schema")
        if self.prompt_strategy.is_map_prompt() or self.prompt_strategy.is_agg_prompt():
            assert output_schema is not None, "Output schema must be provided for convert prompts."

            for field_name in sorted(output_fields):
                desc = output_schema.model_fields[field_name].description
                output_fields_desc += f"- {field_name}: {'no description available' if desc is None else desc}\n"

        # strip the last newline characters from the field descriptions and return
        return output_fields_desc[:-1]

    def _get_agg_instruction(self, **kwargs) -> str | None:
        """
        Returns the aggregation instruction for the aggregation operation.

        Returns:
            str | None: The aggregation instruction (if applicable).
        """
        agg_instruction = kwargs.get("agg_instruction")
        if self.prompt_strategy.is_agg_prompt():
            assert agg_instruction is not None, "Aggregation instruction must be provided for aggregation operations."

        return agg_instruction

    def _get_filter_condition(self, **kwargs) -> str | None:
        """
        Returns the filter condition for the filter operation.

        Returns:
            str | None: The filter condition (if applicable).
        """
        filter_condition = kwargs.get("filter_condition")
        if self.prompt_strategy.is_filter_prompt():
            assert filter_condition is not None, "Filter condition must be provided for filter operations."

        return filter_condition

    def _get_join_condition(self, **kwargs) -> str | None:
        """
        Returns the join condition for the join operation.

        Returns:
            str | None: The join condition (if applicable).
        """
        join_condition = kwargs.get("join_condition")
        if self.prompt_strategy.is_join_prompt():
            assert join_condition is not None, "Join condition must be provided for join operations."

        return join_condition

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
                model_responses += f"MODEL RESPONSE {idx + 1}: {model_response.rstrip()}\n\n"
        model_responses = model_responses.rstrip() if model_responses is not None else None

        return model_responses

    def _get_chunk_outputs(self, **kwargs) -> str | None:
        """
        Returns the chunk outputs for the split-convert.

        Args:
            kwargs: The keyword arguments provided by the user.

        Returns:
            str | None: The chunk outputs.
        """
        chunk_outputs = None
        if self.prompt_strategy.is_split_merger_prompt():
            chunk_outputs = ""
            for idx, chunk_output in enumerate(kwargs.get("chunk_outputs")):
                chunk_outputs += f"CHUNK OUTPUT {idx + 1}: {chunk_output.rstrip()}\n\n"
        chunk_outputs = chunk_outputs.rstrip() if chunk_outputs is not None else None

        return chunk_outputs

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

    def _get_job_instruction(self, input_modalities: set[Modality]) -> str | None:
        """
        Returns the job instruction based on the prompt strategy.

        Args:
            input_modalities (set[Modality]): The modalities of the input fields.

        Returns:
            str | None: The job instruction.
        """
        # get the job instruction based on the prompt strategy
        job_instruction = None
        if self.prompt_strategy.is_moa_proposer_prompt() or self.prompt_strategy.is_split_proposer_prompt():
            job_instruction = PROPOSER_JOB_INSTRUCTION
        elif self.prompt_strategy.is_map_prompt():
            job_instruction = MAP_JOB_INSTRUCTION
        elif self.prompt_strategy.is_filter_prompt():
            job_instruction = FILTER_JOB_INSTRUCTION
        elif self.prompt_strategy.is_join_prompt():
            job_instruction = JOIN_JOB_INSTRUCTION
        elif self.prompt_strategy.is_agg_prompt():
            job_instruction = AGG_JOB_INSTRUCTION

        # format the job instruction based on the input modalities
        modalities = self._get_modalities_str(input_modalities)
        if job_instruction is not None:
            job_instruction = job_instruction.format(modalities=modalities)

        return job_instruction

    def _get_desc_section(self) -> str:
        """
        Returns the description section for the prompt.

        Returns:
            str: The description section (if applicable).
        """
        desc_section = ""
        if self.desc is not None:
            desc_section = DESC_SECTION.format(desc=self.desc)

        return desc_section

    def _get_critique_criteria(self) -> str | None:
        """
        Returns the critique criteria for the critique operation.

        Returns:
            str | None: The critique criteria (if applicable).
        """
        critique_criteria = None
        if self.prompt_strategy.is_critic_prompt():
            critique_criteria = MAP_CRITIQUE_CRITERIA if self.prompt_strategy.is_map_prompt() else FILTER_CRITIQUE_CRITERIA

        return critique_criteria

    def _get_refinement_criteria(self) -> str | None:
        """
        Returns the refinement criteria for the refinement operation.

        Returns:
            str | None: The refinement criteria (if applicable).
        """
        refinement_criteria = None
        if self.prompt_strategy.is_refine_prompt():
            refinement_criteria = MAP_REFINEMENT_CRITERIA if self.prompt_strategy.is_map_prompt() else FILTER_REFINEMENT_CRITERIA

        return refinement_criteria

    def _get_finish_instruction(self) -> str | None:
        """
        Returns the finish instruction for the critique and refinement operations.

        Returns:
            str | None: The finish instruction (if applicable).
        """
        finish_instruction = None
        if self.prompt_strategy.is_critic_prompt():
            finish_instruction = MAP_CRITIQUE_FINISH_INSTRUCTION if self.prompt_strategy.is_map_prompt() else FILTER_CRITIQUE_FINISH_INSTRUCTION
        elif self.prompt_strategy.is_refine_prompt():
            finish_instruction = MAP_REFINEMENT_FINISH_INSTRUCTION if self.prompt_strategy.is_map_prompt() else FILTER_REFINEMENT_FINISH_INSTRUCTION

        return finish_instruction

    def _get_example_input_fields(self, input_modalities: set[Modality], right: bool = False) -> str:
        """
        Returns the example input fields for the prompt.

        Args:
            input_modalities (set[Modality]): The modalities of the input fields.
            right (bool): Whether to return the right input fields for the join prompt.

        Returns:
            str: The example input fields.
        """
        input_modality_to_example_input_fields = {
            Modality.TEXT: RIGHT_TEXT_EXAMPLE_INPUT_FIELDS if right else TEXT_EXAMPLE_INPUT_FIELDS,
            Modality.IMAGE: RIGHT_IMAGE_EXAMPLE_INPUT_FIELDS if right else IMAGE_EXAMPLE_INPUT_FIELDS,
            Modality.AUDIO: RIGHT_AUDIO_EXAMPLE_INPUT_FIELDS if right else AUDIO_EXAMPLE_INPUT_FIELDS,
        }

        example_input_fields = ""
        for input_modality in input_modalities:
            example_input_fields += input_modality_to_example_input_fields[input_modality].rstrip()
        example_input_fields = example_input_fields.lstrip() + "\n"

        return example_input_fields

    def _get_example_output_fields(self, input_modalities: set[Modality]) -> str:
        """
        Returns the example output fields for the prompt.

        Returns:
            str: The example output fields.
        """
        if self.prompt_strategy.is_agg_prompt():
            return AGG_EXAMPLE_OUTPUT_FIELDS

        input_modality_to_example_output_fields = {
            Modality.TEXT: TEXT_EXAMPLE_OUTPUT_FIELDS,
            Modality.IMAGE: IMAGE_EXAMPLE_OUTPUT_FIELDS,
            Modality.AUDIO: AUDIO_EXAMPLE_OUTPUT_FIELDS,
        }

        example_output_fields = ""
        for input_modality in input_modalities:
            example_output_fields += input_modality_to_example_output_fields[input_modality].rstrip()
        example_output_fields = example_output_fields.lstrip() + "\n"

        return example_output_fields

    def _get_example_context(self, input_modalities: set[Modality], right: bool = False, second: bool = False, third: bool = False) -> str:
        """
        Returns the example context for the prompt.

        Returns:
            str: The example context.
        """
        assert not (second and third), "Cannot have both second and third example contexts."
        assert not (right and (second or third)), "Right context is only used for joins; second and third contexts only use for aggregations."
        text_example_context = TEXT_EXAMPLE_CONTEXT
        image_example_context = IMAGE_EXAMPLE_CONTEXT
        audio_example_context = AUDIO_EXAMPLE_CONTEXT
        if second:
            text_example_context = SECOND_TEXT_EXAMPLE_CONTEXT
            image_example_context = SECOND_IMAGE_EXAMPLE_CONTEXT
            audio_example_context = SECOND_AUDIO_EXAMPLE_CONTEXT
        elif third:
            text_example_context = THIRD_TEXT_EXAMPLE_CONTEXT
            image_example_context = THIRD_IMAGE_EXAMPLE_CONTEXT
            audio_example_context = THIRD_AUDIO_EXAMPLE_CONTEXT

        input_modality_to_example_context = {
            Modality.TEXT: RIGHT_TEXT_EXAMPLE_CONTEXT if right else text_example_context,
            Modality.IMAGE: RIGHT_IMAGE_EXAMPLE_CONTEXT if right else image_example_context,
            Modality.AUDIO: RIGHT_AUDIO_EXAMPLE_CONTEXT if right else audio_example_context,
        }

        example_context = ""
        for input_modality in input_modalities:
            example_context += input_modality_to_example_context[input_modality].rstrip() + ","
        example_context = example_context[:-1] + "\n"

        return example_context

    def _get_image_disclaimer(self, input_modalities: set[Modality], right: bool = False, agg: bool = False) -> str:
        """
        Returns the image disclaimer for the prompt. The disclaimer must be an empty string
        for non-image prompts.

        Returns:
            str: The image disclaimer. If this is a text prompt then it is an empty string.
        """
        assert not (right and agg), "Right image disclaimer is only used for joins; agg image disclaimer only used for aggregations."
        image_disclaimer = AGG_IMAGE_DISCLAIMER if agg else IMAGE_DISCLAIMER
        image_disclaimer = RIGHT_IMAGE_DISCLAIMER if right else image_disclaimer
        return image_disclaimer if Modality.IMAGE in input_modalities else ""

    def _get_audio_disclaimer(self, input_modalities: set[Modality], right: bool = False, agg: bool = False) -> str:
        """
        Returns the audio disclaimer for the prompt. The disclaimer must be an empty string
        for non-audio prompts.

        Returns:
            str: The audio disclaimer. If this is a text prompt then it is an empty string.
        """
        assert not (right and agg), "Right audio disclaimer is only used for joins; agg audio disclaimer only used for aggregations."
        audio_disclaimer = AGG_AUDIO_DISCLAIMER if agg else AUDIO_DISCLAIMER
        audio_disclaimer = RIGHT_AUDIO_DISCLAIMER if right else audio_disclaimer
        return audio_disclaimer if Modality.AUDIO in input_modalities else ""

    def _get_example_reasoning(self, input_modalities: set[Modality]) -> str:
        """
        Returns the example reasoning for the prompt.

        Returns:
            str: The example reasoning.
        """
        if self.prompt_strategy.is_filter_prompt():
            return FILTER_EXAMPLE_REASONING
        elif self.prompt_strategy.is_join_prompt():
            return JOIN_EXAMPLE_REASONING
        elif self.prompt_strategy.is_agg_prompt():
            return AGG_EXAMPLE_REASONING

        input_modality_to_example_reasoning = {
            Modality.TEXT: TEXT_EXAMPLE_REASONING,
            Modality.IMAGE: IMAGE_EXAMPLE_REASONING,
            Modality.AUDIO: AUDIO_EXAMPLE_REASONING,
        }

        example_reasoning = ""
        for input_modality in input_modalities:
            example_reasoning += input_modality_to_example_reasoning[input_modality] + " "
        example_reasoning = example_reasoning.rstrip()

        return example_reasoning

    def _get_example_answer(self, input_modalities: set[Modality]) -> str:
        """
        Returns the example answer for the prompt.

        Returns:
            str: The example answer.
        """
        if self.prompt_strategy.is_agg_prompt():
            return AGG_EXAMPLE_ANSWER

        use_sentence_answers = self.prompt_strategy.is_split_proposer_prompt() or self.prompt_strategy.is_moa_proposer_prompt()
        input_modality_to_example_answer = {
            Modality.TEXT: TEXT_SENTENCE_EXAMPLE_ANSWER if use_sentence_answers else TEXT_EXAMPLE_ANSWER,
            Modality.IMAGE: IMAGE_SENTENCE_EXAMPLE_ANSWER if use_sentence_answers else IMAGE_EXAMPLE_ANSWER,
            Modality.AUDIO: AUDIO_SENTENCE_EXAMPLE_ANSWER if use_sentence_answers else AUDIO_EXAMPLE_ANSWER,
        }

        example_answer = ""
        for input_modality in input_modalities:
            example_answer += input_modality_to_example_answer[input_modality].rstrip()
            if use_sentence_answers:
                example_answer += " "
        example_answer = example_answer + "\n"

        return example_answer

    def _get_all_format_kwargs(
        self,
        candidate: DataRecord | list[DataRecord],
        input_fields: list[str],
        input_modalities: set[Modality],
        output_fields: list[str],
        right_candidate: DataRecord | None,
        right_input_fields: list[str],
        right_input_modalities: set[Modality],
        **kwargs,
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
            "input_fields_desc": self._get_input_fields_desc(candidate[0] if isinstance(candidate, list) else candidate, input_fields),
            "output_fields_desc": self._get_output_fields_desc(output_fields, **kwargs),
            "agg_instruction": self._get_agg_instruction(**kwargs),
            "filter_condition": self._get_filter_condition(**kwargs),
            "join_condition": self._get_join_condition(**kwargs),
            "original_output": self._get_original_output(**kwargs),
            "critique_output": self._get_critique_output(**kwargs),
            "model_responses": self._get_model_responses(**kwargs),
            "chunk_outputs": self._get_chunk_outputs(**kwargs),
        }

        # if a right candidate is provided, we also get the context and input field descriptions for the right candidate
        if right_candidate is not None:
            input_format_kwargs.update({
                "right_context": self._get_context(right_candidate, right_input_fields),
                "right_input_fields_desc": self._get_input_fields_desc(right_candidate, right_input_fields),
            })

        # get format kwargs which depend on the prompt strategy
        full_input_modalities = input_modalities.union(right_input_modalities)
        prompt_strategy_format_kwargs = {
            "output_format_instruction": self._get_output_format_instruction(),
            "job_instruction": self._get_job_instruction(full_input_modalities),
            "desc_section": self._get_desc_section(),
            "critique_criteria": self._get_critique_criteria(),
            "refinement_criteria": self._get_refinement_criteria(),
            "finish_instruction": self._get_finish_instruction(),
            "example_input_fields": self._get_example_input_fields(input_modalities),
            "right_example_input_fields": self._get_example_input_fields(right_input_modalities, right=True),
            "example_output_fields": self._get_example_output_fields(input_modalities),
            "example_context": self._get_example_context(input_modalities),
            "second_example_context": self._get_example_context(input_modalities, second=True) if self.prompt_strategy.is_agg_prompt() else "",
            "third_example_context": self._get_example_context(input_modalities, third=True) if self.prompt_strategy.is_agg_prompt() else "",
            "right_example_context": self._get_example_context(right_input_modalities, right=True),
            "image_disclaimer": self._get_image_disclaimer(input_modalities, agg=self.prompt_strategy.is_agg_prompt()),
            "audio_disclaimer": self._get_audio_disclaimer(input_modalities, agg=self.prompt_strategy.is_agg_prompt()),
            "right_image_disclaimer": self._get_image_disclaimer(right_input_modalities, right=True),
            "right_audio_disclaimer": self._get_audio_disclaimer(right_input_modalities, right=True),
            "example_agg_instruction": EXAMPLE_AGG_INSTRUCTION,
            "example_filter_condition": EXAMPLE_FILTER_CONDITION,
            "example_join_condition": EXAMPLE_JOIN_CONDITION,
            "example_reasoning": self._get_example_reasoning(input_modalities),
            "example_answer": self._get_example_answer(input_modalities),
        }

        # return all format kwargs
        return {**input_format_kwargs, **prompt_strategy_format_kwargs}

    def _create_audio_messages(self, candidate: DataRecord | list[DataRecord], input_fields: list[str]) -> list[dict]:
        """
        Parses the candidate record(s) and returns the audio messages for the chat payload.

        Args:
            candidate (DataRecord | list[DataRecord]): The input record(s).
            input_fields (list[str]): The list of input fields.

        Returns:
            list[dict]: The audio messages for the chat payload.
        """
        # normalize type to be list[DataRecord]
        if isinstance(candidate, DataRecord):
            candidate = [candidate]

        # create a message for each audio recording in an input field with an audio (or list of audio) type
        audio_content = []
        for field_name in input_fields:
            for dr in candidate:
                field_value = dr[field_name]
                field_type = dr.get_field_type(field_name)

                # audio filepath (or list of audio filepaths)
                if field_type.annotation in [AudioFilepath, AudioFilepath | None, AudioFilepath | Any] and field_value is not None:
                    with open(field_value, "rb") as f:
                        base64_audio_str = base64.b64encode(f.read()).decode("utf-8")
                    audio_content.append(
                        {"type": "input_audio", "input_audio": {"data": base64_audio_str, "format": "wav"}}
                    )

                elif field_type.annotation in [list[AudioFilepath], list[AudioFilepath] | None, list[AudioFilepath] | Any]:
                    for audio_filepath in field_value:
                        if audio_filepath is None:
                            continue
                        with open(audio_filepath, "rb") as f:
                            base64_audio_str = base64.b64encode(f.read()).decode("utf-8")
                        audio_content.append(
                            {"type": "input_audio", "input_audio": {"data": base64_audio_str, "format": "wav"}}
                        )

                # pre-encoded images (or list of pre-encoded images)
                elif field_type.annotation in [AudioBase64, AudioBase64 | None, AudioBase64 | Any] and field_value is not None:
                    audio_content.append(
                        {"type": "input_audio", "input_audio": {"data": field_value, "format": "wav"}}
                    )

                elif field_type.annotation in [list[AudioBase64], list[AudioBase64] | None, list[AudioBase64] | Any]:
                    for base64_audio in field_value:
                        if base64_audio is None:
                            continue
                        audio_content.append(
                            {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}
                        )

        return [{"role": "user", "type": "input_audio", "content": audio_content}] if len(audio_content) > 0 else []

    def _create_image_messages(self, candidate: DataRecord | list[DataRecord], input_fields: list[str]) -> list[dict]:
        """
        Parses the candidate record(s) and returns the image messages for the chat payload.

        Args:
            candidate (DataRecord | list[DataRecord]): The input record(s).
            input_fields (list[str]): The list of input fields.

        Returns:
            list[dict]: The image messages for the chat payload.
        """
        # normalize type to be list[DataRecord]
        if isinstance(candidate, DataRecord):
            candidate = [candidate]

        # create a message for each image in an input field with an image (or list of image) type
        image_content = []
        for field_name in input_fields:
            for dr in candidate:
                field_value = dr[field_name]
                field_type = dr.get_field_type(field_name)

                # image filepath (or list of image filepaths)
                if field_type.annotation in [ImageFilepath, ImageFilepath | None, ImageFilepath | Any] and field_value is not None:
                    with open(field_value, "rb") as f:
                        base64_image_str = base64.b64encode(f.read()).decode("utf-8")
                    image_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}}
                    )

                elif field_type.annotation in [list[ImageFilepath], list[ImageFilepath] | None, list[ImageFilepath] | Any]:
                    for image_filepath in field_value:
                        if image_filepath is None:
                            continue
                        with open(image_filepath, "rb") as f:
                            base64_image_str = base64.b64encode(f.read()).decode("utf-8")
                        image_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}}
                        )

                # image url (or list of image urls)
                elif field_type.annotation in [ImageURL, ImageURL | None, ImageURL | Any] and field_value is not None:
                    image_content.append({"type": "image_url", "image_url": {"url": field_value}})

                elif field_type.annotation in [list[ImageURL], list[ImageURL] | None, list[ImageURL] | Any]:
                    for image_url in field_value:
                        if image_url is None:
                            continue
                        image_content.append({"type": "image_url", "image_url": {"url": image_url}})

                # pre-encoded images (or list of pre-encoded images)
                elif field_type.annotation in [ImageBase64, ImageBase64 | None, ImageBase64 | Any] and field_value is not None:
                    image_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{field_value}"}}
                    )

                elif field_type.annotation in [list[ImageBase64], list[ImageBase64] | None, list[ImageBase64] | Any]:
                    for base64_image in field_value:
                        if base64_image is None:
                            continue
                        image_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        )

        return [{"role": "user", "type": "image", "content": image_content}] if len(image_content) > 0 else []

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

    def _get_user_messages(self, candidate: DataRecord | list[DataRecord], input_fields: list[str], right_candidate: DataRecord | None, right_input_fields: list[str], **kwargs) -> str:
        """
        Returns a list of messages for the chat payload based on the prompt strategy.

        Args:
            candidate (DataRecord | list[DataRecord]): The input record(s).
            input_fields (list[str]): The input fields.
            output_fields (list[str]): The output fields.
            kwargs: The formatting kwargs and some keyword arguments provided by the user.

        Returns:
            Tuple[str, str | None]: The fully templated start and end of the user prompt.
                The second element will be None for text prompts.
        """
        # get the base prompt template
        base_prompt = self.BASE_USER_PROMPT_MAP.get(self.prompt_strategy)

        # get any image messages for the chat payload (will be an empty list if no image fields exist)
        image_messages = self._create_image_messages(candidate, input_fields)

        # get any audio messages for the chat payload (will be an empty list if no audio fields exist)
        audio_messages = self._create_audio_messages(candidate, input_fields)

        # get any right image / audio messages for the chat payload (will be an empty list if image / audio not present)
        right_image_messages, right_audio_messages = [], []
        if self.prompt_strategy.is_join_prompt():
            assert right_candidate is not None, "Right candidate must be provided for join prompts."
            right_image_messages = self._create_image_messages(right_candidate, right_input_fields)
            right_audio_messages = self._create_audio_messages(right_candidate, right_input_fields)

        # get any original messages for critique and refinement operations
        original_messages = kwargs.get("original_messages")
        if self.prompt_strategy.is_critic_prompt() or self.prompt_strategy.is_refine_prompt():
            assert original_messages is not None, (
                "Original messages must be provided for critique and refinement operations."
            )

        # combine image and audio messages
        image_audio_messages = image_messages + audio_messages
        right_image_audio_messages = right_image_messages + right_audio_messages
        has_image_audio = len(image_audio_messages) > 0
        has_right_image_audio = len(right_image_audio_messages) > 0

        # construct the user messages based on the prompt strategy
        user_messages = []
        if self.prompt_strategy.is_critic_prompt() or self.prompt_strategy.is_refine_prompt():
            # NOTE: if this critic / refinement prompt is processing images / audio, those images / audio
            # will be part of the `original_messages` and will show up in the final chat payload
            base_prompt_start, base_prompt_end = base_prompt.split("<<original-prompt-placeholder>>\n")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(original_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        # handle joins with left and right images / audio
        elif self.prompt_strategy.is_join_prompt() and has_image_audio and has_right_image_audio:
            base_prompt_start, base_prompt_rest = base_prompt.split("<<image-audio-placeholder>>")
            base_prompt_mid, base_prompt_end = base_prompt_rest.split("<<right-image-audio-placeholder>>")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(image_audio_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_mid.format(**kwargs)})
            user_messages.extend(right_image_audio_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        # handle joins with only left images / audio
        elif self.prompt_strategy.is_join_prompt() and has_image_audio and not has_right_image_audio:
            base_prompt = base_prompt.replace("<<right-image-audio-placeholder>>", "")
            base_prompt_start, base_prompt_end = base_prompt.split("<<image-audio-placeholder>>")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(image_audio_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        # handle joins with only right images / audio
        elif self.prompt_strategy.is_join_prompt() and not has_image_audio and has_right_image_audio:
            base_prompt = base_prompt.replace("<<image-audio-placeholder>>", "")
            base_prompt_start, base_prompt_end = base_prompt.split("<<right-image-audio-placeholder>>")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(right_image_audio_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        # handle non-joins with images / audio
        elif not self.prompt_strategy.is_join_prompt() and has_image_audio and not self.prompt_strategy.is_moa_aggregator_prompt():
            base_prompt_start, base_prompt_end = base_prompt.split("<<image-audio-placeholder>>")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_start.format(**kwargs)})
            user_messages.extend(image_audio_messages)
            user_messages.append({"role": "user", "type": "text", "content": base_prompt_end.format(**kwargs)})

        # handle prompts w/no images or audio
        else:
            base_prompt = base_prompt.replace("<<image-audio-placeholder>>", "")
            base_prompt = base_prompt.replace("<<right-image-audio-placeholder>>", "")
            user_messages.append({"role": "user", "type": "text", "content": base_prompt.format(**kwargs)})

        return user_messages

    def create_messages(self, candidate: DataRecord | list[DataRecord], output_fields: list[str], right_candidate: DataRecord | None = None, **kwargs) -> list[dict]:
        """
        Creates the messages for the chat payload based on the prompt strategy.

        Each message will be a dictionary with the following format:
        {
            "role": "user" | "system",
            "type": "text" | "image",
            "content": str
        }

        Args:
            candidate (DataRecord | list[DataRecord]): The input record(s).
            output_fields (list[str]): The output fields.
            right_candidate (DataRecord | None): The other join input record (only provided for joins).
            kwargs: The keyword arguments provided by the user.

        Returns:
            list[dict]: The messages for the chat payload.
        """
        # compute the set of input fields
        input_fields = self._get_input_fields(candidate[0] if isinstance(candidate, list) else candidate, **kwargs)
        right_input_fields = [] if right_candidate is None else self._get_input_fields(right_candidate, **kwargs)

        # use input fields to determine the left / right input modalities
        input_modalities = self._get_input_modalities(candidate[0] if isinstance(candidate, list) else candidate, input_fields)
        right_input_modalities = set() if right_candidate is None else self._get_input_modalities(right_candidate, right_input_fields)

        # initialize messages
        messages = []

        # compute the full dictionary of format kwargs and add to kwargs
        format_kwargs = self._get_all_format_kwargs(candidate, input_fields, input_modalities, output_fields, right_candidate, right_input_fields, right_input_modalities, **kwargs)
        kwargs = {**kwargs, **format_kwargs}

        # generate system message (if applicable)
        system_prompt = self._get_system_prompt(**kwargs)
        if system_prompt is not None:
            messages.append({"role": "system", "type": "text", "content": system_prompt})

        # generate user messages and add to messages
        user_messages = self._get_user_messages(candidate, input_fields, right_candidate, right_input_fields, **kwargs)
        messages.extend(user_messages)

        return messages

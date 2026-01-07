"""
This file contains the Generator classes and generator factory.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from copy import deepcopy
from typing import Any, Generic, TypeVar

import litellm
import regex as re  # Use regex instead of re to used variable length lookbehind
from colorama import Fore, Style
from pydantic.fields import FieldInfo

from palimpzest.constants import (
    MODEL_CARDS,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats
from palimpzest.prompts import PromptFactory

# DEFINITIONS
GenerationOutput = tuple[dict, str | None, GenerationStats, list[dict]]
ContextType = TypeVar("ContextType")
InputType = TypeVar("InputType")


logger = logging.getLogger(__name__)

def get_json_from_answer(answer: str, model: Model, cardinality: Cardinality) -> dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    # model-specific trimming for LLAMA3 responses
    if model.is_llama_model():
        answer = answer.split("---")[0]
        answer = answer.replace("True", "true")
        answer = answer.replace("False", "false")

    # split off context / excess, which models sometimes output after answer
    answer = answer.split("Context:")[0]
    answer = answer.split("# this is the answer")[0]

    # trim the answer to only include the JSON dictionary
    if cardinality == Cardinality.ONE_TO_ONE:
        if not answer.strip().startswith("{"):
            # Find the start index of the actual JSON string assuming the prefix is followed by the JSON dictionary
            start_index = answer.find("{")
            if start_index != -1:
                # Remove the prefix and any leading characters before the JSON starts
                answer = answer[start_index:]

        if not answer.strip().endswith("}"):
            # Find the end index of the actual JSON string assuming the suffix is preceded by the JSON dictionary
            end_index = answer.rfind("}")
            if end_index != -1:
                # Remove the suffix and any trailing characters after the JSON ends
                answer = answer[: end_index + 1]

    # otherwise, trim the answer to only include the JSON array
    else:
        if not answer.strip().startswith("["):
            # Find the start index of the actual JSON string assuming the prefix is followed by the JSON array
            start_index = answer.find("[")
            if start_index != -1:
                # Remove the prefix and any leading characters before the JSON starts
                answer = answer[start_index:]

        if not answer.strip().endswith("]"):
            # Find the end index of the actual JSON string
            # assuming the suffix is preceded by the JSON object/array
            end_index = answer.rfind("]")
            if end_index != -1:
                # Remove the suffix and any trailing characters after the JSON ends
                answer = answer[: end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace(r"\_", "_")
    answer = answer.replace("\\n", "\n")
    # Remove https and http prefixes to not conflict with comment detection
    # Handle comments in the JSON response. Use regex from // until end of line
    answer = re.sub(r"(?<!https?:)\/\/.*?$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r",\n.*\.\.\.$", "", answer, flags=re.MULTILINE)
    # Sanitize newlines in the JSON response
    answer = answer.replace("\n", " ")

    # finally, parse and return the JSON object; errors are handled by the caller
    return json.loads(answer)

# TODO: push parallelism of generations into LiteLLM rather than threadpool in executor
# TODO: make sure answer parsing works with custom prompts / parsers (can defer this)
class Generator(Generic[ContextType, InputType]):
    """
    Class for generating new fields for a record using an LLM.
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy,
        reasoning_effort: str | None,
        api_base: str | None = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        desc: str | None = None,
        verbose: bool = False,
    ):
        self.model = model
        self.model_name = model.value
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        self.api_base = api_base
        self.desc = desc
        self.verbose = verbose
        self.prompt_factory = PromptFactory(prompt_strategy, model, cardinality, desc)

    def _parse_reasoning(self, completion_text: str, **kwargs) -> str:
        """Extract the reasoning for the generated output from the completion object."""
        # use a custom reasoning parser if provided
        if kwargs.get("parse_reasoning"):
            parse_reasoning_fn = kwargs.get("parse_reasoning")
            return parse_reasoning_fn(completion_text)

        # if the model followed the default instructions, the completion text will have reasoning
        # before the "ANSWER:"; if this is the case, we simply extract and return that full section
        if "answer" in completion_text.lower():
            regex = re.compile("(.*?)answer:.*", re.IGNORECASE | re.DOTALL)
            matches = regex.findall(completion_text)
            if len(matches) > 0:
                return matches[0].strip()

        # otherwise, return the full completion text
        return completion_text

    def _prepare_field_answers(self, field_answers: dict | list[dict], fields: dict[str, FieldInfo]) -> dict[str, list]:
        """
        field_answers is a dictionary mapping fields to their values. For one-to-one converts, wrap each
        answer in a list. For one-to-many converts, invert the list of dictionaries into a dictionary with
        list values.
        """
        # if this is a one-to-one convert, we need to wrap each answer in a list
        if self.cardinality == Cardinality.ONE_TO_ONE:
            field_answers = {field_name: [field_answers[field_name]] for field_name in fields}

        # otherwise, we need to invert the list of dictionaries into a dictionary with list values
        else:
            field_answers_lst: list[dict] = deepcopy(field_answers)

            field_answers = {field_name: [] for field_name in fields}
            for answer_dict in field_answers_lst:
                for field_name in fields:
                    answer = answer_dict.get(field_name, None)
                    field_answers[field_name].append(answer)

        return field_answers

    def _check_convert_answer_text(self, answer_text: str, fields: dict[str, FieldInfo], throw_exception: bool=False) -> dict | list[dict] | None:
        """
        Try parsing the answer text into a JSON object. If the parsing fails, return None.
        """
        try:
            # extract json from the answer text
            field_answers = get_json_from_answer(answer_text, self.model, self.cardinality)

            # prepare the field answers to match the expected output and return
            return self._prepare_field_answers(field_answers, fields)

        except Exception as e:
            if throw_exception:
                raise e

        return None

    def _check_bool_answer_text(self, answer_text: str, throw_exception: bool=False) -> dict | None:
        """
        Return {"passed_operator": True} if and only if "true" is in the answer text.
        Return {"passed_operator": False} if and only if "false" is in the answer text.
        Otherwise, raise an exception.
        """
        # NOTE: we may be able to eliminate this condition by specifying this JSON output in the prompt;
        # however, that would also need to coincide with a change to allow the parse_answer_fn to set "passed_operator"
        if "true" in answer_text.lower():
            return {"passed_operator": True}
        elif "false" in answer_text.lower():
            return {"passed_operator": False}

        if throw_exception:
            raise Exception(f"Could not parse answer from completion text: {answer_text}")

        return None

    def _parse_convert_answer(self, completion_text: str, fields: dict[str, FieldInfo], json_output: bool) -> dict[str, list]:
        """Extract the answer from the completion object for convert operations."""
        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()

            # if we don't expect a JSON output, return the answer text as is
            if not json_output:
                return answer_text

            # otherwise, try to parse the answer text into a JSON object
            field_answers = self._check_convert_answer_text(answer_text, fields)
            if field_answers is not None:
                return field_answers

        # if the first regex didn't find an answer, try taking all the text after "ANSWER:"
        regex = re.compile("answer:(.*)", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()

            # if we don't expect a JSON output, return the answer text as is
            if not json_output:
                return answer_text
            
            # otherwise, try to parse the answer text into a JSON object
            field_answers = self._check_convert_answer_text(answer_text, fields)
            if field_answers is not None:
                return field_answers

        # finally, try taking all of the text; for JSON output, throw an exception if parsing fails
        if not json_output:
            return completion_text

        return self._check_convert_answer_text(completion_text, fields, throw_exception=True)

    def _parse_bool_answer(self, completion_text: str, json_output: bool) -> dict[str, list]:
        """Extract the answer from the completion object for filter and join operations."""
        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()

            # if we don't expect a JSON output, return the answer text as is
            if not json_output:
                return answer_text

            # otherwise, try to parse the answer text into a JSON object
            field_answers = self._check_bool_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # if the first regex didn't find an answer, try taking all the text after "ANSWER:"
        regex = re.compile("answer:(.*)", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()

            # if we don't expect a JSON output, return the answer text as is
            if not json_output:
                return answer_text

            # otherwise, try to parse the answer text into a JSON object
            field_answers = self._check_bool_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # finally, try taking all of the text; for JSON output, throw an exception if parsing fails
        if not json_output:
            return completion_text

        return self._check_bool_answer_text(completion_text, throw_exception=True)

    def _parse_answer(self, completion_text: str, fields: dict[str, FieldInfo] | None, json_output: bool, **kwargs) -> dict[str, list]:
        """Extract the answer from the completion object."""
        # use a custom answer parser if provided
        if kwargs.get("parse_answer"):
            parse_answer_fn = kwargs.get("parse_answer")
            return parse_answer_fn(completion_text)

        # fields should be a dict if a custom answer parser is not provided
        assert isinstance(fields, dict), "Fields must be provided if a custom answer parser is not provided."

        # extract the per-field answers from the completion text
        field_answers = (
            self._parse_bool_answer(completion_text, json_output)
            if self.prompt_strategy.is_filter_prompt() or self.prompt_strategy.is_join_prompt()
            else self._parse_convert_answer(completion_text, fields, json_output)
        )

        return field_answers

    def __call__(self, candidate: DataRecord | list[DataRecord], fields: dict[str, FieldInfo] | None, right_candidate: DataRecord | None = None, json_output: bool=True, **kwargs) -> GenerationOutput:
        """Take the input record(s) (`candidate`), generate the output `fields`, and return the generated output."""
        logger.debug(f"Generating for candidate(s) {candidate} with fields {fields}")

        # fields can only be None if the user provides an answer parser
        fields_check = fields is not None or "parse_answer" in kwargs
        assert fields_check, "`fields` must be provided if `parse_answer` function is not provided in kwargs."

        # if the user (or operator) provides a system prompt instead of a prompt, treat this as
        # the prompt and print a warning
        if "system_prompt" in kwargs and "prompt" not in kwargs:
            kwargs["prompt"] = kwargs["system_prompt"]
            kwargs.pop("system_prompt")
            warnings.warn("Provided `system_prompt` without providing `prompt`; setting `prompt` = `system_prompt`.")  # noqa: B028

        # generate a list of messages which can be used to construct a payload
        messages = self.prompt_factory.create_messages(candidate, fields, right_candidate, **kwargs)
        is_audio_op = any(msg.get("type") == "input_audio" for msg in messages)

        # generate the text completion
        start_time = time.time()
        completion = None
        try:
            completion_kwargs = {}
            if not self.model.is_o_model() and not self.model.is_gpt_5_model():
                completion_kwargs = {"temperature": kwargs.get("temperature", 0.0), **completion_kwargs}
            if is_audio_op:
                completion_kwargs = {"modalities": ["text"], **completion_kwargs}
            if self.model.is_reasoning_model():
                completion_kwargs = {"reasoning_effort": self.reasoning_effort, **completion_kwargs}
            if self.model.is_vllm_model():
                completion_kwargs = {"api_base": self.api_base, "api_key": os.environ.get("VLLM_API_KEY", "fake-api-key"), **completion_kwargs}
            completion = litellm.completion(model=self.model_name, messages=messages, **completion_kwargs)
            end_time = time.time()
            logger.debug(f"Generated completion in {end_time - start_time:.2f} seconds")
        # if there's an error generating the completion, we have to return an empty answer
        # and can only account for the time spent performing the failed generation
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            field_answers = (
                {"passed_operator": False}
                if self.prompt_strategy.is_filter_prompt() or self.prompt_strategy.is_join_prompt()
                else {field_name: None for field_name in fields}
            )
            reasoning = None
            generation_stats = GenerationStats(
                model_name=self.model_name,
                llm_call_duration_secs=time.time() - start_time,
                total_llm_calls=1,
            )

            return field_answers, reasoning, generation_stats, messages

        # parse usage statistics and create the GenerationStats
        generation_stats = None
        if completion is not None:
            usage = completion.usage.model_dump()

            # get cost per input/output token for the model
            usd_per_input_token = MODEL_CARDS[self.model_name].get("usd_per_input_token", 0.0)
            usd_per_audio_input_token = MODEL_CARDS[self.model_name].get("usd_per_audio_input_token", 0.0)
            usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]

            # TODO: for some models (e.g. GPT-5) we cannot separate text from image prompt tokens yet;
            #       for now, we only use tokens from prompt_token_details if it's an audio prompt
            # get output tokens (all text) and input tokens by modality
            output_tokens = usage["completion_tokens"]
            if is_audio_op:
                input_audio_tokens = usage["prompt_tokens_details"].get("audio_tokens", 0)
                input_text_tokens = usage["prompt_tokens_details"].get("text_tokens", 0)
                input_image_tokens = 0
            else:
                input_audio_tokens = 0
                input_text_tokens = usage["prompt_tokens"]
                input_image_tokens = 0
            input_tokens = input_audio_tokens + input_text_tokens + input_image_tokens

            # compute the input and output token costs
            total_input_cost = (input_text_tokens + input_image_tokens) * usd_per_input_token + input_audio_tokens * usd_per_audio_input_token
            total_output_cost = output_tokens * usd_per_output_token

            generation_stats = GenerationStats(
                model_name=self.model_name,
                llm_call_duration_secs=end_time - start_time,
                fn_call_duration_secs=0.0,
                input_audio_tokens=input_audio_tokens,
                input_text_tokens=input_text_tokens,
                input_image_tokens=input_image_tokens,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                total_input_cost=total_input_cost,
                total_output_cost=total_output_cost,
                cost_per_record=total_input_cost + total_output_cost,
                total_llm_calls=1,
            )

        # pretty print prompt + full completion output for debugging
        completion_text = completion.choices[0].message.content
        prompt, system_prompt = "", ""
        for message in messages:
            if message["role"] == "system":
                system_prompt += message["content"] + "\n"
            if message["role"] == "user":
                if message["type"] == "text":
                    prompt += message["content"] + "\n"
                elif message["type"] == "image":
                    prompt += "<image>\n" * len(message["content"])
                elif message["type"] == "input_audio":
                    prompt += "<audio>\n" * len(message["content"])
        logger.debug(f"PROMPT:\n{prompt}")
        logger.debug(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

        # parse reasoning
        reasoning = None
        try:
            reasoning = self._parse_reasoning(completion_text, **kwargs)
        except Exception as e:
            logger.error(f"Error parsing reasoning and answers: {e}")
            pass

        # parse field answers
        field_answers = None 
        if fields is not None and (self.prompt_strategy.is_filter_prompt() or self.prompt_strategy.is_join_prompt()):
            field_answers = {"passed_operator": False}
        elif fields is not None and not (self.prompt_strategy.is_filter_prompt() or self.prompt_strategy.is_join_prompt()):
            field_answers = {field_name: None for field_name in fields}
        try:
            field_answers = self._parse_answer(completion_text, fields, json_output, **kwargs)
        except Exception as e:
            logger.error(f"Error parsing answers: {e}")
            os.makedirs("parse-answer-errors", exist_ok=True)
            ts = time.time()
            with open(f"parse-answer-errors/error-{ts}.txt", "w") as f:
                f.write(f"{str(self.model_name)}\n")
                f.write("#####\n")
                f.write(f"{str(self.prompt_strategy)}\n")
                f.write("#####\n")
                f.write(f"{str(completion_text)}\n")
                f.write("#####\n")
                f.write(f"{str(fields)}\n")
                f.write("#####\n")
                f.write(f"{str(e)}\n")

        logger.debug(f"Generated field answers: {field_answers}")
        return field_answers, reasoning, generation_stats, messages

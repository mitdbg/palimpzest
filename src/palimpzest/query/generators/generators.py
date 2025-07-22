"""
This file contains the Generator classes and generator factory.
"""

from __future__ import annotations

import logging
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import Any, Generic, TypeVar

from colorama import Fore, Style
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from together import Together
from together.types.chat_completions import ChatCompletionResponse

from palimpzest.constants import (
    MODEL_CARDS,
    APIClient,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import Field, ListField
from palimpzest.prompts import PromptFactory
from palimpzest.query.generators.api_client_factory import APIClientFactory
from palimpzest.utils.generation_helpers import get_json_from_answer
from palimpzest.utils.sandbox import API

# DEFINITIONS
GenerationOutput = tuple[dict, str | None, GenerationStats, list[dict]]
ContextType = TypeVar("ContextType")
InputType = TypeVar("InputType")


logger = logging.getLogger(__name__)

def generator_factory(
    model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality, verbose: bool = False
) -> BaseGenerator:
    """
    Factory function to return the correct generator based on the model, strategy, and cardinality.
    """
    if model.is_openai_model():
        return OpenAIGenerator(model, prompt_strategy, cardinality, verbose)

    elif model.is_together_model():
        return TogetherGenerator(model, prompt_strategy, cardinality, verbose)

    raise Exception(f"Unsupported model: {model}")


def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        raise ValueError("key not found in environment variables")

    return os.environ[key]


# TODO: make sure answer parsing works with custom prompts / parsers (can defer this)
class BaseGenerator(Generic[ContextType, InputType], ABC):
    """
    Abstract base class for Generators.
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        verbose: bool = False,
        system_role: str = "system",
    ):
        self.model = model
        self.model_name = model.value
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.verbose = verbose
        self.system_role = system_role
        self.prompt_factory = PromptFactory(prompt_strategy, model, cardinality)

    @abstractmethod
    def _get_client_or_model(self, **kwargs) -> Any:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        pass

    @abstractmethod
    def _generate_completion(self, client_or_model: Any, payload: dict, **kwargs) -> Any:
        """Generates a completion object using the client (or local model)."""
        pass

    @abstractmethod
    def _get_completion_text(self, completion: Any, **kwargs) -> Any:
        """Extract the completion text from the completion object."""
        pass

    @abstractmethod
    def _get_usage(self, completion: Any, **kwargs) -> Any:
        """Extract the usage statistics from the completion object."""
        pass

    @abstractmethod
    def _get_finish_reason(self, completion: Any, **kwargs) -> Any:
        """Extract the finish reason from the completion object."""
        pass

    @abstractmethod
    def _get_answer_log_probs(self, completion: Any, **kwargs) -> Any:
        """Extract the log probabilities from the completion object."""
        pass

    def _generate_payload(self, messages: list[dict], **kwargs) -> dict:
        """
        Generates the payload which will be fed into the client (or local model).

        Each message will be a dictionary with the following format:
        {
            "role": "user" | "system",
            "type": "text" | "image",
            "content": str
        }
        """
        # get basic parameters
        model = self.model_name
        temperature = kwargs.get("temperature", 0.0)

        # construct messages and add system prompt if present
        chat_messages, user_content = [], []
        for message in messages:
            # flush user content into a message and add system message
            if message["role"] == "system":
                if len(user_content) > 0:
                    chat_messages.append({"role": "user", "content": user_content})
                    user_content = []

                chat_messages.append({"role": self.system_role, "content": message["content"]})

            # add user content for text messages
            elif message["role"] == "user" and message["type"] == "text":
                user_content.append({"type": "text", "text": message["content"]})

            # add user content for image messages
            elif message["role"] == "user" and message["type"] == "image":
                user_content.append({"type": "image_url", "image_url": {"url": message["content"]}})

        # flush any remaining user content into a final message
        if len(user_content) > 0:
            chat_messages.append({"role": "user", "content": user_content})

        # construct and return payload
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": chat_messages,
        }

        return payload

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

    def _prepare_field_answers(self, field_answers: dict | list[dict], fields: dict[str, Field]) -> dict[str, list]:
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

    def _check_convert_answer_text(self, answer_text: str, fields: dict[str, Field], throw_exception: bool=False) -> dict | list[dict] | None:
        """
        Try parsing the answer text into a JSON object. If the parsing fails, return None.
        """
        try:
            # extract json from the answer text
            field_answers = get_json_from_answer(answer_text, self.model, self.cardinality)

            # TODO: wrap non-list outputs in a list if expected output is a list

            # common error for one-to-one: if the output is a singleton list which contains a list, but the expected field type
            # is a list of strings, or a list of floats, i.e. not a list of lists; then extract the inner list
            if self.cardinality == Cardinality.ONE_TO_ONE:
                for field, field_type in fields.items():
                    answer = field_answers[field]
                    field_type_is_not_list_of_lists = isinstance(field_type, ListField) and not issubclass(field_type.element_type, ListField)
                    answer_is_list_of_lists = isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], list)
                    if field_type_is_not_list_of_lists and answer_is_list_of_lists:
                        field_answers[field] = answer[0]

            # prepare the field answers to match the expected output and return
            return self._prepare_field_answers(field_answers, fields)

        except Exception as e:
            if throw_exception:
                raise e

        return None

    def _check_filter_answer_text(self, answer_text: str) -> dict | None:
        """
        Return {"passed_operator": True} if and only if "true" is in the answer text.
        Return {"passed_operator": False} if and only if "false" is in the answer text.
        Otherwise, return None.
        """
        # NOTE: we may be able to eliminate this condition by specifying this JSON output in the prompt;
        # however, that would also need to coincide with a change to allow the parse_answer_fn to set "passed_operator"
        if "true" in answer_text.lower():
            return {"passed_operator": True}
        elif "false" in answer_text.lower():
            return {"passed_operator": False}

        return None

    def _parse_convert_answer(self, completion_text: str, fields: dict[str, Field], json_output: bool) -> dict[str, list]:
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

    def _parse_filter_answer(self, completion_text: str) -> dict[str, list]:
        """Extract the answer from the completion object for filter operations."""
        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # if the first regex didn't find an answer, try taking all the text after "ANSWER:"
        regex = re.compile("answer:(.*)", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # finally, try taking all of the text; throw an exception if this doesn't work
        field_answers = self._check_filter_answer_text(completion_text)
        if field_answers is None:
            raise Exception(f"Could not parse answer from completion text: {completion_text}")

        return field_answers

    def _parse_answer(self, completion_text: str, fields: dict[str, Field] | None, json_output: bool, **kwargs) -> dict[str, list]:
        """Extract the answer from the completion object."""
        # use a custom answer parser if provided
        if kwargs.get("parse_answer"):
            parse_answer_fn = kwargs.get("parse_answer")
            return parse_answer_fn(completion_text)

        # fields should be a dict if a custom answer parser is not provided
        assert isinstance(fields, dict), "Fields must be provided if a custom answer parser is not provided."

        # extract the per-field answers from the completion text
        field_answers = (
            self._parse_filter_answer(completion_text)
            if self.prompt_strategy.is_bool_prompt()
            else self._parse_convert_answer(completion_text, fields, json_output)
        )

        return field_answers

    def __call__(self, candidate: DataRecord, fields: dict[str, Field] | None, json_output: bool=True, **kwargs) -> GenerationOutput:
        """Take the input record (`candidate`), generate the output `fields`, and return the generated output."""
        client = self._get_client_or_model()
        logger.debug(f"Generating for candidate {candidate} with fields {fields}")

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
        messages = self.prompt_factory.create_messages(candidate, fields, **kwargs)

        # create the chat payload
        chat_payload = self._generate_payload(messages, **kwargs)

        # generate the text completion
        start_time = time.time()
        completion = None
        try:
            completion = self._generate_completion(client, chat_payload, **kwargs)
            end_time = time.time()
            logger.debug(f"Generated completion in {end_time - start_time:.2f} seconds")
        # if there's an error generating the completion, we have to return an empty answer
        # and can only account for the time spent performing the failed generation
        except Exception:
            # logger.error(f"Error generating completion: {e}")
            field_answers = {field_name: None for field_name in fields}
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
            usage = self._get_usage(completion, **kwargs)
            # finish_reason = self._get_finish_reason(completion, **kwargs)
            # answer_log_probs = self._get_answer_log_probs(completion, **kwargs)

            # get cost per input/output token for the model and parse number of input and output tokens
            usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
            usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]

            generation_stats = GenerationStats(
                model_name=self.model_name,
                llm_call_duration_secs=end_time - start_time,
                fn_call_duration_secs=0.0,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                total_input_cost=input_tokens * usd_per_input_token,
                total_output_cost=output_tokens * usd_per_output_token,
                cost_per_record=input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
                total_llm_calls=1,
                # "system_prompt": system_prompt,
                # "prompt": prompt,
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": answer,
            )

        # pretty print prompt + full completion output for debugging
        completion_text = self._get_completion_text(completion, **kwargs)
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += message["content"] + "\n" if message["type"] == "text" else "<image>\n"
        logger.debug(f"PROMPT:\n{prompt}")
        logger.debug(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

        # parse reasoning
        reasoning = None
        try:
            reasoning = self._parse_reasoning(completion_text, **kwargs)
        except Exception:
            # logger.error(f"Error parsing reasoning and answers: {e}")
            logger.debug("TODO: undo this")
            pass

        # parse field answers
        field_answers = None if fields is None else {field_name: None for field_name in fields}
        try:
            field_answers = self._parse_answer(completion_text, fields, json_output, **kwargs)
        except Exception as e:
            # logger.error(f"Error parsing answers: {e}")
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


class OpenAIGenerator(BaseGenerator[str | list[str], str]):
    """
    Class for generating text using the OpenAI chat API.
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        verbose: bool = False,
    ):
        # assert that model is an OpenAI model
        assert model.is_openai_model()
        super().__init__(model, prompt_strategy, cardinality, verbose, "developer")

    def _get_client_or_model(self, **kwargs) -> OpenAI:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return APIClientFactory.get_client(APIClient.OPENAI, get_api_key("OPENAI_API_KEY"))

    def _generate_completion(self, client: OpenAI, payload: dict, **kwargs) -> ChatCompletion:
        """Generates a completion object using the client (or local model)."""
        return client.chat.completions.create(**payload)

    def _get_completion_text(self, completion: ChatCompletion, **kwargs) -> str:
        """Extract the completion text from the completion object."""
        return completion.choices[0].message.content

    def _get_usage(self, completion: ChatCompletion, **kwargs) -> dict:
        """Extract the usage statistics from the completion object."""
        return {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

    def _get_finish_reason(self, completion: ChatCompletion, **kwargs) -> str:
        """Extract the finish reason from the completion object."""
        return completion.choices[0].finish_reason

    def _get_answer_log_probs(self, completion: ChatCompletion, **kwargs) -> list[float]:
        """Extract the log probabilities from the completion object."""
        return completion.choices[0].logprobs


class TogetherGenerator(BaseGenerator[str | list[str], str]):
    """
    Class for generating text using the Together chat API.
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        verbose: bool = False,
    ):
        # assert that model is a model offered by Together
        assert model.is_together_model()
        super().__init__(model, prompt_strategy, cardinality, verbose, "system")

    def _generate_payload(self, messages: list[dict], **kwargs) -> dict:
        """
        Generates the payload which will be fed into the client (or local model).

        Each message will be a dictionary with the following format:
        {
            "role": "user" | "system",
            "type": "text" | "image",
            "content": str
        }

        For LLAMA3, the payload needs to be in a {"role": <role>, "content": <content>} format.
        """
        # for other models, use our standard payload generation
        if not self.model.is_llama_model():
            return super()._generate_payload(messages, **kwargs)

        # get basic parameters
        model = self.model_name
        temperature = kwargs.get("temperature", 0.0)

        # construct messages in simple {"role": <role>, "content": <content>} format
        chat_messages = []
        for message in messages:
            chat_messages.append({"role": message["role"], "content": message["content"]})

        # construct and return payload
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": chat_messages,
        }

        return payload

    def _get_client_or_model(self, **kwargs) -> Together:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return APIClientFactory.get_client(APIClient.TOGETHER, get_api_key("TOGETHER_API_KEY"))

    def _generate_completion(self, client: Together, payload: dict, **kwargs) -> ChatCompletionResponse:
        """Generates a completion object using the client (or local model)."""
        return client.chat.completions.create(**payload)

    def _get_completion_text(self, completion: ChatCompletionResponse, **kwargs) -> str:
        """Extract the completion text from the completion object."""
        return completion.choices[0].message.content

    def _get_usage(self, completion: ChatCompletionResponse, **kwargs) -> dict:
        """Extract the usage statistics from the completion object."""
        return {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

    def _get_finish_reason(self, completion: ChatCompletionResponse, **kwargs) -> str:
        """Extract the finish reason from the completion object."""
        return completion.choices[0].finish_reason.value

    def _get_answer_log_probs(self, completion: ChatCompletionResponse, **kwargs) -> list[float]:
        """Extract the log probabilities from the completion object."""
        return completion.choices[0].logprobs


### CODE SYNTHESIS EXECUTION ###
def code_execution(api: API, code: str, candidate_dict: dict[str, Any], verbose: bool = False):
    inputs = {field_name: candidate_dict[field_name] for field_name in api.inputs}
    response = api.api_execute(code, inputs)
    pred = response["response"] if response["status"] and response["response"] else None
    return pred


def code_ensemble_execution(
    api: API, code_ensemble: dict[str, str], candidate_dict: dict[str, Any], verbose: bool = True
) -> GenerationOutput:
    start_time = time.time()
    try:
        preds = list()
        for _, code in code_ensemble.items():
            pred = code_execution(api, code, candidate_dict)
            preds.append(pred)

        preds = [pred for pred in preds if pred is not None]

        if len(preds) == 1:
            majority_response = preds[0]
            exec_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)
            return majority_response, None, exec_stats

        if len(preds) > 0:
            majority_response = Counter(preds).most_common(1)[0][0]
            exec_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)
            return majority_response, None, exec_stats

    except Exception:
        pass

    return None, None, GenerationStats(fn_call_duration_secs=time.time() - start_time)

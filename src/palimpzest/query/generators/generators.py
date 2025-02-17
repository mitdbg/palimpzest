"""
This file contains the Generator classes and generator factory.
"""
from __future__ import annotations

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

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from together.types.chat_completions import ChatCompletionResponse

from palimpzest.constants import (
    MODEL_CARDS,
    # RETRY_MAX_ATTEMPTS,
    # RETRY_MAX_SECS,
    # RETRY_MULTIPLIER,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.prompts import PromptFactory
from palimpzest.utils.generation_helpers import get_json_from_answer
from palimpzest.utils.sandbox import API

# DEFINITIONS
GenerationOutput = tuple[dict, str | None, GenerationStats]
ContextType = TypeVar("ContextType")
InputType = TypeVar("InputType")


def generator_factory(model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality, verbose: bool = False) -> BaseGenerator:
    """
    Factory function to return the correct generator based on the model, strategy, and cardinality.
    """
    if model in [Model.GPT_4o, Model.GPT_4o_MINI, Model.GPT_4o_V, Model.GPT_4o_MINI_V]:
        return OpenAIGenerator(model, prompt_strategy, cardinality, verbose)

    elif model in [Model.MIXTRAL, Model.LLAMA3, Model.LLAMA3_V]:
        return TogetherGenerator(model, prompt_strategy, cardinality, verbose)

    raise Exception(f"Unsupported model: {model}")


def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]


# TODO: make sure answer parsing works with custom prompts / parsers (can defer this)
class BaseGenerator(Generic[ContextType, InputType], ABC):
    """
    Abstract base class for Generators.
    """
    def __init__(self, model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality = Cardinality.ONE_TO_ONE, verbose: bool = False, system_role: str = "system"):
        self.model = model
        self.model_name = model.value
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.verbose = verbose
        self.system_role = system_role
        self.prompt_factory = PromptFactory(prompt_strategy, model, cardinality)
        self.messages = None

    def get_messages(self) -> list[dict] | None:
        """Returns the messages used in the last generation."""
        return self.messages

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

    def _parse_reasoning(self, completion_text: str, **kwargs) -> Any:
        """Extract the reasoning for the generated output from the completion object."""
        # use a custom reasoning parser if provided
        if kwargs.get("parse_reasoning"):
            parse_reasoning_fn = kwargs.get("parse_reasoning")
            return parse_reasoning_fn(completion_text)

        # if the model followed the default instructions, the completion text will have reasoning
        # before the "ANSWER:"; if this is the case, we simply extract and return that full section
        regex = re.compile("(.*?)answer:.*", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            return matches[0].strip()

        # otherwise, return None
        return None

    def _parse_answer(self, completion_text: str, fields: list[str] | None, **kwargs) -> Any:
        """Extract the answer from the completion object."""
        # use a custom answer parser if provided
        if kwargs.get("parse_answer"):
            parse_answer_fn = kwargs.get("parse_answer")
            return parse_answer_fn(completion_text)

        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        answer_text = None
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()

        # otherwise, take all the text after "ANSWER:" (or just all of the text)
        else:
            regex = re.compile("answer:(.*?)", re.IGNORECASE | re.DOTALL)
            matches = regex.findall(completion_text)
            answer_text = matches[0].strip() if len(matches) > 0 else completion_text

        # if this is a filter operator, return True if and only if "true" is in the answer text
        # NOTE: we may be able to elimiate this condition by specifying this JSON output in the prompt;
        # however, that would also need to coincide with a change to allow the parse_answer_fn to set "passed_operator"
        if self.prompt_strategy in [PromptStrategy.COT_BOOL, PromptStrategy.COT_BOOL_IMAGE]:
            return {"passed_operator": "true" in answer_text.lower()}

        # parse the answer text into a JSON object and return it
        field_answers = get_json_from_answer(answer_text, self.model, self.cardinality)

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

    def __call__(self, candidate: DataRecord, fields: list[str] | None, **kwargs) -> GenerationOutput:
        """Take the input record (`candidate`), generate the output `fields`, and return the generated output."""
        client = self._get_client_or_model()

        # fields can only be None if the user provides an answer parser
        assert fields is not None or "parse_answer" in kwargs, "`fields` must be provided if `parse_answer` function is not provided in kwargs."

        # if the user (or operator) provides a system prompt instead of a prompt, treat this as
        # the prompt and print a warning
        if "system_prompt" in kwargs and "prompt" not in kwargs:
            kwargs["prompt"] = kwargs["system_prompt"]
            kwargs.pop("system_prompt")
            warnings.warn("Provided `system_prompt` without providing `prompt`; setting `prompt` = `system_prompt`.")  # noqa: B028

        # generate a list of messages which can be used to construct a payload
        self.messages = self.prompt_factory.create_messages(candidate, fields, **kwargs)

        # create the chat payload
        chat_payload = self._generate_payload(self.messages, **kwargs)

        # generate the text completion
        start_time = time.time()
        completion = None
        try:
            completion = self._generate_completion(client, chat_payload, **kwargs)
            end_time = time.time()

        # if there's an error generating the completion, we have to return an empty answer
        # and can only account for the time spent performing the failed generation
        except Exception as e:
            print(f"Error generating completion: {e}")
            field_answers = {field_name: None for field_name in fields}
            reasoning = None
            generation_stats = GenerationStats(model_name=self.model_name, llm_call_duration_secs=time.time() - start_time)

            return field_answers, reasoning, generation_stats

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
                # "system_prompt": system_prompt,
                # "prompt": prompt,
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": answer,
            )

        # pretty print prompt + full completion output for debugging
        completion_text = self._get_completion_text(completion, **kwargs)
        if self.verbose:
            prompt = ""
            for message in self.messages:
                if message["role"] == "user":
                    prompt += message["content"] + "\n" if message["type"] == "text" else "<image>\n"
            print(f"PROMPT:\n{prompt}")
            print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

        # parse reasoning
        reasoning = None
        try:
            reasoning = self._parse_reasoning(completion_text, **kwargs)
        except Exception as e:
            print(f"Error parsing reasoning and answers: {e}")

        # parse field answers
        field_answers = None if fields is None else {field_name: None for field_name in fields}
        try:
            field_answers = self._parse_answer(completion_text, fields, **kwargs)
        except Exception as e:
            print(f"Error parsing answers: {e}")

        return field_answers, reasoning, generation_stats


class OpenAIGenerator(BaseGenerator[str | list[str], str]):
    """
    Class for generating text using the OpenAI chat API.
    """
    def __init__(self, model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality = Cardinality.ONE_TO_ONE, verbose: bool = False):
        # assert that model is an OpenAI model
        assert model in [Model.GPT_4o, Model.GPT_4o_MINI, Model.GPT_4o_V, Model.GPT_4o_MINI_V]
        super().__init__(model, prompt_strategy, cardinality, verbose, "developer")

    def _get_client_or_model(self, **kwargs) -> OpenAI:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

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
    def __init__(self, model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality = Cardinality.ONE_TO_ONE, verbose: bool = False):
        # assert that model is a model offered by Together
        assert model in [Model.MIXTRAL, Model.LLAMA3, Model.LLAMA3_V]
        super().__init__(model, prompt_strategy, cardinality, verbose, "system")

    def _get_client_or_model(self, **kwargs) -> Together:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return Together(api_key=get_api_key("TOGETHER_API_KEY"))

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
        print(preds)

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

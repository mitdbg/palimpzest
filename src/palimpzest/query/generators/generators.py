"""
This file contains the Generator classes and generator factory.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from string import Formatter
from typing import Any, Generic, TypeVar

from colorama import Fore, Style
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from together.types.chat_completions import ChatCompletionResponse

import palimpzest.prompts as prompts
from palimpzest.constants import (
    MODEL_CARDS,
    TOKENS_PER_CHARACTER,
    # RETRY_MAX_ATTEMPTS,
    # RETRY_MAX_SECS,
    # RETRY_MULTIPLIER,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BytesField, ImageBase64Field, ImageFilepathField, ImageURLField, ListField
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
    def __init__(self, model: Model, prompt_strategy: PromptStrategy, cardinality: Cardinality = Cardinality.ONE_TO_ONE, verbose: bool = False):
        self.model = model
        self.model_name = model.value
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.verbose = verbose

    @abstractmethod
    def _get_client_or_model(self, **kwargs) -> Any:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        pass

    @abstractmethod
    def _generate_payload(self, context: ContextType, prompt: InputType, **kwargs) -> Any:
        """Generates the payload which will be fed into the client (or local model)."""
        pass

    @abstractmethod
    def _generate_completion(self, client_or_model: Any, **kwargs) -> Any:
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

    def _generate_developer_prompt(self) -> str:
        """Returns a prompt based on the prompt strategy with high-level instructions for the generation."""
        if self.prompt_strategy == PromptStrategy.COT_BOOL:
            prompt = prompts.COT_BOOL_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_BOOL_IMAGE:
            prompt = prompts.COT_BOOL_IMAGE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_QA:
            prompt = prompts.COT_QA_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_QA_IMAGE:
            prompt = prompts.COT_QA_IMAGE_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER:
            prompt = prompts.COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER_IMAGE:
            prompt = prompts.COT_MOA_PROPOSER_IMAGE_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_MOA_AGG:
            prompt = prompts.COT_MOA_AGG_BASE_SYSTEM_PROMPT

        if self.prompt_strategy not in [PromptStrategy.COT_BOOL, PromptStrategy.COT_BOOL_IMAGE]:
            output_format_instruction = (
                prompts.ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION
                if self.cardinality == Cardinality.ONE_TO_ONE
                else prompts.ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION
            )
            prompt = prompt.format(output_format_instruction=output_format_instruction)

        return prompt

    def _generate_user_prompt(self, candidate: DataRecord, fields: list[str], **kwargs) -> str:
        """Returns a prompt based on the prompt strategy with instance-specific instructions."""
        # get context from input record (project_cols will be None if not provided in kwargs)
        context = candidate.to_json_str(include_bytes=False, project_cols=kwargs.get("project_cols"))

        # get filter condition for filter operations
        filter_condition = (
            kwargs.get("filter_condition")
            if self.prompt_strategy in [PromptStrategy.COT_BOOL, PromptStrategy.COT_BOOL_IMAGE]
            else None
        )

        # get model responses for mixture-of-agents aggregation
        model_responses = None
        if self.prompt_strategy in [PromptStrategy.COT_MOA_AGG]:
            model_responses = ""
            for idx, model_response in enumerate(kwargs.get("model_responses")):
                model_responses += f"MODEL RESPONSE {idx + 1}: {model_response}\n"

        # generate input and output fields descriptions
        input_fields_desc = ""
        for field_name in kwargs.get("project_cols", candidate.get_field_names()):
            input_fields_desc += f"- {field_name}: {candidate.get_field_type(field_name)._desc}\n"

        output_fields_desc = ""
        if 'output_schema' in kwargs:
            field_desc_map = kwargs.get('output_schema').field_desc_map()
            for field_name in fields:
                output_fields_desc += f"- {field_name}: {field_desc_map[field_name]}\n"

        # strip the last newline characters from the field descriptions
        input_fields_desc = input_fields_desc[:-1]
        output_fields_desc = output_fields_desc[:-1]

        # set formatting instruction for non-filter prompts
        output_format_instruction = (
            prompts.ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION
            if self.cardinality == Cardinality.ONE_TO_ONE
            else prompts.ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION
        )

        # cut down on context based on window length
        if self.model in [Model.LLAMA3, Model.MIXTRAL]:
            total_context_len = len(json.dumps(context, indent=2))

            # sort fields by length and progressively strip from the longest field until it is short enough;
            # NOTE: 6000 is a rough estimate which leaves room for the rest of the prompt text
            while total_context_len * TOKENS_PER_CHARACTER > 6000:
                # sort fields by length
                field_lengths = [(field, len(value)) for field, value in context.items()]
                sorted_fields = sorted(field_lengths, key=lambda item: item[1], reverse=True)

                # get field with longest context
                longest_field_name, longest_field_length = sorted_fields[0]

                # trim the field
                context_factor =  6000.0 / (total_context_len * TOKENS_PER_CHARACTER)
                keep_frac_idx = int(len(longest_field_length) * context_factor)
                context[longest_field_name] = context[longest_field_name][:keep_frac_idx]

                # update total context length
                total_context_len = len(json.dumps(context, indent=2))

        # initialize format_kwargs
        format_kwargs = {
            "context": context,
            "input_fields_desc": input_fields_desc,
        }

        # select prompt based on prompt strategy and update format_kwargs as needed
        if self.prompt_strategy == PromptStrategy.COT_BOOL:
            prompt = prompts.COT_BOOL_USER_PROMPT
            format_kwargs.update({"filter_condition": filter_condition})

        elif self.prompt_strategy == PromptStrategy.COT_BOOL_IMAGE:
            prompt = prompts.COT_BOOL_IMAGE_USER_PROMPT
            format_kwargs.update({"filter_condition": filter_condition})

        elif self.prompt_strategy == PromptStrategy.COT_QA:
            prompt = prompts.COT_QA_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        elif self.prompt_strategy == PromptStrategy.COT_QA_IMAGE:
            prompt = prompts.COT_QA_IMAGE_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER:
            prompt = prompts.COT_MOA_PROPOSER_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER_IMAGE:
            prompt = prompts.COT_MOA_PROPOSER_IMAGE_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        elif self.prompt_strategy == PromptStrategy.COT_MOA_AGG:
            prompt = prompts.COT_MOA_AGG_BASE_USER_PROMPT
            format_kwargs.pop("context")
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
                "model_responses": model_responses,
            })

        return prompt.format(**format_kwargs)

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
            field_answers_lst: list[dict] = field_answers.copy()

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

        # if the user (or operator) provides a developer prompt instead of a prompt, treat this as
        # the prompt and print a warning
        if "developer_prompt" in kwargs and "prompt" not in kwargs:
            kwargs["prompt"] = kwargs["developer_prompt"]
            kwargs.pop("developer_prompt")
            warnings.warn("Provided `developer_prompt` without providing `prompt`; setting `prompt` = `developer_prompt`.")  # noqa: B028

        # if the user provides a prompt, use it; otherwise, generate a prompt based on the prompt strategy
        prompt = None
        if "prompt" in kwargs:
            prompt: str = kwargs["prompt"]
            prompt_field_names = [fname for _, fname, _, _ in Formatter().parse(prompt) if fname]
            assert all([field in candidate.get_field_names() for field in prompt_field_names]), f"Prompt string has fields which are not in candidate record.\nPrompt fields: {prompt_field_names}\nRecord fields: {candidate.get_field_names()}"
            prompt = prompt.format({
                field_name: "<bytes>" if issubclass(candidate.get_field_type(field_name), BytesField) else candidate[field_name]
                for field_name in prompt_field_names
            })

        else:
            prompt = self._generate_user_prompt(candidate, fields, **kwargs)

        # if the user (or operator) provides a user prompt and no developer prompt, then we just
        # use the user prompt because our default developer prompt may have conflicting instructions;
        # otherwise, we take the provided developer prompt or generate a default developer prompt
        developer_prompt = (
            None
            if "prompt" in kwargs and "developer_prompt" not in kwargs
            else kwargs.get("developer_prompt", self._generate_developer_prompt())
        )

        # create the chat payload
        chat_payload = self._generate_payload(candidate, prompt, developer_prompt, **kwargs)

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
                # "developer_prompt": developer_prompt,
                # "prompt": prompt,
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": answer,
            )

        # pretty print prompt + full completion output for debugging
        completion_text = self._get_completion_text(completion, **kwargs)
        if self.verbose:
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
        assert prompt_strategy in [
            PromptStrategy.COT_BOOL,
            PromptStrategy.COT_BOOL_IMAGE,
            PromptStrategy.COT_QA,
            PromptStrategy.COT_QA_IMAGE,
            PromptStrategy.COT_MOA_PROPOSER,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE,
            PromptStrategy.COT_MOA_AGG,
        ]
        super().__init__(model, prompt_strategy, cardinality, verbose)

    def _get_client_or_model(self, **kwargs) -> OpenAI:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    def _generate_payload(self, candidate: DataRecord, user_prompt: str, developer_prompt: str | None, **kwargs) -> dict:
        """Generates the payload which will be fed into the client (or local model)."""
        # get basic parameters
        model = self.model_name
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 4096)

        # construct messages and add developer prompt if present
        messages = []
        if developer_prompt is not None:
            messages.append({"role": "developer", "content": developer_prompt})

        # construct user content
        user_content = [{"type": "text", "text": user_prompt}]

        # determine if any field is an image filepath, image URL, or base64 encoded image bytes
        is_image_conversion = False
        for field_name, field_value in candidate:
            field_type = candidate.field_types[field_name]

            # image filepath (or list of image filepaths)
            if isinstance(field_type, ImageFilepathField):
                is_image_conversion = True
                with open(field_value, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            elif isinstance(field_type, ListField) and isinstance(field_type.element_type, ImageFilepathField):
                is_image_conversion = True
                for image_filepath in field_value:
                    with open(image_filepath, 'rb') as f:
                        base64_image = base64.b64encode(f.read()).decode('utf-8')
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            # image url (or list of image urls)
            elif isinstance(field_type, ImageURLField):
                is_image_conversion = True
                user_content.append({"type": "image_url", "image_url": {"url": field_value}})

            elif isinstance(field_type, ListField) and isinstance(field_type.element_type, ImageURLField):
                is_image_conversion = True
                for image_url in field_value:
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})

            # pre-encoded images (or list of pre-encoded images)
            elif isinstance(field_type, ImageBase64Field):
                is_image_conversion = True
                base64_image_str = field_value.decode("utf-8")
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}})

            elif isinstance(field_type, ListField) and isinstance(field_type.element_type, ImageBase64Field):
                is_image_conversion = True
                for base64_image in field_value:
                    base64_image_str = base64_image.decode("utf-8")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}})

        # if this is an image conversion, we need to add the reasoning prompt suffix after the image
        if is_image_conversion:
            suffix = (
                prompts.IMAGE_ANSWER_SUFFIX
                if self.prompt_strategy  == PromptStrategy.COT_MOA_PROPOSER_IMAGE
                else prompts.IMAGE_REASONING_SUFFIX
            )
            user_content.append({"type": "text", "text": suffix})

        # add user message(s)
        messages.append({"role": "user", "content": user_content})

        # construct and return payload
        payload = {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }

        return payload

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
        assert prompt_strategy in [
            PromptStrategy.COT_BOOL,
            PromptStrategy.COT_BOOL_IMAGE,
            PromptStrategy.COT_QA,
            PromptStrategy.COT_QA_IMAGE,
            PromptStrategy.COT_MOA_PROPOSER,
            PromptStrategy.COT_MOA_PROPOSER_IMAGE,
            PromptStrategy.COT_MOA_AGG,
        ]
        super().__init__(model, prompt_strategy, cardinality, verbose)

    def _get_client_or_model(self, **kwargs) -> Together:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return Together(api_key=get_api_key("TOGETHER_API_KEY"))

    def _generate_payload(self, candidate: DataRecord, user_prompt: str, system_prompt: str | None, **kwargs) -> dict:
        """Generates the payload which will be fed into the client (or local model)."""
        # get basic parameters
        model = self.model_name
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 4096)

        # construct messages and add system prompt if present
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        # construct user content
        user_content = [{"type": "text", "text": user_prompt}]

        # for Together model(s), there can only be a single image field
        assert sum([field.is_image_field for _, field in candidate.field_types.items()]) <= 1, "Together models can only have a single image field."

        # determine if any field is an image filepath, image URL, or base64 encoded image bytes
        # NOTE: the Rules for the various convert operators will not consider Together models when converting
        #       fields that are lists of images; thus, we only need to worry about processing image fields directly
        is_image_conversion = False
        for field_name, field_value in candidate:
            field_type = candidate.field_types[field_name]

            # image filepath
            if isinstance(field_type, ImageFilepathField):
                is_image_conversion = True
                with open(field_value, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            # image url
            elif isinstance(field_type, ImageURLField):
                is_image_conversion = True
                user_content.append({"type": "image_url", "image_url": {"url": field_value}})

            # pre-encoded images
            elif isinstance(field_type, ImageBase64Field):
                is_image_conversion = True
                base64_image_str = field_value.decode("utf-8")
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}})

        # if this is an image conversion, we need to add the reasoning prompt suffix after the image
        if is_image_conversion:
            suffix = (
                prompts.IMAGE_ANSWER_SUFFIX
                if self.prompt_strategy  == PromptStrategy.COT_MOA_PROPOSER_IMAGE
                else prompts.IMAGE_REASONING_SUFFIX
            )
            user_content.append({"type": "text", "text": suffix})

        # add user message(s)
        messages.append({"role": "user", "content": user_content})

        # construct and return payload
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        return payload

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

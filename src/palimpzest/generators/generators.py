"""GV: This class is about LLM wrappers.
My suggestion is to rename at least the base generator into LLMGenerator.
See llm_wrapper.py for a proposed refactoring of generators.py using the class factory pattern.
"""
import base64
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from string import Formatter
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union

import dsp
import dspy
import google.generativeai as genai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together

import palimpzest.prompts as prompts
from palimpzest.constants import (
    MODEL_CARDS,
    # RETRY_MAX_ATTEMPTS,
    # RETRY_MAX_SECS,
    # RETRY_MULTIPLIER,
    Cardinality,
    Model,
    PromptStrategy,
    # log_attempt_number,
)
from palimpzest.dataclasses import GenerationStats
from palimpzest.elements.records import DataRecord
from palimpzest.generators.dspy_utils import (
    DSPyCOT,
    TogetherHFAdaptor,
    gen_filter_signature_class,
    gen_moa_agg_qa_signature_class,
    gen_qa_signature_class,
)
from palimpzest.utils.sandbox import API

# DEFINITIONS
GenerationOutput = Tuple[str, str | None, GenerationStats]
ContextType = TypeVar("ContextType")
InputType = TypeVar("InputType")


def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]


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
    def _get_reasoning(self, completion: Any, **kwargs) -> Any:
        """Extract the reasoning for the generated output from the completion object."""
        pass

    @abstractmethod
    def _get_answer(self, completion: Any, **kwargs) -> Any:
        """Extract the answer from the completion object."""
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

    @abstractmethod
    def generate(self, context: ContextType, prompt: InputType, **kwargs) -> GenerationOutput:
        """asdf"""
        pass


class OpenAIGenerator(BaseGenerator[str | List[str], str]):
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
            PromptStrategy.COT_MOA_PROPOSER,
            PromptStrategy.COT_MOA_AGG,
            PromptStrategy.COT_QA_IMAGE,
        ]
        super().__init__(model, prompt_strategy, cardinality, verbose)

    def _get_client_or_model(self, **kwargs) -> Any:
        """Returns a client (or local model) which can be invoked to perform the generation."""
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    def _generate_developer_prompt(self) -> str:
        """Returns a prompt based on the prompt strategy with high-level instructions for the generation."""
        if self.prompt_strategy == PromptStrategy.COT_BOOL:
            prompt = prompts.COT_BOOL_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_BOOL_IMAGE:
            prompt = prompts.COT_BOOL_IMAGE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_QA:
            prompt = prompts.COT_QA_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER:
            prompt = prompts.COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_MOA_AGG:
            prompt = prompts.COT_MOA_AGG_BASE_SYSTEM_PROMPT
        elif self.prompt_strategy == PromptStrategy.COT_QA_IMAGE:
            prompt = prompts.COT_QA_IMAGE_BASE_SYSTEM_PROMPT

        if self.prompt_strategy not in [PromptStrategy.COT_BOOL, PromptStrategy.COT_BOOL_IMAGE]:
            output_format_instruction = (
                prompts.ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION
                if self.cardinality == Cardinality.ONE_TO_ONE
                else prompts.ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION
            )
            prompt = prompt.format(output_format_instruction=output_format_instruction)

        return prompt

    def _generate_user_prompt(self, candidate: DataRecord, fields: List[str], **kwargs) -> str:
        """Returns a prompt based on the prompt strategy with instance-specific instructions."""
        # get context from input record (project_cols will be None if not provided in kwargs)
        context = candidate.as_json_str(include_bytes=False, project_cols=kwargs.get("project_cols"), include_data_cols=False)

        # get filter condition for filter operations
        filter_condition = (
            kwargs.get("filter_condition")
            if self.prompt_strategy in [PromptStrategy.COT_BOOL, PromptStrategy.COT_BOOL_IMAGE]
            else None
        )

        # get model responses for mixture-of-agents aggregation
        model_responses = None
        if self.prompt_strategy == PromptStrategy.COT_MOA_AGG:
            model_responses = ""
            for idx, model_response in enumerate(kwargs.get("model_responses")):
                model_responses += f"MODEL RESPONSE {idx + 1}: {model_response}\n"

        # generate input and output fields descriptions
        input_fields_desc = ""
        for field in kwargs.get("project_cols", candidate.get_fields()):
            input_fields_desc += f"- {field}: {candidate.get_field_desc(field)}\n" # TODO: add field descriptions to kwargs?

        output_fields_desc = ""
        for field in fields:
            output_fields_desc += f"- {field}: {candidate.get_field_desc(field)}\n" # TODO: add field descriptions to kwargs?

        # strip the last newline characters from the field descriptions
        input_fields_desc = input_fields_desc[:-1]
        output_fields_desc = output_fields_desc[:-1]

        # set formatting instruction for non-filter prompts
        output_format_instruction = (
            prompts.ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION
            if self.cardinality == Cardinality.ONE_TO_ONE
            else prompts.ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION
        )

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

        elif self.prompt_strategy == PromptStrategy.COT_MOA_PROPOSER:
            prompt = prompts.COT_MOA_PROPOSER_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        elif self.prompt_strategy == PromptStrategy.COT_MOA_AGG:
            prompt = prompts.COT_MOA_AGG_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
                "model_responses": model_responses,
            })

        elif self.prompt_strategy == PromptStrategy.COT_QA_IMAGE:
            prompt = prompts.COT_QA_IMAGE_BASE_USER_PROMPT
            format_kwargs.update({
                "output_format_instruction": output_format_instruction,
                "output_fields_desc": output_fields_desc,
            })

        return prompt.format(**format_kwargs)

    def _generate_payload(self, candidate: DataRecord, user_prompt: str, developer_prompt: str | None, **kwargs) -> Any:
        """Generates the payload which will be fed into the client (or local model)."""
        # get basic parameters
        model = self.model_name
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 4096)

        # construct messages
        messages = []
        if developer_prompt is not None:
            messages.append({"role": "developer", "content": developer_prompt})
        
        image_conversion = kwargs.get("image_conversion", False) # TODO
        if image_conversion:
            messages.append({
                "role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": kwargs.get("image_url")}, # TODO: handle image url vs. image bytes
                ]
            })
            # TODO: iterate over fields in candidate and add image content to messages; you may also need to iterate over a list[bytes] within the candidate
        else:
            messages.append({"role": "user", "content": user_prompt})

        # construct and return payload
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        return payload

    def _generate_completion(self, client: OpenAI, payload: dict, **kwargs) -> ChatCompletion:
        """Generates a completion object using the client (or local model)."""
        return client.chat.completions.create(**payload)

    def _get_reasoning(self, completion: ChatCompletion, **kwargs) -> Any:
        """Extract the reasoning for the generated output from the completion object."""
        pass

    def _get_answer(self, completion: ChatCompletion, **kwargs) -> Any:
        """Extract the answer from the completion object."""
        pass

    def _get_usage(self, completion: ChatCompletion, **kwargs) -> Any:
        """Extract the usage statistics from the completion object."""
        return 

    def _get_finish_reason(self, completion: ChatCompletion, **kwargs) -> Any:
        """Extract the finish reason from the completion object."""
        return completion.choices[0].finish_reason

    def _get_log_probs(self, completion: ChatCompletion, **kwargs) -> Any:
        """Extract the log probabilities from the completion object."""
        return completion.choices[0].logprobs

    def generate(self, candidate: DataRecord, fields: List[str], **kwargs) -> GenerationOutput:
        """Process the given context and prompt using the specified model and return the generated output."""
        client = self._get_client_or_model()

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
            assert all([field in candidate.get_fields() for field in prompt_field_names]), f"Prompt string has fields which are not in candidate record.\nPrompt fields: {prompt_field_names}\nRecord fields: {candidate.get_fields()}"
            prompt = prompt.format({field: getattr(candidate, field) for field in prompt_field_names})
        
        else:
            prompt = self._generate_user_prompt(candidate, fields, **kwargs) # TODO: add field descriptions to kwargs? (and filter_condition)

        # if the user (or operator) provides a user prompt and no developer prompt, then we just
        # use the user prompt because our default developer prompt may have conflicting instructions;
        # otherwise, we take the provided developer prompt or generate a default developer prompt
        developer_prompt = (
            None
            if "prompt" in kwargs and "developer_prompt" not in kwargs
            else kwargs.get("developer_prompt", self._generate_developer_prompt())
        )

        # generate payload and completion
        chat_payload = self._generate_payload(candidate, prompt, developer_prompt, **kwargs)

        # TODO: catch exception(s) with generation?
        start_time = time.time()
        completion = self._generate_completion(client, chat_payload, **kwargs)
        end_time = time.time()

        # parse answer, reasoning, and other features of generation
        answer = self._get_answer(completion, **kwargs)
        reasoning = self._get_reasoning(completion, **kwargs)
        usage = self._get_usage(completion, **kwargs)
        finish_reason = self._get_finish_reason(completion, **kwargs)
        answer_log_probs = self._get_log_probs(completion, **kwargs)

        # get cost per input/output token for the model and parse number of input and output tokens
        usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        # create GenerationStats
        stats = GenerationStats(
            **{
                "model_name": self.model_name,
                "llm_call_duration_secs": end_time - start_time,
                "fn_call_duration_secs": 0.0,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_input_cost": input_tokens * usd_per_input_token,
                "total_output_cost": output_tokens * usd_per_output_token,
                "cost_per_record": input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
                # "developer_prompt": developer_prompt,
                # "prompt": prompt,
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": answer,
            }
        )

        # TODO: pretty print prompt + full completion output for debugging
        if self.verbose:
            dspy_lm.inspect_history(n=1)

        return answer, reasoning, stats


class CustomGenerator(BaseGenerator[str | None, str]):
    """
    Class for generating outputs with a given model using a custom prompt.
    """

    def __init__(self, model: Model, verbose: bool = False):
        super().__init__()
        self.model = model
        self.model_name = model.value
        self.verbose = verbose

    def _get_model(self, temperature: float = 0.0) -> dspy.OpenAI | dspy.Google | TogetherHFAdaptor:
        model = None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            openai_key = get_api_key("OPENAI_API_KEY")
            max_tokens = 4096
            model = dspy.OpenAI(
                model=self.model_name,
                api_key=openai_key,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
            )

        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            together_key = get_api_key("TOGETHER_API_KEY")
            model = TogetherHFAdaptor(self.model_name, together_key, temperature=temperature, logprobs=1)

        # elif self.model_name in [Model.GEMINI_1.value]:
        #     google_key = get_api_key("GOOGLE_API_KEY")
        #     model = dspy.Google(model=self.model_name, api_key=google_key)

        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["choices"][-1]["finish_reason"]
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     usage = {"prompt_tokens": 0, "completion_tokens": 0}
        #     finish_reason = (
        #         dspy_lm.history[-1]["response"][0]._result.candidates[0].finish_reason
        #     )
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["finish_reason"]

        return usage, finish_reason

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float] | None:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        token_logprobs = None

        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]["response"]["choices"][-1]["logprobs"]["content"]
            token_logprobs = list(map(lambda elt: elt["logprob"], log_probs))
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     return None
        #     # TODO Google gemini does not provide log probabilities!
        #     # https://github.com/google/generative-ai-python/issues/238
        #     # tok_count = dspy_lm.llm.count_tokens(answer).total_tokens
        #     # tokens = [""] * tok_count
        #     # token_logprobs = [0] * len(tokens)
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
            token_logprobs = dspy_lm.history[-1]["response"]["token_logprobs"]
        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        # get indices of the start and end token for the answer
        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
        # if answer.startswith(tokens[start_idx]):
        # end_idx += 1
        # else:
        # start_idx += 1
        # end_idx = start_idx
        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs
        # return those tokens log probabilities
        return answer_log_probs

    # TODO: undo after paper submission
    # @retry(
    #     wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
    #     stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    #     after=log_attempt_number,
    #     reraise=True,
    # )
    def generate(self, context: str | None, prompt: str, **kwargs) -> GenerationOutput:
        # fetch model
        dspy_lm = self._get_model(temperature=kwargs.get("temperature", 0.0))

        start_time = time.time()

        response = dspy_lm.request(prompt)
        if not response:
            raise ValueError("Model did not return a response.")

        end_time = time.time()

        answer = response["choices"][0]["message"]["content"]
        usage = response["usage"]

        # collect statistics on prompt, usage, and timing
        usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        # create GenerationStats
        stats = GenerationStats(
            **{
                "model_name": self.model_name,
                "llm_call_duration_secs": end_time - start_time,
                "fn_call_duration_secs": 0.0,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_input_cost": input_tokens * usd_per_input_token,
                "total_output_cost": output_tokens * usd_per_output_token,
                "cost_per_record": input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
                # "prompt": dspy_lm.history[-1]["prompt"],
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": answer,
            }
        )

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)

        return answer, None, stats


class DSPyGenerator(BaseGenerator[str, str]):
    """
    Class for generating outputs with a given model using DSPy for prompting optimization(s).
    """

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy,
        doc_schema: str,
        doc_type: str,
        verbose: bool = False,
    ):
        super().__init__()
        self.model = model
        self.model_name = model.value
        self.prompt_strategy = prompt_strategy
        self.verbose = verbose

        # set prompt signature based on prompt_strategy
        if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            self.promptSignature = gen_filter_signature_class(doc_schema, doc_type)
        elif prompt_strategy == PromptStrategy.DSPY_COT_QA:
            self.promptSignature = gen_qa_signature_class(doc_schema, doc_type)
        elif prompt_strategy == PromptStrategy.DSPY_COT_MOA_AGG:
            self.promptSignature = gen_moa_agg_qa_signature_class(doc_type)
        else:
            raise ValueError(f"DSPyGenerator does not support prompt_strategy: {prompt_strategy.value}")

    def _get_model(self, temperature: float=0.0) -> dsp.LM:
        model = None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            openai_key = get_api_key("OPENAI_API_KEY")
            max_tokens = 4096 if self.prompt_strategy in [PromptStrategy.DSPY_COT_QA, PromptStrategy.DSPY_COT_MOA_AGG] else 250
            model = dspy.OpenAI(
                model=self.model_name,
                api_key=openai_key,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
            )

        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            together_key = get_api_key("TOGETHER_API_KEY")
            model = TogetherHFAdaptor(self.model_name, together_key, temperature=temperature, logprobs=1)

        # elif self.model_name in [Model.GEMINI_1.value]:
        #     google_key = get_api_key(f"GOOGLE_API_KEY")
        #     model = dspy.Google(model=self.model_name, api_key=google_key)

        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["choices"][-1]["finish_reason"]
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     usage = {"prompt_tokens": 0, "completion_tokens": 0}
        #     finish_reason = (
        #         dspy_lm.history[-1]["response"][0]._result.candidates[0].finish_reason
        #     )
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["finish_reason"]

        # raise if usage or finish_reason is None
        if usage is None or finish_reason is None:
            raise ValueError(f"Usage or finish_reason is None: {usage}, {finish_reason}")

        return usage, finish_reason

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float]:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        token_logprobs = None

        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]["response"]["choices"][-1]["logprobs"]["content"]
            token_logprobs = list(map(lambda elt: elt["logprob"], log_probs))
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     return None
        #     # TODO Google gemini does not provide log probabilities!
        #     # https://github.com/google/generative-ai-python/issues/238
        #     # tok_count = dspy_lm.llm.count_tokens(answer).total_tokens
        #     # tokens = [""] * tok_count
        #     # token_logprobs = [0] * len(tokens)
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
            token_logprobs = dspy_lm.history[-1]["response"]["token_logprobs"]
        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        # get indices of the start and end token for the answer
        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
        # if answer.startswith(tokens[start_idx]):
        # end_idx += 1
        # else:
        # start_idx += 1
        # end_idx = start_idx
        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs
        # return those tokens log probabilities
        return answer_log_probs

    # TODO: undo after paper submission
    # @retry(
    #     wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
    #     stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    #     after=log_attempt_number,
    #     reraise=True,
    # )
    def generate(self, context: str, prompt: str, **kwargs) -> GenerationOutput:
        dspy_lm = self._get_model(temperature=kwargs.get("temperature", 0.0))
        dspy.settings.configure(lm=dspy_lm)
        cot = DSPyCOT(self.promptSignature)

        # execute LLM generation
        if self.verbose:
            print(f"Generating -- {self.model_name}")
        start_time = time.time()
        # TODO: remove after SIGMOD
        if self.model_name == Model.LLAMA3.value or self.model_name == Model.MIXTRAL.value:
            TOKENS_PER_CHARACTER = 0.25  # noqa: N806
            if len(context) * TOKENS_PER_CHARACTER > 6000:
                context_factor = len(context) * TOKENS_PER_CHARACTER / 6000.0
                context = context[:int(len(context)/context_factor)]

        pred = (
            cot(prompt, context=context)
            if self.prompt_strategy != PromptStrategy.DSPY_COT_MOA_AGG
            else cot(prompt, responses=context)
        )
        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        usage, finish_reason = self._get_usage_and_finish_reason(dspy_lm)

        # collect statistics on prompt, usage, and timing
        usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        # NOTE: needs to match subset of keys produced by LLMConvert._create_empty_query_stats()
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            fn_call_duration_secs=0.0,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_tokens * usd_per_input_token,
            total_output_cost=output_tokens * usd_per_output_token,
            cost_per_record=input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
        )
        # # create GenerationStats
        # stats = GenerationStats(**{
        #     "model_name": self.model_name,
        #     "llm_call_duration_secs": end_time - start_time,
        #     "fn_call_duration_secs": 0.0,
        #     "total_input_tokens": input_tokens,
        #     "total_output_tokens": output_tokens,
        #     "total_input_cost": input_tokens * usd_per_input_token,
        #     "total_output_cost": output_tokens * usd_per_output_token,
        #     "cost_per_record": input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
        #     # "prompt": dspy_lm.history[-1]["prompt"],
        #     # "usage": usage,
        #     # "finish_reason": finish_reason,
        #     # "answer_log_probs": answer_log_probs,
        #     # "answer": pred.answer,
        # })

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)
            print("---")
            print(f"{pred.answer}")
            # output_str = (
            #     f"{question}\n{pred.answer}"
            #     if self.prompt_strategy == PromptStrategy.DSPY_COT_QA
            #     else f"{question}:\n{pred.answer}"
            # )
            # print(output_str)

        return pred.answer, pred.rationale, stats


class ImageTextGenerator(BaseGenerator[List[str | bytes], str]):
    """
    Class for generating field descriptions for an image with a given image model.
    """

    def __init__(self, model: Model, verbose: bool = False):
        super().__init__()
        self.model = model
        self.model_name = model.value
        self.verbose = verbose

    def _decode_image(self, base64_string: str) -> bytes:
        return base64.b64decode(base64_string)

    def _get_model_client(self) -> Union[OpenAI, genai.GenerativeModel]:
        client = None
        if self.model_name in [Model.GPT_4o_V.value, Model.GPT_4o_MINI_V.value]:
            api_key = get_api_key("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

        # elif self.model_name == Model.GEMINI_1V.value:
        #     api_key = get_api_key("GOOGLE_API_KEY")
        #     genai.configure(api_key=api_key)
        #     client = genai.GenerativeModel("gemini-pro-vision")

        elif self.model_name in [Model.LLAMA3_V.value]:
            api_key = get_api_key("TOGETHER_API_KEY")
            client = Together(api_key=api_key)

        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        return client

    def _make_payload(self, prompt: str, base64_images: list[str], temperature: float = 0.0):
        payload = None
        if self.model_name in [Model.GPT_4o_V.value, Model.GPT_4o_MINI_V.value]:
            # create content list
            content: List[Dict[str, str | Dict[str, str]]] = [{"type": "text", "text": prompt}]
            for base64_image in base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

            # create payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": 4000,
                "temperature": temperature,
                "logprobs": True,
            }

        # elif self.model_name == Model.GEMINI_1V.value:
        #     payloads = [
        #         [prompt, Image.open(io.BytesIO(self._decode_image(base64_image)))]
        #         for base64_image in base64_images
        #     ]

        # NOTE: it seems Together + LLama3 can only process a single message; we concat multiple images in operator layer for this model
        elif self.model_name in [Model.LLAMA3_V.value]:
            # create one message per image
            messages = []
            for base64_image in base64_images:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            #"image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            "image_url": {"url": base64_image}, # this will be a URL for Together
                        }
                    ]
                }
                messages.append(message)

            # create payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": temperature,
                "logprobs": 1,
            }

        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        return payload

    def _generate_response(self, client: OpenAI | genai.GenerativeModel, payload: dict[str, Any]) -> tuple[str, str, dict]:
        answer, finish_reason, usage = None, None, None

        if self.model_name in [Model.GPT_4o_V.value, Model.GPT_4o_MINI_V.value]:
            if not isinstance(client, OpenAI):
                raise ValueError("Client must be an instance of OpenAI for GPT-4V model")
            # GPT-4V will always have a single payload
            completion = client.chat.completions.create(**payload)
            candidate = completion.choices[-1]
            answer = candidate.message.content
            finish_reason = candidate.finish_reason
            usage = dict(completion.usage)
            tokens = list(map(lambda elt: elt.token, completion.choices[-1].logprobs.content))
            token_logprobs = list(map(lambda elt: elt.logprob, completion.choices[-1].logprobs.content))

        # elif self.model_name == Model.GEMINI_1V.value:
        #     if not isinstance(client, genai.GenerativeModel):
        #         raise ValueError("Client must be an instance of genai.GenerativeModel for Gemini-1V model")
        #     # iterate through images to generate multiple responses
        #     answers, finish_reasons = [], []
        #     for idx, payload in enumerate(payloads):
        #         response = client.generate_content(payload)
        #         candidate = response.candidates[-1]
        #         answer = f"Image {idx}: " + candidate.content.parts[0].text
        #         finish_reason = candidate.finish_reason
        #         answers.append(answer)
        #         finish_reasons.append(finish_reason)

        #     # combine answers and compute most frequent finish reason
        #     answer = "\n".join(answers)
        #     finish_reason = max(set(finish_reasons), key=finish_reasons.count)

        #     # TODO: implement when google suppports usage and logprob stats
        #     usage = {}
        #     tokens = []
        #     token_logprobs = []

        elif self.model_name in [Model.LLAMA3_V.value]:
            completion = client.chat.completions.create(**payload)
            candidate = completion.choices[-1]
            answer = candidate.message.content
            finish_reason = candidate.finish_reason.value
            usage = dict(completion.usage)
            tokens = []
            token_logprobs = []

        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ",
                self.model_name,
            )

        return answer, finish_reason, usage, tokens, token_logprobs

    def _get_answer_log_probs(self, tokens: List[str], token_logprobs: List[float], answer: str) -> List[float]:
        """
        Filter and return the list of log probabilities for the tokens which appear in `answer`.
        """
        # get indices of the start and end token for the answer

        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
        # if answer.startswith(tokens[start_idx]):
        # end_idx += 1
        # else:
        # start_idx += 1
        # end_idx = start_idx

        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs

        # return those tokens log probabilities
        return answer_log_probs

    # TODO: undo after paper submission
    # @retry(
    #     wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
    #     stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    #     after=log_attempt_number,
    # )
    def generate(self, context: List[str | bytes], prompt: str, **kwargs) -> GenerationOutput:
        # NOTE: context is list of base64 images and question is prompt
        # fetch model client
        client = self._get_model_client()

        # create payload
        payload = self._make_payload(prompt, context, temperature=kwargs.get("temperature", 0.0))

        # generate response
        if self.verbose:
            print(f"Generating -- {self.model_name}")
        start_time = time.time()
        answer, finish_reason, usage, tokens, token_logprobs = self._generate_response(client, payload)
        end_time = time.time()
        if self.verbose:
            print(answer)

        # collect statistics on prompt, usage, and timing
        usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        # create GenerationStats
        stats = GenerationStats(
            **{
                "model_name": self.model_name,
                "llm_call_duration_secs": end_time - start_time,
                "fn_call_duration_secs": 0.0,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_input_cost": input_tokens * usd_per_input_token,
                "total_output_cost": output_tokens * usd_per_output_token,
                "cost_per_record": input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
                # "prompt": dspy_lm.history[-1]["prompt"],
                # "usage": usage,
                # "finish_reason": finish_reason,
                # "answer_log_probs": answer_log_probs,
                # "answer": pred.answer,
            }
        )

        return answer, None, stats


# TODO: refactor this to have a CodeSynthGenerator
def code_execution(api: API, code: str, candidate_dict: Dict[str, Any], verbose: bool = False):
    inputs = {field_name: candidate_dict[field_name] for field_name in api.inputs}
    response = api.api_execute(code, inputs)
    pred = response["response"] if response["status"] and response["response"] else None
    return pred


# Temporarily set default verbose to True for debugging
def code_ensemble_execution(
    api: API, code_ensemble: Dict[str, str], candidate_dict: Dict[str, Any], verbose: bool = True
) -> GenerationOutput:
    start_time = time.time()
    try:
        preds = list()
        for _, code in code_ensemble.items():
            pred = code_execution(api, code, candidate_dict)
            preds.append(pred)

        preds = [pred for pred in preds if pred is not None]
        print(preds)

        # TODO: short-term hack to avoid calling Counter(preds) when preds is a list for biofabric (which is unhashable)
        #       
        if len(preds) == 1:
            majority_response = preds[0]
            exec_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)
            return majority_response, None, exec_stats

        if len(preds) > 0:
            majority_response = Counter(preds).most_common(1)[0][0]
            exec_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)
            # return majority_response+(" (codegen)" if verbose else ""), ensemble_stats
            return majority_response, None, exec_stats

    except Exception:
        pass

    return None, None, GenerationStats(fn_call_duration_secs=time.time() - start_time)

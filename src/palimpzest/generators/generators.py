"""GV: This class is about LLM wrappers. 
My suggestion is to rename at least the base generator into LLMGenerator.
See llm_wrapper.py for a proposed refactoring of generators.py using the class factory pattern.
"""
from palimpzest.constants import *
from palimpzest.generators import (
    dspyCOT,
    gen_filter_signature_class,
    gen_moa_agg_qa_signature_class,
    gen_qa_signature_class,
    TogetherHFAdaptor,
)
from palimpzest.dataclasses import GenerationStats
from palimpzest.utils import API

from collections import Counter
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from typing import Any, Dict, List, Tuple, Union

import google.generativeai as genai

import base64
import dsp
import dspy
import io
import os
import time

# DEFINITIONS
GenerationOutput = Tuple[str, GenerationStats]


def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError(f"key not found in environment variables")

    return os.environ[key]


class BaseGenerator:
    """
    Abstract base class for Generators.
    """

    def __init__(self):
        pass

    def generate(self) -> GenerationOutput:
        raise NotImplementedError("Abstract method")


class CustomGenerator(BaseGenerator):
    """
    Class for generating outputs with a given model using a custom prompt.
    """

    def __init__(self, model_name: str, verbose: bool = False):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose

    def _get_model(self, temperature: float=0.0) -> dsp.LM:
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
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
            )

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["choices"][-1][
                "finish_reason"
            ]
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     usage = {"prompt_tokens": 0, "completion_tokens": 0}
        #     finish_reason = (
        #         dspy_lm.history[-1]["response"][0]._result.candidates[0].finish_reason
        #     )
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["finish_reason"]

        return usage, finish_reason

    def _get_attn(self, dspy_lm: dsp.LM):
        """
        TODO
        """
        pass

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float]:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        tokens, token_logprobs = None, None

        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]["response"]["choices"][-1]["logprobs"][
                "content"
            ]
            tokens = list(map(lambda elt: elt["token"], log_probs))
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
            tokens = dspy_lm.history[-1]["response"]["tokens"]
            token_logprobs = dspy_lm.history[-1]["response"]["token_logprobs"]
        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
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
    def generate(self, prompt: str, temperature: float=0.0) -> GenerationOutput:
        # fetch model
        dspy_lm = self._get_model(temperature=temperature)

        start_time = time.time()

        response = dspy_lm.request(prompt)

        end_time = time.time()

        answer = response["choices"][0]["message"]["content"]
        answer_log_probs = response["choices"][0]["logprobs"]
        finish_reason = response["choices"][0]["finish_reason"]
        usage = response["usage"]

        # collect statistics on prompt, usage, and timing
        usd_per_input_token = MODEL_CARDS[self.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model_name]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        # create GenerationStats
        stats = GenerationStats(**{
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
        })

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)

        return answer, stats


class DSPyGenerator(BaseGenerator):
    """
    Class for generating outputs with a given model using DSPy for prompting optimization(s).
    """

    def __init__(
        self,
        model_name: str,
        prompt_strategy: PromptStrategy,
        doc_schema: str,
        doc_type: str,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
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
            raise ValueError(
                f"DSPyGenerator does not support prompt_strategy: {prompt_strategy.value}"
            )

    def _get_model(self, temperature: float=0.0) -> dsp.LM:
        model = None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            openai_key = get_api_key("OPENAI_API_KEY")
            max_tokens = (
                4096 if self.prompt_strategy in [PromptStrategy.DSPY_COT_QA, PromptStrategy.DSPY_COT_MOA_AGG] else 250
            )
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
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
            )

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["choices"][-1][
                "finish_reason"
            ]
        # elif self.model_name in [Model.GEMINI_1.value]:
        #     usage = {"prompt_tokens": 0, "completion_tokens": 0}
        #     finish_reason = (
        #         dspy_lm.history[-1]["response"][0]._result.candidates[0].finish_reason
        #     )
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA3.value]:
            usage = dspy_lm.history[-1]["response"]["usage"]
            finish_reason = dspy_lm.history[-1]["response"]["finish_reason"]

        return usage, finish_reason

    def _get_attn(self, dspy_lm: dsp.LM):
        """
        TODO
        """
        pass

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float]:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        tokens, token_logprobs = None, None

        if self.model_name in [Model.GPT_4o.value, Model.GPT_4o_MINI.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]["response"]["choices"][-1]["logprobs"][
                "content"
            ]
            tokens = list(map(lambda elt: elt["token"], log_probs))
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
            tokens = dspy_lm.history[-1]["response"]["tokens"]
            token_logprobs = dspy_lm.history[-1]["response"]["token_logprobs"]
        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
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
    def generate(
        self,
        context: str,
        question: str,
        temperature: float=0.0,
    ) -> GenerationOutput:
        # configure DSPy
        dspy_lm = self._get_model(temperature=temperature)
        dspy.settings.configure(lm=dspy_lm)
        cot = dspyCOT(self.promptSignature)

        # execute LLM generation
        if self.verbose:
            print(f"Generating -- {self.model_name}")
        start_time = time.time()
        # TODO: remove after SIGMOD
        if self.model_name == Model.LLAMA3.value or self.model_name == Model.MIXTRAL.value:
            TOKENS_PER_CHARACTER = 0.25
            if len(context) * TOKENS_PER_CHARACTER > 6000:
                context_factor = len(context) * TOKENS_PER_CHARACTER / 6000.0
                context = context[:int(len(context)/context_factor)]

        pred = (
            cot(question, context=context)
            if self.prompt_strategy != PromptStrategy.DSPY_COT_MOA_AGG
            else cot(question, responses=context)
        )
        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._get_answer_log_probs(dspy_lm, pred.answer)
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

        return pred.answer, stats


class ImageTextGenerator(BaseGenerator):
    """
    Class for generating field descriptions for an image with a given image model.
    """

    def __init__(self, model_name: str, verbose: bool = False):
        super().__init__()
        self.model_name = model_name
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
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
            )

        return client

    def _make_payload(self, prompt: str, base64_images: List[str], temperature: float=0.0):
        payload = None
        if self.model_name in [Model.GPT_4o_V.value, Model.GPT_4o_MINI_V.value]:
            # create content list
            content = [{"type": "text", "text": prompt}]
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
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
            )

        return payload

    def _generate_response(
        self, client: Union[OpenAI, genai.GenerativeModel], payload: Dict[str, Any]
    ) -> Tuple[str, str, dict]:
        answer, finish_reason, usage = None, None, None

        if self.model_name in [Model.GPT_4o_V.value, Model.GPT_4o_MINI_V.value]:
            # GPT-4V will always have a single payload
            completion = client.chat.completions.create(**payload)
            candidate = completion.choices[-1]
            answer = candidate.message.content
            finish_reason = candidate.finish_reason
            usage = dict(completion.usage)
            tokens = list(
                map(lambda elt: elt.token, completion.choices[-1].logprobs.content)
            )
            token_logprobs = list(
                map(lambda elt: elt.logprob, completion.choices[-1].logprobs.content)
            )

        # elif self.model_name == Model.GEMINI_1V.value:
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
                "Model must be one of the language models specified in palimpzest.constants.Model but it is ", self.model_name
            )

        return answer, finish_reason, usage, tokens, token_logprobs

    def _get_answer_log_probs(
        self, tokens: List[str], token_logprobs: List[float], answer: str
    ) -> List[float]:
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
    def generate(self, context: List[bytes], question: str, temperature: float=0.0) -> GenerationOutput:
        # NOTE: context is list of base64 images and question is prompt
        # fetch model client
        client = self._get_model_client()

        # create payload
        payload = self._make_payload(question, context, temperature=temperature)

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
        stats = GenerationStats(**{
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
        })

        return answer, stats



# TODO: refactor this to have a CodeSynthGenerator
def codeExecution(api: API, code: str, candidate_dict: Dict[str, Any], verbose:bool=False):
    inputs = {field_name: candidate_dict[field_name] for field_name in api.inputs}
    response = api.api_execute(code, inputs)
    pred = response['response'] if response['status'] and response['response'] else None
    return pred

# Temporarily set default verbose to True for debugging
def codeEnsembleExecution(api: API, code_ensemble: List[Dict[str, str]], candidate_dict: Dict[str, Any], verbose: bool=True) -> GenerationOutput:
    start_time = time.time()
    try:
        preds = list()
        for _, code in code_ensemble.items():
            pred = codeExecution(api, code, candidate_dict)
            preds.append(pred)

        preds = [pred for pred in preds if pred is not None]
        print(preds)

        # TODO: short-term hack to avoid calling Counter(preds) when preds is a list for biofabric (which is unhashable)
        #       
        if len(preds) == 1:
            majority_response = preds[0]
            exec_stats = GenerationStats(**{
                "fn_call_duration_secs": time.time() - start_time,
            })
            return majority_response, exec_stats

        if len(preds) > 0:
            majority_response = Counter(preds).most_common(1)[0][0]
            exec_stats = GenerationStats(**{
                "fn_call_duration_secs": time.time() - start_time,
            })
            # return majority_response+(" (codegen)" if verbose else ""), ensemble_stats
            return majority_response, exec_stats
    except:
        pass

    return None, GenerationStats(**{
            "fn_call_duration_secs": time.time() - start_time,
        })

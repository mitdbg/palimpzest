"""GV: 
Maybe this class should be deleted? 
This class is about LLM wrappers. 
My suggestion is to rename at least the base generator into LLMGenerator.
Refactoring goal: provide a single interface to "LLM Generation" where we specify the model and prompt as input.
This class serves as a wrapper around the DSPy and OpenAI models, abstracting away vendor-specific interfaces and parsing of the generation outputs.

How I envision the code to be called within the physical planner.

for model_name in available_models:
    generator = LLMGeneratorFactory(model_name)
    answer, stats = generator.generate(prompt)

"""

import json
from palimpzest.constants import *
from palimpzest.generators import (
    dspyCOT,
    gen_filter_signature_class,
    gen_qa_signature_class,
    TogetherHFAdaptor,
)
from palimpzest.profiler import GenerationStats
from palimpzest.datamanager import DataDirectory

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, List, Tuple, Union

import google.generativeai as genai

import base64
import dsp
import dspy
import io
import os
import time

from palimpzest.profiler.attentive_trim import (
    find_best_range,
    trim_context,
    best_substring_match,
    update_heatmap_json,
)

# DEFINITIONS
GenerationOutput = Tuple[str, GenerationStats]


def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError(f"key not found in environment variables")

    return os.environ[key]


class LLMGeneratorWrapper:
    """
    Abstract base class for LLM calls.
    No matter the API of the LLM itself, its interface should be the same.
    We could or could not use DsPY as a unique interface for all LLMs - but this decouples our dependency to it.
    """

    def __init__(self, model_name: str):
        """Implementation of the LLMGeneratorWrapper abstract class SHOULD instantiate a model parameter"""
        self.model_name = model_name
        self.model = None

    def generate(self) -> GenerationOutput:
        raise NotImplementedError("Abstract method")

    # GV: Perhaps is overkill to have a function and we can just store usage and finish_reason as variables in the class after a generation is run?
    # This intuition is confirmed by the fact that the only places where these values are used are in the generate() function
    def get_usage_and_finish_reason(self) -> Tuple[dict, str]:
        """
        Parse and return the usage statistics and finish reason.
        Output: usage, finish_reason
        """
        raise NotImplementedError("Abstract method")

    def get_attn(self):
        """
        TODO
        """
        pass

    def get_answer_log_probs(self, answer: str) -> List[float]:
        """
        For the given LLM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        raise NotImplementedError("Abstract method")
        # Dumping commented code for reference
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

    def get_prompt(self) -> str:
        """
        Return the prompt used for the last generation call.
        """
        raise NotImplementedError("Abstract method")


class LLMGeneratorFactory:
    """
    Factory class to create the right LLM generator based on the model name.
    """

    @staticmethod
    def __call__(model_name: str) -> LLMGeneratorWrapper:
        """
        This method returns the appropriate LLM wrapper for a given model name
        """

        if model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            api_key = get_api_key("OPENAI_API_KEY")
            return OpenAIWrapper(model_name, api_key)
        elif model_name in [Model.GEMINI_1.value]:
            api_key = get_api_key("GOOGLE_API_KEY")
            return GoogleWrapper(model_name, api_key)
        elif model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            api_key = get_api_key("TOGETHER_API_KEY")
            return TogetherWrapper(model_name, api_key)
        elif model_name in [Model.GPT_4V.value]:
            api_key = get_api_key("OPENAI_API_KEY")
            return OpenAIVisionWrapper()
        elif model_name in [Model.GEMINI_1V.value]:
            api_key = get_api_key("GOOGLE_API_KEY")
            return GeminiVisionWrapper()
        else:
            raise ValueError(
                "Model must be one of the language models specified in palimpzest.constants.Model"
            )


# TODO GV: This class has little to do with DSPyGenerator - because in its simplest form is only using DSPY as an interface to the models.
# On the other hand, the DSPyGenerator is a class that actively uses DSPy to optimize prompts.
class DSPyInterfaceWrapper(LLMGeneratorWrapper):
    """This class exists as a middle man for all models that are supported by DSPy. We can reuse the generate() code because dspy provides a common interface.
    Alternatively, we can choose to get rid of the get_usage_... and get_answer... methods and code unique generate() functions per model class.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        dspy.settings.configure(self.model)

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
        reraise=True,
    )
    def generate(self, prompt: str) -> GenerationOutput:
        # fetch model
        start_time = time.time()
        response = self.model.request(prompt)
        end_time = time.time()

        answer = response["choices"][0]["message"]["content"]
        answer_log_probs = response["choices"][0]["logprobs"]
        finish_reason = response["choices"][0]["finish_reason"]
        usage = response["usage"]
        print("Generator usage", usage)

        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=self.model.history[-1]["prompt"],
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=answer,
        )

        if self.verbose:
            print("Prompt history:")
            self.model.inspect_history(n=1)

        return answer, stats

    def get_prompt(self) -> str:
        return self.model.history[-1]["prompt"]

    def inspect_history(self, n: int = 1):
        self.model.inspect_history(n)


class OpenAIWrapper(DSPyInterfaceWrapper):

    def __init__(
        self,
        model_name: str,
        api_key: str,
    ):

        super().__init__(model_name)
        max_tokens = 4096
        self.model = dspy.OpenAI(
            model=self.model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=True,
        )

    def get_usage_and_finish_reason(self) -> Tuple[dict, str]:
        usage = self.model.history[-1]["response"]["usage"]
        finish_reason = self.model.history[-1]["response"]["choices"][-1][
            "finish_reason"
        ]
        return usage, finish_reason

    def get_answer_log_probs(self, answer: str) -> List[float]:
        # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
        log_probs = self.model.history[-1]["response"]["choices"][-1]["logprobs"][
            "content"
        ]
        tokens = list(map(lambda elt: elt["token"], log_probs))
        token_logprobs = list(map(lambda elt: elt["logprob"], log_probs))

        return token_logprobs


class GoogleWrapper(DSPyInterfaceWrapper):

    def __init__(
        self,
        model_name: str,
        api_key: str,
    ):

        super().__init__(model_name)
        self.model = dspy.Google(model=self.model_name, api_key=api_key)

    def get_usage_and_finish_reason(self) -> Tuple[dict, str]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason = (
            self.model.history[-1]["response"][0]._result.candidates[0].finish_reason
        )
        return usage, finish_reason

    def get_answer_log_probs(self, answer: str) -> List[float]:
        # TODO Google gemini does not provide log probabilities!
        # https://github.com/google/generative-ai-python/issues/238
        # tok_count = self.model.llm.count_tokens(answer).total_tokens
        # tokens = [""] * tok_count
        # token_logprobs = [0] * len(tokens)
        return [0.0]


class TogetherWrapper(DSPyInterfaceWrapper):

    def __init__(
        self,
        model_name: str,
        api_key: str,
    ):

        super().__init__(model_name)
        self.model = TogetherHFAdaptor(model_name, api_key, logprobs=1)

    def get_usage_and_finish_reason(self) -> Tuple[dict, str]:
        usage = self.model.history[-1]["response"]["usage"]
        finish_reason = self.model.history[-1]["response"]["finish_reason"]
        return usage, finish_reason

    def get_answer_log_probs(self, answer: str) -> List[float]:
        # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
        tokens = self.model.history[-1]["response"]["tokens"]
        token_logprobs = self.model.history[-1]["response"]["token_logprobs"]
        return token_logprobs


class DSPyGenerator(LLMGeneratorWrapper):
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

        # TODO is this clean or spaghetti?
        self._generator = LLMGeneratorFactory(model_name)

        # set prompt signature based on prompt_strategy
        if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            self.promptSignature = gen_filter_signature_class(doc_schema, doc_type)
        elif prompt_strategy == PromptStrategy.DSPY_COT_QA:
            self.promptSignature = gen_qa_signature_class(doc_schema, doc_type)
        else:
            raise ValueError(
                f"DSPyGenerator does not support prompt_strategy: {prompt_strategy.value}"
            )

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
        reraise=True,
    )
    # the generate method requires a user-provided budget parameter to specify te token budget. Default is 1.0, meaning the full context will be used.
    def generate(
        self,
        context: str,
        question: str,
        budget: float = 1.0,
        heatmap_json_obj: dict = None,
    ) -> GenerationOutput:
        # initialize variables around token reduction
        reduction, full_context = False, context

        # fetch model
        # configure DSPy to use this model; both DSPy prompt strategies currently use COT
        cot = dspyCOT(self.promptSignature)

        json_object = {}
        heatmap_file = ""
        # check if the promptSignature is a QA signature, so we can match the answer to get heatmap
        if budget < 1.0 and self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # file_cache = DataDirectory().getFileCacheDir()
            prompt_schema = self.promptSignature
            # print("Prompt QA Signature: ", prompt_schema)
            # print('Question: ', question)
            # ordered = f'{prompt_schema} {question} {plan_idx}'
            # task_hash = hashlib.sha256(ordered.encode()).hexdigest()
            # heatmap_file = os.path.join(file_cache, f"heatmap-{task_hash}.json")
            # print("Heatmap file: ", heatmap_file)
            # if not os.path.exists(heatmap_file):
            if heatmap_json_obj is None:
                # create the heatmap structure with default resolution of 0.001 and count of 0
                buckets = int(1.0 / TOKEN_REDUCTION_GRANULARITY)
                hist = [0] * buckets
                heatmap_json_obj = {
                    "prompt_schema": f"{prompt_schema}",
                    "question": question,
                    "resolution": TOKEN_REDUCTION_GRANULARITY,
                    "count": 0,
                    "heatmap": hist,
                }

            else:
                # only parse the heatmap file if token reduction is enabled (budget is less than 1.0)
                # with open(heatmap_file, 'r') as f:
                #     json_object = json.load(f)
                #     heatmap = json_object['heatmap']
                #     count = json_object['count']
                #     print("count:", count)
                heatmap = heatmap_json_obj["heatmap"]
                count = heatmap_json_obj["count"]
                print("count:", count)
                # only refer to the heatmap if the count is greater than a enough sample size
                # TODO: only trim the context if the attention is clustered in a small region
                if count >= TOKEN_REDUCTION_SAMPLE:
                    si, ei = find_best_range(
                        heatmap,
                        int(budget / TOKEN_REDUCTION_GRANULARITY),
                        trim_zeros=False,
                    )
                    sr, er = (
                        si * TOKEN_REDUCTION_GRANULARITY,
                        ei * TOKEN_REDUCTION_GRANULARITY,
                    )
                    print("start ratio:", sr, "end ratio:", er)
                    context = trim_context(context, sr, er)
                    reduction = True

        # execute LLM generation
        start_time = time.time()

        print(f"Generating -- {self.model_name} -- Token budget: {budget}")
        # print(f"FALL BACK question: {question}")
        # print(f"FALL BACK CONTEXT")
        # print("--------------------")
        # print(f"{context}")
        pred = cot(question, context)

        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._generator.get_answer_log_probs(pred.answer)
        usage, finish_reason = self._generator.get_usage_and_finish_reason()
        prompt = self._generator.get_prompt()

        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=prompt,
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=pred.answer,
        )

        # if reduction is enabled but the answer is None, fallback to the full context
        if reduction and pred.answer == None:
            # run query on full context
            pred = cot(question, full_context)

            # NOTE: in the future, we should capture each of these^ calls in two separate
            #       GenerationStats objects, but for now we just aggregate them
            end_time = time.time()

            # extract the log probabilities for the actual result(s) which are returned
            answer_log_probs = self._generator.get_answer_log_probs(pred.answer)
            usage, finish_reason = self._generator.get_usage_and_finish_reason()
            prompt = self._generator.get_prompt()

            stats.llm_call_duration_secs = end_time - start_time
            stats.prompt = prompt
            for k, _ in stats.usage.items():
                stats.usage[k] += usage[k]
            stats.finish_reason = finish_reason
            stats.answer_log_probs = answer_log_probs
            stats.answer = pred.answer

        # print("----------------")
        # print(f"PROMPT")
        # print("----------------")
        # print(dspy_lm.history[-1]['prompt'])

        # print("----------------")
        # print(f"ANSWER")
        # print("----------------")
        print(pred.answer)

        # taken reduction post processing if enabled
        if (
            budget < 1.0
            and self.prompt_strategy == PromptStrategy.DSPY_COT_QA
            and heatmap_json_obj["count"] < MAX_HEATMAP_UPDATES
        ):
            print("Reduction enabled")
            print("answer:", pred.answer)
            try:
                gsi, gei = best_substring_match(pred.answer, full_context)
            except Exception as e:
                print("Error in substring match:", e)
                gsi, gei = 0, len(full_context)
            context_len = len(full_context)
            gsr, ger = gsi / context_len, gei / context_len
            norm_si, norm_ei = int(gsr / TOKEN_REDUCTION_GRANULARITY), int(
                ger / TOKEN_REDUCTION_GRANULARITY
            )
            print("best_start:", gsi, "best_end:", gei)
            heatmap_json_obj = update_heatmap_json(heatmap_json_obj, norm_si, norm_ei)
            # with open(heatmap_file, 'w') as f:
            #     json.dump(json_object, f)

        if self.verbose:
            print("Prompt history:")
            self._generator.inspect_history(n=1)

        return pred.answer, heatmap_json_obj, stats


class ImageTextGenerator(LLMGeneratorWrapper):
    """
    Middle class wrapping LLM models that can generate field descriptions for an image with a given image model.
    """

    def _decode_image(self, base64_string: str) -> bytes:
        return base64.b64decode(base64_string)

    def _make_payloads(self, prompt: str, base64_images: List[str]):
        raise NotImplementedError("Abstract method")

    def get_answer_log_probs(
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

    def _generate_response(self, payloads: List[Any]) -> Tuple[str, str, dict]:
        raise NotImplementedError("Abstract method")

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
    )
    def generate(self, base64_images: str, prompt: str) -> GenerationOutput:
        # fetch model client
        # create payload
        payloads = self._make_payloads(prompt, base64_images)

        # generate response
        print(f"Generating")
        start_time = time.time()
        answer, finish_reason, usage, tokens, token_logprobs = self._generate_response(
            payloads
        )
        end_time = time.time()
        if self.verbose:
            print(answer)

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self.get_answer_log_probs(tokens, token_logprobs, answer)

        # TODO: To simplify life for the time being, I am aggregating stats for multiple call(s)
        #       to the Gemini vision model into a single GenerationStats object (when we have
        #       more than one image to process). This has no effect on most of our fields --
        #       especially since many of them are not implemented for the Gemini model -- but
        #       we will likely want a more robust solution in the future.
        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=prompt,
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=answer,
        )

        return answer, stats


class OpenAIVisionWrapper(ImageTextGenerator):

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name=model_name)
        self.model = OpenAI(api_key=api_key)

    def _make_payloads(self, prompt: str, base64_images: List[str]):
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
        payloads = [
            {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.0,
                "logprobs": True,
            }
        ]
        return payloads

    def _generate_response(self, payloads: List[Any]) -> Tuple[str, str, dict]:
        # GPT-4V will always have a single payload
        completion = self.model.chat.completions.create(**payloads[0])
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
        return answer, finish_reason, usage, tokens, token_logprobs


class GeminiVisionWrapper(ImageTextGenerator):

    def __init__(self, model_name: str, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-pro-vision"
        )  # Why is this hardcoded? Shouldn't it be model_name?

    def _make_payloads(self, prompt: str, base64_images: List[str]):
        payloads = [
            [prompt, Image.open(io.BytesIO(self._decode_image(base64_image)))]
            for base64_image in base64_images
        ]
        return payloads

    def _generate_response(self, payloads: List[Any]) -> Tuple[str, str, dict]:
        # iterate through images to generate multiple responses
        answers, finish_reasons = [], []
        for idx, payload in enumerate(payloads):
            response = self.model.generate_content(payload)
            candidate = response.candidates[-1]
            answer = f"Image {idx}: " + candidate.content.parts[0].text
            finish_reason = candidate.finish_reason
            answers.append(answer)
            finish_reasons.append(finish_reason)

        # combine answers and compute most frequent finish reason
        answer = "\n".join(answers)
        finish_reason = max(set(finish_reasons), key=finish_reasons.count)

        # TODO: implement when google suppports usage and logprob stats
        usage = {}
        tokens = []
        token_logprobs = []
        return answer, finish_reason, usage, tokens, token_logprobs

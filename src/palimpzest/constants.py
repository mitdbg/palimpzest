### This file contains constants used by Palimpzest ###
import os
from enum import Enum


# ENUMS
class Model(str, Enum):
    """
    Model describes the underlying LLM which should be used to perform some operation
    which requires invoking an LLM. It does NOT specify whether the model need be executed
    remotely or locally (if applicable).
    """
    LLAMA3_2_3B = "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo"
    LLAMA3_1_8B = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA3_3_70B = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    LLAMA3_2_90B_V = "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    DEEPSEEK_V3 = "together_ai/deepseek-ai/DeepSeek-V3"
    DEEPSEEK_R1_DISTILL_QWEN_1_5B = "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    GPT_4o = "openai/gpt-4o-2024-08-06"
    GPT_4o_MINI = "openai/gpt-4o-mini-2024-07-18"
    GPT_5 = "openai/gpt-5"
    GPT_5_MINI = "openai/gpt-5-mini"
    o4_MINI = "openai/o4-mini-2025-04-16"  # noqa: N815
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    CLIP_VIT_B_32 = "clip-ViT-B-32"
    CLAUDE_3_5_SONNET = "anthropic/claude-3-5-sonnet-20241022"
    CLAUDE_3_7_SONNET = "anthropic/claude-3-7-sonnet-20250219"
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-20241022"
    GEMINI_2_0_FLASH = "vertex_ai/gemini-2.0-flash"
    GEMINI_2_5_FLASH = "vertex_ai/gemini-2.5-flash"
    GEMINI_2_5_PRO = "vertex_ai/gemini-2.5-pro"
    LLAMA_4_MAVERICK = "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"
    GPT_4o_AUDIO_PREVIEW = "openai/gpt-4o-audio-preview"
    GPT_4o_MINI_AUDIO_PREVIEW = "openai/gpt-4o-mini-audio-preview"
    # o1 = "o1-2024-12-17"

    def __repr__(self):
        return f"{self.name}"

    def is_llama_model(self):
        return "llama" in self.value.lower()

    def is_clip_model(self):
        return "clip" in self.value.lower()

    def is_together_model(self):
        return "together_ai" in self.value.lower() or self.is_clip_model()

    def is_text_embedding_model(self):
        return "text-embedding" in self.value.lower()

    def is_o_model(self):
        return self in [Model.o4_MINI]

    def is_gpt_5_model(self):
        return self in [Model.GPT_5, Model.GPT_5_MINI]

    def is_openai_model(self):
        return "openai" in self.value.lower() or self.is_text_embedding_model()

    def is_anthropic_model(self):
        return "anthropic" in self.value.lower()

    def is_vertex_model(self):
        return "vertex_ai" in self.value.lower()

    def is_reasoning_model(self):
        reasoning_models = [
            Model.GPT_5, Model.GPT_5_MINI, Model.o4_MINI,
            Model.GEMINI_2_5_PRO, Model.GEMINI_2_5_FLASH,
            Model.CLAUDE_3_7_SONNET,
        ]
        return self in reasoning_models

    def is_text_model(self):
        non_text_models = [
            Model.LLAMA3_2_90B_V,
            Model.CLIP_VIT_B_32, Model.TEXT_EMBEDDING_3_SMALL,
            Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o_MINI_AUDIO_PREVIEW,
        ]
        return self not in non_text_models

    # TODO: I think SONNET and HAIKU are vision-capable too
    def is_vision_model(self):
        return self in [
            Model.LLAMA3_2_90B_V, Model.LLAMA_4_MAVERICK,
            Model.GPT_4o, Model.GPT_4o_MINI, Model.o4_MINI, Model.GPT_5, Model.GPT_5_MINI,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
        ]

    def is_audio_model(self):
        return self in [
            Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o_MINI_AUDIO_PREVIEW,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
        ]

    def is_text_image_multimodal_model(self):
        return self in [
            Model.LLAMA_4_MAVERICK,
            Model.GPT_4o, Model.GPT_4o_MINI, Model.o4_MINI, Model.GPT_5, Model.GPT_5_MINI,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
        ]

    def is_text_audio_multimodal_model(self):
        return self in [
            Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o_MINI_AUDIO_PREVIEW,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
        ]

    def is_embedding_model(self):
        return self in [Model.CLIP_VIT_B_32, Model.TEXT_EMBEDDING_3_SMALL]


class PromptStrategy(str, Enum):
    """
    PromptStrategy describes the prompting technique to be used by a Generator when
    performing some task with a specified Model.
    """

    # Chain-of-Thought Boolean Prompt Strategies
    COT_BOOL = "chain-of-thought-bool"
    # COT_BOOL_CRITIC = "chain-of-thought-bool-critic"
    # COT_BOOL_REFINE = "chain-of-thought-bool-refine"

    # Chain-of-Thought Boolean with Image Prompt Strategies
    COT_BOOL_IMAGE = "chain-of-thought-bool-image"
    COT_BOOL_AUDIO = "chain-of-thought-bool-audio"
    # COT_BOOL_IMAGE_CRITIC = "chain-of-thought-bool-image-critic"
    # COT_BOOL_IMAGE_REFINE = "chain-of-thought-bool-image-refine"

    # Chain-of-Thought Join Prompt Strategies
    COT_JOIN = "chain-of-thought-join"
    COT_JOIN_IMAGE = "chain-of-thought-join-image"
    COT_JOIN_AUDIO = "chain-of-thought-join-audio"

    # Chain-of-Thought Question Answering Prompt Strategies
    COT_QA = "chain-of-thought-question"
    COT_QA_CRITIC = "chain-of-thought-question-critic"
    COT_QA_REFINE = "chain-of-thought-question-refine"

    # Chain-of-Thought Question with Image Prompt Strategies
    COT_QA_IMAGE = "chain-of-thought-question-image"
    COT_QA_IMAGE_CRITIC = "chain-of-thought-question-critic-image"
    COT_QA_IMAGE_REFINE = "chain-of-thought-question-refine-image"

    # Chain-of-Thought Queestion with Audio Prompt Strategies
    COT_QA_AUDIO = "chain-of-thought-question-audio"
    # TODO: COT_QA_AUDIO_CRITIC/REFINE

    # Mixture-of-Agents Prompt Strategies
    COT_MOA_PROPOSER = "chain-of-thought-mixture-of-agents-proposer"
    COT_MOA_PROPOSER_IMAGE = "chain-of-thought-mixture-of-agents-proposer-image"
    COT_MOA_AGG = "chain-of-thought-mixture-of-agents-aggregation"
    # TODO: COT_MOA_PROPOSER_AUDIO 

    # Split Convert Prompt Strategies
    SPLIT_PROPOSER = "split-proposer"
    SPLIT_MERGER = "split-merger"

    def is_image_prompt(self):
        return "image" in self.value

    def is_audio_prompt(self):
        return "audio" in self.value

    def is_bool_prompt(self):
        return "bool" in self.value

    def is_join_prompt(self):
        return "join" in self.value

    def is_convert_prompt(self):
        return "bool" not in self.value and "join" not in self.value

    def is_critic_prompt(self):
        return "critic" in self.value

    def is_refine_prompt(self):
        return "refine" in self.value

    def is_moa_proposer_prompt(self):
        return "mixture-of-agents-proposer" in self.value

    def is_moa_aggregator_prompt(self):
        return "mixture-of-agents-aggregation" in self.value

    def is_split_proposer_prompt(self):
        return "split-proposer" in self.value

    def is_split_merger_prompt(self):
        return "split-merger" in self.value


class AggFunc(str, Enum):
    COUNT = "count"
    AVERAGE = "average"


class Cardinality(str, Enum):
    ONE_TO_ONE = "one-to-one"
    ONE_TO_MANY = "one-to-many"

    @classmethod
    def _missing_(cls, value):
        if value:
            normalized_value = "".join([x for x in value if x.isalpha()]).lower()
            for member in cls:
                normalized_member = "".join([x for x in member if x.isalpha()]).lower()
                if normalized_member == normalized_value:
                    return member
        return cls.ONE_TO_ONE


class PickOutputStrategy(str, Enum):
    CHAMPION = "champion"
    ENSEMBLE = "ensemble"


AUDIO_EXTENSIONS = [".wav"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
PDF_EXTENSIONS = [".pdf"]
XLS_EXTENSIONS = [".xls", ".xlsx"]
HTML_EXTENSIONS = [".html", ".htm"]

# the number of seconds the parallel execution will sleep for while waiting for futures to complete
PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS = 0.3

# default PDF parser
DEFAULT_PDF_PROCESSOR = "pypdf"

# character limit for various IDs
MAX_ID_CHARS = 10

# maximum number of rows to display in a table
MAX_ROWS = 5

# maximum number of rows to parse from an HTML
MAX_HTML_ROWS = 10000


def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    print(f"Retrying: {retry_state.attempt_number}...")


# Palimpzest root directory
PZ_DIR = os.path.join(os.path.expanduser("~"), ".palimpzest")

# Assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# Assume 30 GB/sec for sequential access of memory
MEMORY_SCAN_TIME_PER_KB = 1 / (float(30) * 1024 * 1024)

# Assume 1 KB per record
NAIVE_BYTES_PER_RECORD = 1024

# Rough conversion from # of characters --> # of tokens; assumes 1 token ~= 4 chars
TOKENS_PER_CHARACTER = 0.25

# Rough estimate of the number of tokens the context is allowed to take up for LLAMA3 models
LLAMA_CONTEXT_TOKENS_LIMIT = 6000

# a naive estimate for the input record size
NAIVE_EST_SOURCE_RECORD_SIZE_IN_BYTES = 1_000_000

# a naive estimate for filter selectivity
NAIVE_EST_FILTER_SELECTIVITY = 0.5

# a naive estimate for join selectivity
NAIVE_EST_JOIN_SELECTIVITY = 0.5

# a naive estimate for the number of input tokens processed per record
NAIVE_EST_NUM_INPUT_TOKENS = 1000

# a naive estimate for the number of output tokens processed per record
NAIVE_EST_NUM_OUTPUT_TOKENS = 100

# a naive estimate for the number of groups returned by a group by
NAIVE_EST_NUM_GROUPS = 3

# a naive estimate for the factor of increase (loosely termed "selectivity") for one-to-many cardinality operations
NAIVE_EST_ONE_TO_MANY_SELECTIVITY = 2

# a naive estimate of the time it takes to extract the latex for an equation from an image file using Skema
NAIVE_IMAGE_TO_EQUATION_LATEX_TIME_PER_RECORD = 10.0

# a naive estimate of the time it takes to extract the text from a PDF using a PDF processor
NAIVE_PDF_PROCESSOR_TIME_PER_RECORD = 10.0

# Whether or not to log LLM outputs
LOG_LLM_OUTPUT = False


#### MODEL PERFORMANCE & COST METRICS ####
# Overall model quality is computed using MMLU-Pro; multi-modal models currently use the same score for vision
# - in the future we should split quality for vision vs. multi-modal vs. text
# - code quality was computed using HumanEval, but that benchmark is too easy and should be replaced.
# - https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
# - https://www.vals.ai/benchmarks/mmlu_pro-08-12-2025
#
# Cost is presented in terms of USD / token for input tokens and USD / token for
# generated tokens.
#
# Time is presented in seconds per output token. I grabbed some semi-recent estimates
# from the internet for this quick POC, but we can and should do more to model these
# values more precisely:
# - https://artificialanalysis.ai/models/llama-3-1-instruct-8b
#
LLAMA3_2_3B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.06 / 1e6,
    "usd_per_output_token": 0.06 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0064,
    ##### Agg. Benchmark #####
    "overall": 36.50, # https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/discussions/13
}
LLAMA3_1_8B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1e6,
    "usd_per_output_token": 0.18 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0059,
    ##### Agg. Benchmark #####
    "overall": 44.25,
}
LLAMA3_3_70B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.88 / 1e6,
    "usd_per_output_token": 0.88 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0139,
    ##### Agg. Benchmark #####
    "overall": 69.9,
}
LLAMA3_2_90B_V_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.2 / 1e6,
    "usd_per_output_token": 1.2 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0222,
    ##### Agg. Benchmark #####
    "overall": 65.00, # set to be slightly higher than gpt-4o-mini
}
DEEPSEEK_V3_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1E6,
    "usd_per_output_token": 1.25 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0769,
    ##### Agg. Benchmark #####
    "overall": 73.8,
}
DEEPSEEK_R1_DISTILL_QWEN_1_5B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1E6,
    "usd_per_output_token": 0.18 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0026,
    ##### Agg. Benchmark #####
    "overall": 39.90, # https://www.reddit.com/r/LocalLLaMA/comments/1iserf9/deepseek_r1_distilled_models_mmlu_pro_benchmarks/
}
GPT_4o_AUDIO_PREVIEW_MODEL_CARD = {
    # NOTE: COPYING OVERALL AND SECONDS_PER_OUTPUT_TOKEN FROM GPT_4o; need to update when we have audio-specific benchmarks
    ##### Cost in USD #####
    "usd_per_audio_input_token": 2.5 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0079,
    ##### Agg. Benchmark #####
    "overall": 74.1,
}
GPT_4o_MINI_AUDIO_PREVIEW_MODEL_CARD = {
    # NOTE: COPYING OVERALL AND SECONDS_PER_OUTPUT_TOKEN FROM GPT_4o; need to update when we have audio-specific benchmarks
    ##### Cost in USD #####
    "usd_per_audio_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0098,
    ##### Agg. Benchmark #####
    "overall": 62.7,
}
GPT_4o_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 2.5 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0079,
    ##### Agg. Benchmark #####
    "overall": 74.1,
}
GPT_4o_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0098,
    ##### Agg. Benchmark #####
    "overall": 62.7,
}
GPT_5_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0139,
    ##### Agg. Benchmark #####
    "overall": 87.00,
}
GPT_5_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.25 / 1e6,
    "usd_per_output_token": 2.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0094,
    ##### Agg. Benchmark #####
    "overall": 82.50,
}
o4_MINI_MODEL_CARD = {  # noqa: N816
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 1.1 / 1e6,
    "usd_per_output_token": 4.4 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0093,
    ##### Agg. Benchmark #####
    "overall": 80.6,  # using number reported for o3-mini; true number is likely higher
}
o1_MODEL_CARD = {  # noqa: N816
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 15 / 1e6,
    "usd_per_output_token": 60 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0110,
    ##### Agg. Benchmark #####
    "overall": 83.50,
}
TEXT_EMBEDDING_3_SMALL_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.02 / 1e6,
    "usd_per_output_token": None,
    ##### Time #####
    "seconds_per_output_token": 0.0098,  # NOTE: just copying GPT_4o_MINI_MODEL_CARD for now
    ##### Agg. Benchmark #####
    "overall": 63.09,  # NOTE: just copying GPT_4o_MINI_MODEL_CARD for now
}
CLIP_VIT_B_32_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.00,
    "usd_per_output_token": None,
    ##### Time #####
    "seconds_per_output_token": 0.0098,  # NOTE: just copying TEXT_EMBEDDING_3_SMALL_MODEL_CARD for now
    ##### Agg. Benchmark #####
    "overall": 63.3,  # NOTE: ImageNet top-1 accuracy
}
CLAUDE_3_5_SONNET_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 3.0 / 1e6,
    "usd_per_output_token": 15.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0127,
    ##### Agg. Benchmark #####
    "overall": 78.4,
}
CLAUDE_3_7_SONNET_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 3.0 / 1e6,
    "usd_per_output_token": 15.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0130,
    ##### Agg. Benchmark #####
    "overall": 80.7,
}
CLAUDE_3_5_HAIKU_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.8 / 1e6,
    "usd_per_output_token": 4.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0152,
    ##### Agg. Benchmark #####
    "overall": 64.1,
}
GEMINI_2_0_FLASH_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    "usd_per_audio_input_token": 1.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0049,
    ##### Agg. Benchmark #####
    "overall": 77.40,
}
GEMINI_2_5_FLASH_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.30 / 1e6,
    "usd_per_output_token": 2.5 / 1e6,
    "usd_per_audio_input_token": 1.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0039,
    ##### Agg. Benchmark #####
    "overall": 80.75, # NOTE: interpolated between gemini 2.0 flash and gemini 2.5 pro
}
GEMINI_2_5_PRO_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    "usd_per_audio_input_token": 1.25 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0070,
    ##### Agg. Benchmark #####
    "overall": 84.10,
}
LLAMA_4_MAVERICK_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.35 / 1e6,
    "usd_per_output_token": 1.15 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0058,
    ##### Agg. Benchmark #####
    "overall": 79.4,
}

MODEL_CARDS = {
    Model.LLAMA3_2_3B.value: LLAMA3_2_3B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_1_8B.value: LLAMA3_1_8B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_3_70B.value: LLAMA3_3_70B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_2_90B_V.value: LLAMA3_2_90B_V_MODEL_CARD,
    Model.DEEPSEEK_V3.value: DEEPSEEK_V3_MODEL_CARD,
    Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B.value: DEEPSEEK_R1_DISTILL_QWEN_1_5B_MODEL_CARD,
    Model.GPT_4o.value: GPT_4o_MODEL_CARD,
    Model.GPT_4o_MINI.value: GPT_4o_MINI_MODEL_CARD,
    Model.GPT_4o_AUDIO_PREVIEW.value: GPT_4o_AUDIO_PREVIEW_MODEL_CARD,
    Model.GPT_4o_MINI_AUDIO_PREVIEW.value: GPT_4o_MINI_AUDIO_PREVIEW_MODEL_CARD,
    Model.GPT_5.value: GPT_5_MODEL_CARD,
    Model.GPT_5_MINI.value: GPT_5_MINI_MODEL_CARD,
    Model.o4_MINI.value: o4_MINI_MODEL_CARD,
    # Model.o1.value: o1_MODEL_CARD,
    Model.TEXT_EMBEDDING_3_SMALL.value: TEXT_EMBEDDING_3_SMALL_MODEL_CARD,
    Model.CLIP_VIT_B_32.value: CLIP_VIT_B_32_MODEL_CARD,
    Model.CLAUDE_3_5_SONNET.value: CLAUDE_3_5_SONNET_MODEL_CARD,
    Model.CLAUDE_3_7_SONNET.value: CLAUDE_3_7_SONNET_MODEL_CARD,
    Model.CLAUDE_3_5_HAIKU.value: CLAUDE_3_5_HAIKU_MODEL_CARD,
    Model.GEMINI_2_0_FLASH.value: GEMINI_2_0_FLASH_MODEL_CARD,
    Model.GEMINI_2_5_FLASH.value: GEMINI_2_5_FLASH_MODEL_CARD,
    Model.GEMINI_2_5_PRO.value: GEMINI_2_5_PRO_MODEL_CARD,
    Model.LLAMA_4_MAVERICK.value: LLAMA_4_MAVERICK_MODEL_CARD,
}

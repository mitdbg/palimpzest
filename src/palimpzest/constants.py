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
    GPT_4_1 = "openai/gpt-4.1-2025-04-14"
    GPT_4_1_MINI = "openai/gpt-4.1-mini-2025-04-14"
    GPT_4_1_NANO = "openai/gpt-4.1-nano-2025-04-14"
    GPT_5 = "openai/gpt-5-2025-08-07"
    GPT_5_MINI = "openai/gpt-5-mini-2025-08-07"
    GPT_5_NANO = "openai/gpt-5-nano-2025-08-07"
    o4_MINI = "openai/o4-mini-2025-04-16"  # noqa: N815
    CLAUDE_3_5_SONNET = "anthropic/claude-3-5-sonnet-20241022"
    CLAUDE_3_7_SONNET = "anthropic/claude-3-7-sonnet-20250219"
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-20241022"
    GEMINI_2_0_FLASH = "vertex_ai/gemini-2.0-flash"
    GEMINI_2_5_FLASH = "vertex_ai/gemini-2.5-flash"
    GEMINI_2_5_PRO = "vertex_ai/gemini-2.5-pro"
    GOOGLE_GEMINI_2_5_FLASH = "gemini/gemini-2.5-flash"
    GOOGLE_GEMINI_2_5_FLASH_LITE = "gemini/gemini-2.5-flash-lite"
    GOOGLE_GEMINI_2_5_PRO = "gemini/gemini-2.5-pro"
    LLAMA_4_MAVERICK = "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"
    GPT_4o_AUDIO_PREVIEW = "openai/gpt-4o-audio-preview"
    GPT_4o_MINI_AUDIO_PREVIEW = "openai/gpt-4o-mini-audio-preview"
    VLLM_QWEN_1_5_0_5B_CHAT = "hosted_vllm/qwen/Qwen1.5-0.5B-Chat"
    # o1 = "o1-2024-12-17"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    CLIP_VIT_B_32 = "clip-ViT-B-32"

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
        return self in [Model.GPT_5, Model.GPT_5_MINI, Model.GPT_5_NANO]

    def is_openai_model(self):
        return "openai" in self.value.lower() or self.is_text_embedding_model()

    def is_anthropic_model(self):
        return "anthropic" in self.value.lower()

    def is_vertex_model(self):
        return "vertex_ai" in self.value.lower()

    def is_google_ai_studio_model(self):
        return "gemini/" in self.value.lower()

    def is_vllm_model(self):
        return "hosted_vllm" in self.value.lower()

    def is_reasoning_model(self):
        reasoning_models = [
            Model.GPT_5, Model.GPT_5_MINI, Model.GPT_5_NANO, Model.o4_MINI,
            Model.GEMINI_2_5_PRO, Model.GEMINI_2_5_FLASH,
            Model.GOOGLE_GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
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
            Model.GPT_4o, Model.GPT_4o_MINI, Model.GPT_4_1, Model.GPT_4_1_MINI, Model.GPT_4_1_NANO, Model.o4_MINI, Model.GPT_5, Model.GPT_5_MINI, Model.GPT_5_NANO,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
            Model.GOOGLE_GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
        ]

    def is_audio_model(self):
        return self in [
            Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o_MINI_AUDIO_PREVIEW,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
            Model.GOOGLE_GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
        ]

    def is_text_image_multimodal_model(self):
        return self in [
            Model.LLAMA_4_MAVERICK,
            Model.GPT_4o, Model.GPT_4o_MINI, Model.GPT_4_1, Model.GPT_4_1_MINI, Model.GPT_4_1_NANO, Model.o4_MINI, Model.GPT_5, Model.GPT_5_MINI, Model.GPT_5_NANO,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
            Model.GOOGLE_GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
        ]

    def is_text_audio_multimodal_model(self):
        return self in [
            Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o_MINI_AUDIO_PREVIEW,
            Model.GEMINI_2_0_FLASH, Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO,
            Model.GOOGLE_GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
        ]

    def is_embedding_model(self):
        return self in [Model.CLIP_VIT_B_32, Model.TEXT_EMBEDDING_3_SMALL]


class PromptStrategy(str, Enum):
    """
    PromptStrategy describes the prompting technique to be used by a Generator when
    performing some task with a specified Model.
    """

    # aggregation prompt strategies
    AGG = "aggregation"
    AGG_NO_REASONING = "aggregation-no-reasoning"

    # filter prompt strategies
    FILTER = "filter"
    FILTER_NO_REASONING = "filter-no-reasoning"
    FILTER_CRITIC = "filter-critic"
    FILTER_REFINE = "filter-refine"
    FILTER_MOA_PROPOSER = "filter-mixture-of-agents-proposer"
    FILTER_MOA_AGG = "filter-mixture-of-agents-aggregator"
    FILTER_SPLIT_PROPOSER = "filter-split-proposer"
    FILTER_SPLIT_MERGER = "filter-split-merger"

    # join prompt strategies
    JOIN = "join"
    JOIN_NO_REASONING = "join-no-reasoning"

    # map prompt strategies
    MAP = "map"
    MAP_NO_REASONING = "map-no-reasoning"
    MAP_CRITIC = "map-critic"
    MAP_REFINE = "map-refine"
    MAP_MOA_PROPOSER = "map-mixture-of-agents-proposer"
    MAP_MOA_AGG = "map-mixture-of-agents-aggregator"
    MAP_SPLIT_PROPOSER = "map-split-proposer"
    MAP_SPLIT_MERGER = "map-split-merger"

    def is_agg_prompt(self):
        return "aggregation" in self.value

    def is_filter_prompt(self):
        return "filter" in self.value

    def is_join_prompt(self):
        return "join" in self.value

    def is_map_prompt(self):
        return "map" in self.value

    def is_critic_prompt(self):
        return "critic" in self.value

    def is_refine_prompt(self):
        return "refine" in self.value

    def is_moa_proposer_prompt(self):
        return "mixture-of-agents-proposer" in self.value

    def is_moa_aggregator_prompt(self):
        return "mixture-of-agents-aggregator" in self.value

    def is_split_proposer_prompt(self):
        return "split-proposer" in self.value

    def is_split_merger_prompt(self):
        return "split-merger" in self.value

    def is_no_reasoning_prompt(self):
        return "no-reasoning" in self.value


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class AggFunc(str, Enum):
    COUNT = "count"
    AVERAGE = "average"
    SUM = "sum"
    MIN = "min"
    MAX = "max"

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
    "seconds_per_output_token": 0.0079,
    ##### Agg. Benchmark #####
    "overall": 36.50, # https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/discussions/13
}
LLAMA3_1_8B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1e6,
    "usd_per_output_token": 0.18 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0050,
    ##### Agg. Benchmark #####
    "overall": 44.25,
}
LLAMA3_3_70B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.88 / 1e6,
    "usd_per_output_token": 0.88 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0122,
    ##### Agg. Benchmark #####
    "overall": 69.9,
}
LLAMA3_2_90B_V_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.2 / 1e6,
    "usd_per_output_token": 1.2 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0303,
    ##### Agg. Benchmark #####
    "overall": 65.00, # set to be slightly higher than gpt-4o-mini
}
DEEPSEEK_V3_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1E6,
    "usd_per_output_token": 1.25 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0114,
    ##### Agg. Benchmark #####
    "overall": 73.8,
}
DEEPSEEK_R1_DISTILL_QWEN_1_5B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1E6,
    "usd_per_output_token": 0.18 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0050, # NOTE: copied to be same as LLAMA3_1_8B_INSTRUCT_MODEL_CARD; need to update when we have data
    ##### Agg. Benchmark #####
    "overall": 39.90, # https://www.reddit.com/r/LocalLLaMA/comments/1iserf9/deepseek_r1_distilled_models_mmlu_pro_benchmarks/
}
GPT_4o_AUDIO_PREVIEW_MODEL_CARD = {
    # NOTE: COPYING OVERALL AND SECONDS_PER_OUTPUT_TOKEN FROM GPT_4o; need to update when we have audio-specific benchmarks
    ##### Cost in USD #####
    "usd_per_audio_input_token": 2.5 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0080,
    ##### Agg. Benchmark #####
    "overall": 74.1,
}
GPT_4o_MINI_AUDIO_PREVIEW_MODEL_CARD = {
    # NOTE: COPYING OVERALL AND SECONDS_PER_OUTPUT_TOKEN FROM GPT_4o; need to update when we have audio-specific benchmarks
    ##### Cost in USD #####
    "usd_per_audio_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0159,
    ##### Agg. Benchmark #####
    "overall": 62.7,
}
GPT_4o_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 2.5 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0080,
    ##### Agg. Benchmark #####
    "overall": 74.1,
}
GPT_4o_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0159,
    ##### Agg. Benchmark #####
    "overall": 62.7,
}
GPT_4_1_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 2.0 / 1e6,
    "usd_per_output_token": 8.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0076,
    ##### Agg. Benchmark #####
    "overall": 80.5,
}
GPT_4_1_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.4 / 1e6,
    "usd_per_output_token": 1.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0161,
    ##### Agg. Benchmark #####
    "overall": 77.2,
}
GPT_4_1_NANO_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.1 / 1e6,
    "usd_per_output_token": 0.4 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0060,
    ##### Agg. Benchmark #####
    "overall": 62.3,
}
GPT_5_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0060,
    ##### Agg. Benchmark #####
    "overall": 87.00,
}
GPT_5_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.25 / 1e6,
    "usd_per_output_token": 2.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0135,
    ##### Agg. Benchmark #####
    "overall": 82.50,
}
GPT_5_NANO_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.05 / 1e6,
    "usd_per_output_token": 0.4 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0055,
    ##### Agg. Benchmark #####
    "overall": 77.9,
}
o4_MINI_MODEL_CARD = {  # noqa: N816
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 1.1 / 1e6,
    "usd_per_output_token": 4.4 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0092,
    ##### Agg. Benchmark #####
    "overall": 80.6,  # using number reported for o3-mini; true number is likely higher
}
# o1_MODEL_CARD = {  # noqa: N816
#     # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
#     ##### Cost in USD #####
#     "usd_per_input_token": 15 / 1e6,
#     "usd_per_output_token": 60 / 1e6,
#     ##### Time #####
#     "seconds_per_output_token": 0.0110,
#     ##### Agg. Benchmark #####
#     "overall": 83.50,
# }
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
    "overall": 63.3,  # NOTE: imageNet top-1 accuracy
}
CLAUDE_3_5_SONNET_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 3.0 / 1e6,
    "usd_per_output_token": 15.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0154,
    ##### Agg. Benchmark #####
    "overall": 78.4,
}
CLAUDE_3_7_SONNET_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 3.0 / 1e6,
    "usd_per_output_token": 15.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0156,
    ##### Agg. Benchmark #####
    "overall": 80.7,
}
CLAUDE_3_5_HAIKU_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.8 / 1e6,
    "usd_per_output_token": 4.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0189,
    ##### Agg. Benchmark #####
    "overall": 64.1,
}
GEMINI_2_0_FLASH_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    "usd_per_audio_input_token": 1.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0054,
    ##### Agg. Benchmark #####
    "overall": 77.40,
}
GEMINI_2_5_FLASH_LITE_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.1 / 1e6,
    "usd_per_output_token": 0.4 / 1e6,
    "usd_per_audio_input_token": 0.3 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0034,
    ##### Agg. Benchmark #####
    "overall": 79.1, # NOTE: interpolated between gemini 2.5 flash and gemini 2.0 flash
}
GEMINI_2_5_FLASH_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.30 / 1e6,
    "usd_per_output_token": 2.5 / 1e6,
    "usd_per_audio_input_token": 1.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0044,
    ##### Agg. Benchmark #####
    "overall": 80.75, # NOTE: interpolated between gemini 2.0 flash and gemini 2.5 pro
}
GEMINI_2_5_PRO_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    "usd_per_audio_input_token": 1.25 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0072,
    ##### Agg. Benchmark #####
    "overall": 84.10,
}
LLAMA_4_MAVERICK_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.35 / 1e6,
    "usd_per_output_token": 1.15 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0122,
    ##### Agg. Benchmark #####
    "overall": 79.4,
}
VLLM_QWEN_1_5_0_5B_CHAT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.0 / 1e6,
    "usd_per_output_token": 0.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.1000, # TODO: fill-in with a better estimate
    ##### Agg. Benchmark #####
    "overall": 30.0, # TODO: fill-in with a better estimate
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
    Model.GPT_4_1.value: GPT_4_1_MODEL_CARD,
    Model.GPT_4_1_MINI.value: GPT_4_1_MINI_MODEL_CARD,
    Model.GPT_4_1_NANO.value: GPT_4_1_NANO_MODEL_CARD,
    Model.GPT_5.value: GPT_5_MODEL_CARD,
    Model.GPT_5_MINI.value: GPT_5_MINI_MODEL_CARD,
    Model.GPT_5_NANO.value: GPT_5_NANO_MODEL_CARD,
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
    Model.GOOGLE_GEMINI_2_5_FLASH.value: GEMINI_2_5_FLASH_MODEL_CARD,
    Model.GOOGLE_GEMINI_2_5_FLASH_LITE.value: GEMINI_2_5_FLASH_LITE_MODEL_CARD,
    Model.GOOGLE_GEMINI_2_5_PRO.value: GEMINI_2_5_PRO_MODEL_CARD,
    Model.LLAMA_4_MAVERICK.value: LLAMA_4_MAVERICK_MODEL_CARD,
    Model.VLLM_QWEN_1_5_0_5B_CHAT.value: VLLM_QWEN_1_5_0_5B_CHAT_MODEL_CARD,
}

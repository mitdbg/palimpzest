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
    LLAMA3_2_3B = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    LLAMA3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    LLAMA3_2_90B_V = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    DEEPSEEK_R1_DISTILL_QWEN_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    GPT_4o = "gpt-4o-2024-08-06"
    GPT_4o_MINI = "gpt-4o-mini-2024-07-18"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    CLIP_VIT_B_32 = "clip-ViT-B-32"
    # o1 = "o1-2024-12-17"

    def __repr__(self):
        return f"{self.name}"

    def is_deepseek_model(self):
        return "deepseek" in self.value.lower()

    def is_llama_model(self):
        return "llama" in self.value.lower()

    def is_mixtral_model(self):
        return "mixtral" in self.value.lower()

    def is_clip_model(self):
        return "clip" in self.value.lower()

    def is_together_model(self):
        is_llama_model = self.is_llama_model()
        is_mixtral_model = self.is_mixtral_model()
        is_deepseek_model = self.is_deepseek_model()
        is_clip_model = self.is_clip_model()
        return is_llama_model or is_mixtral_model or is_deepseek_model or is_clip_model

    def is_gpt_4o_model(self):
        return "gpt-4o" in self.value.lower()

    def is_o1_model(self):
        return "o1" in self.value.lower()

    def is_text_embedding_model(self):
        return "text-embedding" in self.value.lower()

    def is_openai_model(self):
        is_gpt4_model = self.is_gpt_4o_model()
        is_o1_model = self.is_o1_model()
        is_text_embedding_model = self.is_text_embedding_model()
        return is_gpt4_model or is_o1_model or is_text_embedding_model

    def is_vision_model(self):
        vision_models = [
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "o1-2024-12-17",
        ]
        return self.value in vision_models

    def is_embedding_model(self):
        is_clip_model = self.is_clip_model()
        is_text_embedding_model = self.is_text_embedding_model()
        return is_clip_model or is_text_embedding_model

class APIClient(str, Enum):
    """
    APIClient describes the API client to be used when invoking an LLM.
    """

    OPENAI = "openai"
    TOGETHER = "together"

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
    # COT_BOOL_IMAGE_CRITIC = "chain-of-thought-bool-image-critic"
    # COT_BOOL_IMAGE_REFINE = "chain-of-thought-bool-image-refine"

    # Chain-of-Thought Question Answering Prompt Strategies
    COT_QA = "chain-of-thought-question"
    COT_QA_CRITIC = "chain-of-thought-question-critic"
    COT_QA_REFINE = "chain-of-thought-question-refine"

    # Chain-of-Thought Question with Image Prompt Strategies
    COT_QA_IMAGE = "chain-of-thought-question-image"
    COT_QA_IMAGE_CRITIC = "chain-of-thought-question-critic-image"
    COT_QA_IMAGE_REFINE = "chain-of-thought-question-refine-image"

    # Mixture-of-Agents Prompt Strategies
    COT_MOA_PROPOSER = "chain-of-thought-mixture-of-agents-proposer"
    COT_MOA_PROPOSER_IMAGE = "chain-of-thought-mixture-of-agents-proposer-image"
    COT_MOA_AGG = "chain-of-thought-mixture-of-agents-aggregation"

    # Split Convert Prompt Strategies
    SPLIT_PROPOSER = "split-proposer"
    SPLIT_MERGER = "split-merger"

    def is_image_prompt(self):
        return "image" in self.value

    def is_bool_prompt(self):
        return "bool" in self.value

    def is_convert_prompt(self):
        return "bool" not in self.value

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

# Rough estimate of the number of tokens the context is allowed to take up for MIXTRAL and LLAMA3 models
MIXTRAL_LLAMA_CONTEXT_TOKENS_LIMIT = 6000

# a naive estimate for the input record size
NAIVE_EST_SOURCE_RECORD_SIZE_IN_BYTES = 1_000_000

# a naive estimate for filter selectivity
NAIVE_EST_FILTER_SELECTIVITY = 0.5

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
    ##### Code #####
    "code": 0.0,
}
LLAMA3_1_8B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1e6,
    "usd_per_output_token": 0.18 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0059,
    ##### Agg. Benchmark #####
    "overall": 44.25,
    ##### Code #####
    "code": 72.6,
}
LLAMA3_3_70B_INSTRUCT_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.88 / 1e6,
    "usd_per_output_token": 0.88 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0139,
    ##### Agg. Benchmark #####
    "overall": 65.92,
    ##### Code #####
    "code": 88.4,
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
MIXTRAL_8X_7B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.6 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0112,
    ##### Agg. Benchmark #####
    "overall": 43.27,
    ##### Code #####
    "code": 40.0,
}
DEEPSEEK_V3_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1.25 / 1E6,
    "usd_per_output_token": 1.25 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0769,
    ##### Agg. Benchmark #####
    "overall": 75.87,
    ##### Code #####
    "code": 92.0,
}
DEEPSEEK_R1_DISTILL_QWEN_1_5B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.18 / 1E6,
    "usd_per_output_token": 0.18 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0026,
    ##### Agg. Benchmark #####
    "overall": 39.90, # https://www.reddit.com/r/LocalLLaMA/comments/1iserf9/deepseek_r1_distilled_models_mmlu_pro_benchmarks/
    ##### Code #####
    "code": 0.0,
}
GPT_4o_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 2.5 / 1e6,
    "usd_per_output_token": 10.0 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0079,
    ##### Agg. Benchmark #####
    "overall": 74.68,
    ##### Code #####
    "code": 90.0,
}
GPT_4o_MINI_MODEL_CARD = {
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 0.15 / 1e6,
    "usd_per_output_token": 0.6 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0098,
    ##### Agg. Benchmark #####
    "overall": 63.09,
    ##### Code #####
    "code": 86.0,
}
o1_MODEL_CARD = {  # noqa: N816
    # NOTE: it is unclear if the same ($ / token) costs can be applied for vision, or if we have to calculate this ourselves
    ##### Cost in USD #####
    "usd_per_input_token": 15 / 1e6,
    "usd_per_output_token": 60 / 1e6,
    ##### Time #####
    "seconds_per_output_token": 0.0110,
    ##### Agg. Benchmark #####
    "overall": 89.30,
    ##### Code #####
    "code": 92.3, # NOTE: just copying MMLU score for now
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


MODEL_CARDS = {
    Model.LLAMA3_2_3B.value: LLAMA3_2_3B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_1_8B.value: LLAMA3_1_8B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_3_70B.value: LLAMA3_3_70B_INSTRUCT_MODEL_CARD,
    Model.LLAMA3_2_90B_V.value: LLAMA3_2_90B_V_MODEL_CARD,
    Model.DEEPSEEK_V3.value: DEEPSEEK_V3_MODEL_CARD,
    Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B.value: DEEPSEEK_R1_DISTILL_QWEN_1_5B_MODEL_CARD,
    Model.MIXTRAL.value: MIXTRAL_8X_7B_MODEL_CARD,
    Model.GPT_4o.value: GPT_4o_MODEL_CARD,
    Model.GPT_4o_MINI.value: GPT_4o_MINI_MODEL_CARD,
    # Model.o1.value: o1_MODEL_CARD,
    Model.TEXT_EMBEDDING_3_SMALL.value: TEXT_EMBEDDING_3_SMALL_MODEL_CARD,
    Model.CLIP_VIT_B_32.value: CLIP_VIT_B_32_MODEL_CARD,
}


###### DEPRECATED ######
# # NOTE: seconds_per_output_token is based on `gpt-3.5-turbo-1106`
# GPT_3_5_MODEL_CARD = {
#     ##### Cost in USD #####
#     "usd_per_input_token": 0.5 / 1E6,
#     "usd_per_output_token": 1.5 / 1E6,
#     ##### Time #####
#     "seconds_per_output_token": 0.0065,
#     ##### Agg. Benchmark #####
#     "overall": 70.0, # 5-shot
#     ##### Commonsense Reasoning #####
#     "reasoning": 84.1,
#     ### "HellaSwag": 85.5,^  # 10-shot
#     ### "WinoGrande": 81.6,^ # 5-shot
#     ### "Arc-e": 85.2,^      # 25-shot
#     ##### World Knowledge #####
#     ##### Reading Comprehension #####
#     ### "DROP": 64.1, # 3-shot
#     ##### Code #####
#     "code": 48.1,
#     ### "HumanEval": 48.1,^ # 0-shot
#     ##### Math #####
#     "math": 57.1,
#     ### "GSM8K": 57.1,^     # 5-shot
# }
# # NOTE: the seconds_per_output_token was computed based on a slightly different model ('gpt-4-1106-preview')
# #       and the benchmark statistics were computed based on the GPT-4 Technical Report; these might be
# #       slightly innacurate compared to the real numbers for gpt-4-0125-preview, but we'll use them until
# #       we have something better. (The cost metrics are accurate).
# GPT_4_MODEL_CARD = {
#     ##### Cost in USD #####
#     "usd_per_input_token": 10 / 1E6,
#     "usd_per_output_token": 30 / 1E6,
#     ##### Time #####
#     "seconds_per_output_token": 0.018,
#     ##### Agg. Benchmark #####
#     "overall": 86.4, # 5-shot
#     ##### Commonsense Reasoning #####
#     "reasoning": 93.0,
#     ### "HellaSwag": 95.3,^  # 10-shot
#     ### "WinoGrande": 87.5,^ # 5-shot
#     ### "Arc-e": 96.3,^      # 25-shot
#     ##### World Knowledge #####
#     ##### Reading Comprehension #####
#     ### "DROP": 80.9, # 3-shot
#     ##### Code #####
#     "code": 67.0,
#     ### "HumanEval": 67.0,^ # 0-shot
#     ##### Math #####
#     "math": 92.0,
#     ### "GSM8K": 92.0,^     # 5-shot
# }

# # TODO: use cost info in here: https://platform.openai.com/docs/guides/vision/calculating-costs
# GPT_4V_MODEL_CARD = {
#     ##### Cost in USD #####
#     "usd_per_input_token": 10 / 1E6,
#     "usd_per_output_token": 30 / 1E6,
#     ##### Time #####
#     "seconds_per_output_token": 0.042 / 10.0, # TODO: / 10.0 is a hack; need to figure out why time estimates are so off
#     ##### Agg. Benchmark #####
#     "overall": 86.4,
# }


# GEMINI_1_MODEL_CARD = {
#     ##### Cost in USD #####
#     "usd_per_input_token": 125 / 1E8, # Gemini is free but rate limited for now. Pricing will be updated
#     "usd_per_output_token": 375 / 1E9,
#     ##### Time #####
#     "seconds_per_output_token": 0.042 / 10.0, # TODO:
#     ##### Agg. Benchmark #####
#     "overall": 65.0, # 90.0 TODO: we are using the free version of Gemini which is substantially worse than its paid version; I'm manually revising it's quality below that of Mixtral
#     ##### Commonsense Reasoning #####
#     "reasoning": 80.0, # 87.8, TODO: see note above on overall
#     # "HellaSwag": 87.8,  # 10-shot
#     ##### World Knowledge #####
#     ##### Reading Comprehension #####
#     # "DROP": 82.4, # Variable shots ?
#     ##### Code #####
#     "code": 74.4,
#     # "HumanEval": 74.4, # 0-shot (IT)*
#     # "Natural2Code": 74.9, # 0-shot
#     ##### Math #####
#     "math": 94.4,
#     # "GSM8K": 94.4,     # maj1@32
#     # "MATH": 53.2,      # 4-shot
# }

# GEMINI_1V_MODEL_CARD = {
#     ##### Cost in USD #####
#     "usd_per_input_token": 25 / 1E6,  # Gemini is free but rate limited for now. Pricing will be updated
#     "usd_per_output_token": 375 / 1E9,
#     ##### Time #####
#     "seconds_per_output_token": 0.042, # / 10.0, # TODO:
#     ##### Agg. Benchmark #####
#     "overall": 65.0, # 90.0, TODO: see note above in Gemini_1 model card
#     ##### Commonsense Reasoning #####
#     "reasoning": 80.0, # 87.8, TODO: see note above in Gemini_1 model card
#     # "HellaSwag": 87.8,  # 10-shot
#     ##### World Knowledge #####
#     ##### Reading Comprehension #####
#     # "DROP": 82.4, # Variable shots ?
#     ##### Code #####
#     "code": 74.4,
#     # "HumanEval": 74.4, # 0-shot (IT)*
#     # "Natural2Code": 74.9, # 0-shot
#     ##### Math #####
#     "math": 94.4,
#     # "GSM8K": 94.4,     # maj1@32
#     # "MATH": 53.2,      # 4-shot
# }

### This file contains constants used by Palimpzest ###
from __future__ import annotations

import os
from enum import Enum

from palimpzest.utils.model_info_helpers import ModelMetricsManager


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

# assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# assume 30 GB/sec for sequential access of memory
MEMORY_SCAN_TIME_PER_KB = 1 / (float(30) * 1024 * 1024)

# assume 1 KB per record
NAIVE_BYTES_PER_RECORD = 1024

# rough conversion from # of characters --> # of tokens; assumes 1 token ~= 4 chars
TOKENS_PER_CHARACTER = 0.25

# rough estimate of the number of tokens the context is allowed to take up for LLAMA3 models
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

# whether or not to log LLM outputs
LOG_LLM_OUTPUT = False

# maximum number of models to use when user does not narrow optimization space
MAX_AVAILABLE_MODELS = 5

class Model:
    """
    Model describes the underlying LLM which should be used to perform some operation
    which requires invoking an LLM.
    """
    # Registry of known models (maps value string to Model instance)
    _registry: dict[str, Model] = {}

    def __init__(self, model_id: str, local_model_url: str | None = None):
        self.metrics_manager = ModelMetricsManager()
        self.model_id = model_id
        self.model_specs = self.metrics_manager.get_model_metrics(model_id)
        if not self.model_specs:
            raise ValueError("Palimpzest currently does not contain information about this model.")
        Model._registry[model_id] = self

    def __lt__(self, other):
        if isinstance(other, Model):
            return self.value < other.value
        if isinstance(other, str):
            return self.value < other
        return NotImplemented

    @classmethod
    def get_all_models(cls) -> list[Model]:
        return list(cls._registry.values())

    @property
    def value(self) -> str:
        return self.model_id

    @property
    def provider(self) -> str | None:
        """Returns the provider string for this model."""
        return self.model_specs.get("provider")

    @property
    def api_key_env_var(self) -> str | None:
        """
        Returns the standard environment variable name for this provider's API key.
        """
        if self.provider == "gemini":
            return "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        mapping = {
            "openai": "OPENAI_API_KEY",
            "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
            "anthropic": "ANTHROPIC_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "hosted_vllm": "VLLM_API_KEY"
        }
        return mapping.get(self.provider)

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Model):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def is_llama_model(self) -> bool:
        return self.model_specs.get("is_llama_model", False)
    
    def is_vllm_model(self) -> bool:
        return self.model_specs.get("is_vllm_model", False)
    
    def is_embedding_model(self) -> bool:
        return self.model_specs.get("is_embedding_model", False)
    
    def is_provider_vertex_ai(self) -> bool:
        return self.provider == "vertex_ai"
    
    def is_provider_anthropic(self) -> bool:
        return self.provider == "anthropic"
    
    def is_provider_google_ai_studio(self) -> bool:
        return self.provider == "gemini"

    def is_provider_openai(self) -> bool:
        return self.provider == "openai"
    
    def is_provider_together_ai(self) -> bool:
        return self.provider == "together_ai"
    
    def is_provider_deepseek(self) -> bool:
        return self.provider == "deepseek"

    def is_o_model(self) -> bool:
        return self.model_specs.get("is_o_model", False)

    def is_gpt_5_model(self) -> bool:
        return self.model_specs.get("is_gpt_5_model", False)

    def is_reasoning_model(self) -> bool:
        return self.model_specs.get("is_reasoning_model", False)

    def is_text_model(self) -> bool:
        return self.model_specs.get("is_text_model", False)

    def is_vision_model(self) -> bool:
        return self.model_specs.get("is_vision_model", False)

    def is_audio_model(self) -> bool:
        return self.model_specs.get("is_audio_model", False)

    def is_text_image_multimodal_model(self) -> bool:
        return self.is_text_model() and self.is_vision_model()

    def is_text_audio_multimodal_model(self) -> bool:
        return self.is_audio_model() and self.is_text_model()

    def supports_prompt_caching(self) -> bool:
        return self.model_specs.get("supports_prompt_caching", False)

    def get_usd_per_input_token(self) -> float:
        return self.model_specs.get("usd_per_input_token", 0.0)
    
    def get_usd_per_output_token(self) -> float:
        return self.model_specs.get("usd_per_output_token", 0.0)

    def get_usd_per_audio_input_token(self) -> float:
        return self.model_specs.get("usd_per_audio_input_token", 0.0)

    def get_usd_per_cache_read_token(self) -> float:
        return self.model_specs.get("usd_per_cache_read_token", self.get_usd_per_cache_read_token())
    
    def get_usd_per_audio_cache_read_token(self) -> float:
        return self.model_specs.get("usd_per_audio_cache_read_token", self.get_usd_per_audio_cache_read_token())
    
    def get_usd_per_cache_creation_token(self) -> float:
        return self.model_specs.get("usd_per_cache_creation_token", 0.0)
    
    def get_usd_per_audio_cache_creation_token(self) -> float:
        return self.model_specs.get("usd_per_audio_cache_creation_token", 0.0)
    
    def get_seconds_per_output_token(self) -> float:
        return self.model_specs.get("seconds_per_output_token", 0.0)

    def get_overall_score(self) -> float:
        return self.model_specs.get("MMLU_Pro_score", 0.0)

Model.LLAMA3_2_3B = Model("together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo")
Model.LLAMA3_1_8B = Model("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
Model.LLAMA3_3_70B = Model("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
Model.LLAMA3_2_90B_V = Model("together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
Model.DEEPSEEK_V3 = Model("together_ai/deepseek-ai/DeepSeek-V3")
Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B = Model("together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
Model.GPT_4o = Model("openai/gpt-4o-2024-08-06")
Model.GPT_4o_MINI = Model("openai/gpt-4o-mini-2024-07-18")
Model.GPT_4_1 = Model("openai/gpt-4.1-2025-04-14")
Model.GPT_4_1_MINI = Model("openai/gpt-4.1-mini-2025-04-14")
Model.GPT_4_1_NANO = Model("openai/gpt-4.1-nano-2025-04-14")
Model.GPT_5 = Model("openai/gpt-5-2025-08-07")
Model.GPT_5_MINI = Model("openai/gpt-5-mini-2025-08-07")
Model.GPT_5_NANO = Model("openai/gpt-5-nano-2025-08-07")
Model.GPT_5_2 = Model("openai/gpt-5.2-2025-12-11")
Model.o4_MINI = Model("openai/o4-mini-2025-04-16")  # noqa: N815
# Model.CLAUDE_3_5_SONNET = Model("anthropic/claude-3-5-sonnet-20241022") - retired 10/28/2025
Model.CLAUDE_3_7_SONNET = Model("anthropic/claude-3-7-sonnet-20250219")
Model.CLAUDE_4_SONNET = Model("anthropic/claude-sonnet-4-20250514")
Model.CLAUDE_4_5_SONNET = Model("anthropic/claude-sonnet-4-5-20250929")
Model.CLAUDE_3_5_HAIKU = Model("anthropic/claude-3-5-haiku-20241022")
Model.CLAUDE_4_5_HAIKU = Model("anthropic/claude-haiku-4-5-20251001")
Model.GEMINI_3_0_PRO = Model("vertex_ai/gemini-3-pro-preview")  # image
Model.GEMINI_3_0_FLASH = Model("vertex_ai/gemini-3-flash-preview")  # Text, Image, Video, Audio, and PDF
Model.GEMINI_2_0_FLASH = Model("vertex_ai/gemini-2.0-flash")
Model.GEMINI_2_5_FLASH = Model("vertex_ai/gemini-2.5-flash")
Model.GEMINI_2_5_PRO = Model("vertex_ai/gemini-2.5-pro")
Model.GOOGLE_GEMINI_3_0_PRO = Model("gemini/gemini-3-pro-preview")
Model.GOOGLE_GEMINI_3_0_FLASH = Model("gemini/gemini-3-flash-preview")
Model.GOOGLE_GEMINI_2_5_FLASH = Model("gemini/gemini-2.5-flash")
Model.GOOGLE_GEMINI_2_5_FLASH_LITE = Model("gemini/gemini-2.5-flash-lite")
Model.GOOGLE_GEMINI_2_5_PRO = Model("gemini/gemini-2.5-pro")
Model.LLAMA_4_MAVERICK = Model("vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas")
Model.GPT_4o_AUDIO_PREVIEW = Model("openai/gpt-4o-audio-preview")
Model.GPT_4o_MINI_AUDIO_PREVIEW = Model("openai/gpt-4o-mini-audio-preview")
Model.VLLM_QWEN_1_5_0_5B_CHAT = Model("hosted_vllm/qwen/Qwen1.5-0.5B-Chat")
Model.TEXT_EMBEDDING_3_SMALL = Model("text-embedding-3-small")
Model.CLIP_VIT_B_32 = Model("clip-ViT-B-32")

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
# Model metrics now fetched from singular json file curated_model_info.json

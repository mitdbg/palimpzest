import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

PZ_MODEL_DATA_URL = "https://palimpzest-research.s3.us-east-1.amazonaws.com/pz_models_information.json"

# Known MMLU-Pro scores (manually curated)
MMLU_PRO_SCORES = {
    # OpenAI
    "gpt-4o": 74.1,
    "gpt-4o-mini": 62.7,
    "gpt-4-turbo": 70.6,
    "gpt-4": 64.8,
    "gpt-3.5-turbo": 49.2,
    "o1-preview": 80.3,
    "o1-mini": 80.0,
    "o3-mini": 79.6,
    "o4-mini": 80.6,
    "gpt-4.1": 80.5,
    "gpt-4.1-mini": 77.2,
    "gpt-4.1-nano": 62.3,
    "gpt-5": 87.0,
    "gpt-5-mini": 82.5,
    "gpt-5-nano": 77.9,
    "gpt-5.2": 86.23,
    # Anthropic
    "claude-3-5-sonnet": 78.4,
    "claude-3-7-sonnet": 80.7,
    "claude-3-opus": 72.6,
    "claude-3-sonnet": 68.5,
    "claude-3-haiku": 55.7,
    "claude-3-5-haiku": 64.1,
    "claude-sonnet-4": 83.87,
    "claude-sonnet-4-5": 87.36,
    "claude-haiku-4-5": 78.72,
    "claude-opus-4-5": 87.3,
    # Google
    "gemini-1.5-pro": 75.8,
    "gemini-1.5-flash": 67.5,
    "gemini-2.0-flash": 77.4,
    "gemini-2.5-flash": 80.75,
    "gemini-2.5-flash-lite": 79.1,
    "gemini-2.5-pro": 84.1,
    "gemini-3-flash": 87.63,
    "gemini-3-pro": 90.1,
    # Meta Llama
    "llama-3-8b": 44.25,
    "llama-3-70b": 55.0,
    "llama-3.1-8b": 44.25,
    "llama-3.1-70b": 55.0,
    "llama-3.1-405b": 73.3,
    "llama-3.2-3b": 36.5,
    "llama-3.2-90b": 65.0,
    "llama-3.3-70b": 69.9,
    "llama-4-maverick": 79.4,
    # Mistral
    "mistral-large": 65.0,
    "mistral-medium": 55.0,
    "mistral-small": 50.0,
    "mixtral-8x7b": 49.0,
    "mixtral-8x22b": 58.0,
    # DeepSeek
    "deepseek-v3": 73.8,
    "deepseek-r1": 85.0,
    "deepseek-r1-distill-qwen-1.5b": 39.9,
    "deepseek-r1-distill-qwen-7b": 52.0,
    "deepseek-r1-distill-llama-70b": 72.0,
    # Qwen
    "qwen-2-72b": 55.0,
    "qwen-2.5-72b": 71.1,
    "qwen-2.5-32b": 63.0,
}

# Known latency data (tokens per second)
LATENCY_TPS_DATA = {
    # OpenAI
    "gpt-4o": 125.0,
    "gpt-4o-mini": 63.0,
    "gpt-4-turbo": 35.0,
    "o1-preview": 15.0,
    "o1-mini": 65.0,
    "gpt-4.1": 132.0,
    "gpt-4.1-mini": 62.0,
    "gpt-4.1-nano": 167.0,
    # Anthropic
    "claude-3-5-sonnet": 65.0,
    "claude-3-opus": 25.0,
    "claude-3-sonnet": 60.0,
    "claude-3-haiku": 110.0,
    "claude-3-5-haiku": 53.0,
    "claude-sonnet-4": 71.3,
    "claude-sonnet-4-5": 78.6,
    "claude-haiku-4-5": 118.3,
    # Google
    "gemini-1.5-pro": 70.0,
    "gemini-1.5-flash": 150.0,
    "gemini-2.0-flash": 185.0,
    "gemini-2.5-flash": 227.0,
    "gemini-2.5-pro": 139.0,
    "gemini-3-flash": 219.0,
    "gemini-3-pro": 132.0,
    # Meta Llama
    "llama-3-8b": 200.0,
    "llama-3-70b": 80.0,
    "llama-3.1-8b": 200.0,
    "llama-3.1-70b": 82.0,
    "llama-3.2-3b": 127.0,
    "llama-3.3-70b": 82.0,
    # DeepSeek
    "deepseek-v3": 88.0,
    "deepseek-r1": 50.0,
}

# Default values for when no match is found
DEFAULT_QUALITY_SCORE = 40.0
DEFAULT_SECONDS_PER_OUTPUT_TOKEN = 0.01


def fuzzy_match_score(model_id: str, scores_dict: dict[str, float]) -> float | None:
    """
    Fuzzy match a model ID against a dictionary of scores.
    Tries exact substring matching first, then normalized matching.
    """
    model_lower = model_id.lower()
    model_name = model_lower.split("/")[-1] if "/" in model_lower else model_lower

    for key, score in scores_dict.items():
        if key.lower() in model_name or model_name in key.lower():
            return score

    for key, score in scores_dict.items():
        key_normalized = key.lower().replace("-", "").replace("_", "").replace(".", "")
        model_normalized = model_name.replace("-", "").replace("_", "").replace(".", "")
        if key_normalized in model_normalized or model_normalized in key_normalized:
            return score
    return None


def derive_model_flags(model_id: str) -> dict[str, bool]:
    """
    Derive boolean model flags from the model ID string.
    E.g. is_llama_model, is_gpt_5_model, is_o_model, is_vllm_model, is_clip_model.
    """
    model_lower = model_id.lower()
    flags = {}

    if "llama" in model_lower:
        flags["is_llama_model"] = True
    if "gpt-5" in model_lower or "gpt5" in model_lower:
        flags["is_gpt_5_model"] = True

    model_name = model_lower.split("/")[-1] if "/" in model_lower else model_lower
    if model_name.startswith(("o1", "o3", "o4")) and not model_name.startswith("openai"):
        flags["is_o_model"] = True

    if "clip" in model_lower:
        flags["is_clip_model"] = True

    return flags


def predict_local_model_metrics(model_id: str) -> dict[str, Any]:
    """
    Predict quality and latency metrics for local/vLLM models using fuzzy matching.
    Returns a dict with MMLU_Pro_score and seconds_per_output_token.
    """
    # Try to fuzzy match quality score
    quality_score = fuzzy_match_score(model_id, MMLU_PRO_SCORES)
    if quality_score is None:
        quality_score = DEFAULT_QUALITY_SCORE

    # Try to fuzzy match latency (tokens per second)
    tps = fuzzy_match_score(model_id, LATENCY_TPS_DATA)
    seconds_per_output_token = round(1.0 / tps, 6) if tps is not None else DEFAULT_SECONDS_PER_OUTPUT_TOKEN

    return {
        "MMLU_Pro_score": quality_score,
        "seconds_per_output_token": seconds_per_output_token,
    }


class ModelMetricsManager:
    """
    Manages fetching and caching of model metrics from an external source.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.data_url = PZ_MODEL_DATA_URL
        self._metrics_cache = None
        self._initialized = True

    def _load_data(self):
        if self._metrics_cache is None:
            logger.info(f"Fetching data from URL: {self.data_url}")
            try:
                self._metrics_cache = requests.get(self.data_url).json()
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                self._metrics_cache = {}

    def get_model_metrics(self, model_name) -> dict[str, Any]:
        self._load_data()
        return self._metrics_cache.get(model_name, {})

    def refresh_data(self) -> None:
        self._metrics_cache = None
        self._load_data()

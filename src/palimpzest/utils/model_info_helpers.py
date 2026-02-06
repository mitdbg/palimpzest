import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

PZ_MODEL_DATA_URL = "https://palimpzest-research.s3.us-east-1.amazonaws.com/pz_models_information.json"

# Known MMLU-Pro scores (manually curated)
# Keys should be canonical patterns that fuzzy matching will find
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
    # Meta Llama (include version-specific entries)
    "llama-3-8b": 44.25,
    "llama-3-70b": 55.0,
    "llama-3.1-8b": 44.25,
    "llama-3.1-70b": 55.0,
    "llama-3.1-405b": 73.3,
    "llama-3.2-1b": 24.0,
    "llama-3.2-3b": 36.5,
    "llama-3.2-11b": 48.0,  # vision model
    "llama-3.2-90b": 65.0,  # vision model
    "llama-3.3-70b": 69.9,
    "llama-4-maverick": 79.4,
    "llama-4-scout": 75.0,
    # Mistral
    "mistral-large": 65.0,
    "mistral-medium": 55.0,
    "mistral-small": 50.0,
    "mistral-7b": 45.0,
    "mistral-nemo": 55.0,
    "mixtral-8x7b": 49.0,
    "mixtral-8x22b": 58.0,
    # DeepSeek
    "deepseek-v3": 73.8,
    "deepseek-v2": 65.0,
    "deepseek-coder": 55.0,
    "deepseek-r1": 85.0,
    "deepseek-r1-distill-qwen-1.5b": 39.9,
    "deepseek-r1-distill-qwen-7b": 52.0,
    "deepseek-r1-distill-qwen-32b": 65.0,
    "deepseek-r1-distill-llama-8b": 50.0,
    "deepseek-r1-distill-llama-70b": 72.0,
    # Qwen
    "qwen-2-0.5b": 25.0,
    "qwen-2-1.5b": 30.0,
    "qwen-2-7b": 45.0,
    "qwen-2-72b": 55.0,
    "qwen-2.5-0.5b": 28.0,
    "qwen-2.5-1.5b": 33.0,
    "qwen-2.5-3b": 38.0,
    "qwen-2.5-7b": 48.0,
    "qwen-2.5-14b": 55.0,
    "qwen-2.5-32b": 63.0,
    "qwen-2.5-72b": 71.1,
    "qwen-2.5-coder": 52.0,
    "qwen-vl": 50.0,
    # Phi
    "phi-2": 35.0,
    "phi-3-mini": 45.0,
    "phi-3-small": 50.0,
    "phi-3-medium": 55.0,
    "phi-3.5-mini": 48.0,
    "phi-4": 60.0,
    # Yi
    "yi-1.5-6b": 42.0,
    "yi-1.5-9b": 48.0,
    "yi-1.5-34b": 58.0,
    "yi-34b": 55.0,
    # Gemma
    "gemma-2b": 30.0,
    "gemma-7b": 42.0,
    "gemma-2-2b": 35.0,
    "gemma-2-9b": 50.0,
    "gemma-2-27b": 60.0,
    # InternLM
    "internlm2-7b": 45.0,
    "internlm2-20b": 55.0,
    # Command-R
    "command-r": 55.0,
    "command-r-plus": 65.0,
}

# Known latency data (tokens per second) - used for cloud APIs
# For local models, we'll estimate based on model size
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
    # Meta Llama (cloud-hosted speeds)
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

# Model size to estimated TPS mapping for local inference (conservative estimates)
# These are rough estimates assuming a single GPU setup
LOCAL_MODEL_SIZE_TO_TPS = {
    "0.5b": 300.0,
    "1b": 250.0,
    "1.5b": 220.0,
    "2b": 200.0,
    "3b": 150.0,
    "7b": 80.0,
    "8b": 75.0,
    "9b": 70.0,
    "11b": 60.0,
    "13b": 50.0,
    "14b": 45.0,
    "20b": 35.0,
    "27b": 30.0,
    "32b": 25.0,
    "34b": 23.0,
    "70b": 12.0,
    "72b": 11.0,
    "90b": 8.0,
    "405b": 3.0,
}

# Default values for when no match is found
DEFAULT_QUALITY_SCORE = 40.0
DEFAULT_SECONDS_PER_OUTPUT_TOKEN = 0.02  # Conservative default (~50 TPS)


def _normalize_model_name(name: str) -> str:
    """Normalize a model name for comparison by removing separators and lowercasing."""
    return name.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def _extract_version_info(name: str) -> tuple[str, str | None, str | None]:
    """
    Extract base model name, version, and size from a model name.
    Returns (base_name, version, size) where version/size may be None.

    Examples:
        "Llama-3.1-8B-Instruct" -> ("llama", "3.1", "8b")
        "Qwen2.5-7B-Instruct" -> ("qwen", "2.5", "7b")
        "deepseek-r1-distill-qwen-7b" -> ("deepseek-r1-distill-qwen", None, "7b")
    """
    name_lower = name.lower()

    # Extract size (look for patterns like 7b, 70b, 0.5b, etc.)
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*b(?:illion)?(?:\b|$|-|_)', name_lower)
    size = size_match.group(1) + "b" if size_match else None

    # Extract version (look for patterns like 3.1, 2.5, v3, etc.)
    version_match = re.search(r'[-_]?(\d+(?:\.\d+)?)\s*[-_]', name_lower)
    version = version_match.group(1) if version_match else None

    # Extract base name (first recognizable model family)
    base_patterns = [
        r'(llama)', r'(qwen)', r'(mistral)', r'(mixtral)', r'(gemma)',
        r'(phi)', r'(yi)', r'(deepseek)', r'(internlm)', r'(command)',
        r'(claude)', r'(gpt)', r'(gemini)', r'(o\d)', r'(falcon)',
    ]
    base_name = None
    for pattern in base_patterns:
        match = re.search(pattern, name_lower)
        if match:
            base_name = match.group(1)
            break

    return (base_name or name_lower, version, size)


def fuzzy_match_score(model_id: str, scores_dict: dict[str, float]) -> float | None:
    """
    Fuzzy match a model ID against a dictionary of scores.

    Matching strategy (in order of priority):
    1. Exact key match
    2. Key matches the model name portion (after last /)
    3. Normalized substring matching, preferring longer (more specific) keys
    4. Model family + size matching as fallback

    Prefers the longest (most specific) matching key to avoid e.g.
    "llama-3-8b" matching before "llama-3.1-8b".
    """
    model_lower = model_id.lower()
    # Extract just the model name (after provider prefix)
    model_name = model_lower.split("/")[-1] if "/" in model_lower else model_lower
    model_normalized = _normalize_model_name(model_name)

    # Pass 1: Check for exact key match
    for key, score in scores_dict.items():
        if key.lower() == model_name or key.lower() == model_lower:
            return score

    best_match = None
    best_score = 0  # Track specificity score (higher = better match)

    # Pass 2: Substring matching with specificity scoring
    for key, score in scores_dict.items():
        key_lower = key.lower()
        key_normalized = _normalize_model_name(key)

        # Check substring match
        if key_lower in model_name or key_normalized in model_normalized:
            # Score based on: length of match + bonus for version/size alignment
            specificity = len(key_normalized)

            # Bonus for matching version numbers
            key_base, key_ver, key_size = _extract_version_info(key)
            model_base, model_ver, model_size = _extract_version_info(model_name)

            if key_ver and model_ver and key_ver == model_ver:
                specificity += 10  # Version match bonus
            if key_size and model_size and key_size == model_size:
                specificity += 15  # Size match bonus

            if specificity > best_score:
                best_match = score
                best_score = specificity

    if best_match is not None:
        return best_match

    # Pass 3: Try matching by model family + size
    model_base, model_ver, model_size = _extract_version_info(model_name)
    if model_base:
        for key, score in scores_dict.items():
            key_base, key_ver, key_size = _extract_version_info(key)
            if (key_base and model_base in key_base or key_base in model_base) \
                and (key_size and model_size and key_size == model_size):
                    return score

    return best_match


def _extract_model_size(model_id: str) -> str | None:
    """
    Extract model size from model ID (e.g., "7b", "70b", "0.5b").
    Returns the size string or None if not found.
    """
    model_lower = model_id.lower()
    # Match patterns like: 7b, 70b, 0.5b, 1.5b, 8b-instruct, etc.
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*b(?:illion)?(?:\b|$|[-_])', model_lower)
    if size_match:
        return size_match.group(1) + "b"
    return None


def derive_model_flags(model_id: str) -> dict[str, bool]:
    """
    Derive boolean model flags from the model ID string.
    Detects: is_llama_model, is_gpt_5_model, is_o_model, is_clip_model,
             is_vision_model (from name patterns), is_reasoning_model,
             is_embedding_model, is_text_model
    """
    model_lower = model_id.lower()
    model_name = model_lower.split("/")[-1] if "/" in model_lower else model_lower
    flags = {}

    # Model family detection
    if "llama" in model_lower:
        flags["is_llama_model"] = True
    if "gpt-5" in model_lower or "gpt5" in model_lower:
        flags["is_gpt_5_model"] = True

    if model_name.startswith(("o1", "o3", "o4")) and not model_name.startswith("openai"):
        flags["is_o_model"] = True

    if "clip" in model_lower:
        flags["is_clip_model"] = True

    # Vision/multimodal detection from model name patterns
    vision_patterns = [
        "-vision", "-vl", "vl-", "-v-",  # Common suffixes/infixes
        "vision-", "visual",  # Prefix patterns
        "llava", "cogvlm", "qwen-vl", "internvl",  # Known vision model families
        "pixtral", "idefics", "fuyu",  # More vision models
    ]
    if any(pattern in model_lower for pattern in vision_patterns):
        flags["is_vision_model"] = True

    # Also detect vision for specific Llama 3.2 vision models (11B and 90B variants)
    if "llama" in model_lower and "3.2" in model_lower:
        size = _extract_model_size(model_id)
        if size in ("11b", "90b"):
            flags["is_vision_model"] = True

    # Reasoning model detection
    reasoning_patterns = [
        "deepseek-r1", "o1-", "o3-", "o4-",
        "-cot", "chain-of-thought", "reasoning",
    ]
    if any(pattern in model_lower for pattern in reasoning_patterns):
        flags["is_reasoning_model"] = True

    # Embedding model detection
    embedding_patterns = [
        "embed", "e5-", "e5_", "/e5",  # Common embedding model names
        "bge-", "bge_", "/bge",  # BGE models
        "gte-", "gte_", "/gte",  # GTE models
        "nomic-embed", "jina-embed",  # Nomic and Jina
        "sentence-transformer", "sbert",  # Sentence transformers
        "instructor-", "contriever",  # Other embedding models
    ]
    if any(pattern in model_lower for pattern in embedding_patterns):
        flags["is_embedding_model"] = True
        flags["is_text_model"] = False  # Embedding models are not text generation models

    return flags


def _estimate_tps_from_size(model_id: str) -> float | None:
    """
    Estimate tokens per second based on model size.
    Uses LOCAL_MODEL_SIZE_TO_TPS mapping for conservative local inference estimates.
    """
    size = _extract_model_size(model_id)
    if size and size in LOCAL_MODEL_SIZE_TO_TPS:
        return LOCAL_MODEL_SIZE_TO_TPS[size]

    # Try to find closest size match
    if size:
        try:
            size_num = float(size.replace("b", ""))
            # Find closest known size
            closest_size = None
            closest_diff = float("inf")
            for known_size in LOCAL_MODEL_SIZE_TO_TPS:
                known_num = float(known_size.replace("b", ""))
                diff = abs(known_num - size_num)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_size = known_size
            if closest_size:
                return LOCAL_MODEL_SIZE_TO_TPS[closest_size]
        except ValueError:
            pass

    return None


def predict_local_model_metrics(model_id: str) -> dict[str, Any]:
    """
    Predict quality and latency metrics for local/vLLM models.

    Uses the following strategy:
    1. Fuzzy match against known MMLU-Pro scores for quality
    2. For latency:
       a. Try fuzzy matching against known latency data
       b. Fall back to model size-based estimation
       c. Use conservative default if nothing matches

    Returns a dict with MMLU_Pro_score, seconds_per_output_token, and any
    detected capability flags.
    """
    # Try to fuzzy match quality score
    quality_score = fuzzy_match_score(model_id, MMLU_PRO_SCORES)
    if quality_score is None:
        # Try to estimate based on model size as a rough heuristic
        size = _extract_model_size(model_id)
        if size:
            size_num = float(size.replace("b", ""))
            # Rough heuristic: larger models tend to score better
            # This is a very rough estimate for unknown models
            if size_num < 3:
                quality_score = 30.0
            elif size_num < 10:
                quality_score = 45.0
            elif size_num < 35:
                quality_score = 55.0
            elif size_num < 80:
                quality_score = 65.0
            else:
                quality_score = 70.0
        else:
            quality_score = DEFAULT_QUALITY_SCORE

    # Try to fuzzy match latency (tokens per second)
    tps = fuzzy_match_score(model_id, LATENCY_TPS_DATA)

    if tps is None:
        # Fall back to size-based estimation for local models
        tps = _estimate_tps_from_size(model_id)

    seconds_per_output_token = round(1.0 / tps, 6) if tps is not None else DEFAULT_SECONDS_PER_OUTPUT_TOKEN

    # Also derive capability flags from model name
    flags = derive_model_flags(model_id)

    return {
        "MMLU_Pro_score": quality_score,
        "seconds_per_output_token": seconds_per_output_token,
        **flags,  # Include detected flags
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

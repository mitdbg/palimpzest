#!/usr/bin/env python3
"""
Script to automatically update pz_models_information.json with data from external sources.

Data Sources:
- LiteLLM model_prices_and_context_window.json: Cost and capability data (100% accuracy)
- MMLU-Pro leaderboard: Quality scores (fuzzy matching acceptable)
- Artificial Analysis: Latency data (fuzzy matching acceptable)

Usage:
    python scripts/update_model_info.py [--add-model MODEL_ID] [--dry-run]

Examples:
    # Update all existing models
    python scripts/update_model_info.py

    # Add a new model
    python scripts/update_model_info.py --add-model "openai/gpt-4-turbo"

    # Preview changes without writing
    python scripts/update_model_info.py --dry-run
"""

import argparse
import json
import os
import re
from typing import Any

import requests

# Constants
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PZ_MODELS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "palimpzest",
    "utils",
    "pz_models_information.json",
)

# Provider mapping from LiteLLM prefixes to our provider strings
PROVIDER_MAPPING = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "vertex_ai": "vertex_ai",
    "gemini": "gemini",
    "together_ai": "together_ai",
    "together": "together_ai",
    "hosted_vllm": "hosted_vllm",
    "groq": "groq",
    "mistral": "mistral",
    "cohere": "cohere",
    "bedrock": "bedrock",
    "azure": "azure",
    "deepseek": "deepseek",
    "fireworks_ai": "fireworks_ai",
    "xai": "xai",
}

# Known MMLU-Pro scores (manually curated from https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro)
# These are fuzzy matched against model names
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
    # Cohere
    "command-r": 50.0,
    "command-r-plus": 55.0,
}

# Known latency data (tokens per second) from https://artificialanalysis.ai/leaderboards/models
# seconds_per_output_token = 1 / tokens_per_second
LATENCY_DATA = {
    # OpenAI
    "gpt-4o": 125.0,  # ~0.008 sec/token
    "gpt-4o-mini": 63.0,  # ~0.0159 sec/token
    "gpt-4-turbo": 35.0,
    "o1-preview": 15.0,
    "o1-mini": 65.0,
    "gpt-4.1": 132.0,
    "gpt-4.1-mini": 62.0,
    "gpt-4.1-nano": 167.0,
    # Anthropic
    "claude-3-5-sonnet": 65.0,  # ~0.0154 sec/token
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
    # Meta Llama (via Together AI)
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


def fetch_litellm_data() -> dict[str, Any]:
    """Fetch the latest model pricing data from LiteLLM."""
    print(f"Fetching LiteLLM data from {LITELLM_URL}...")
    try:
        response = requests.get(LITELLM_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"  Found {len(data)} models in LiteLLM database")
        return data
    except Exception as e:
        print(f"  Error fetching LiteLLM data: {e}")
        return {}


def load_existing_data() -> dict[str, Any]:
    """Load existing pz_models_information.json."""
    if os.path.exists(PZ_MODELS_PATH):
        with open(PZ_MODELS_PATH) as f:
            return json.load(f)
    return {}


def save_data(data: dict[str, Any]) -> None:
    """Save data to pz_models_information.json."""
    with open(PZ_MODELS_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} models to {PZ_MODELS_PATH}")


def extract_provider(model_id: str) -> str:
    """Extract provider from model ID."""
    if "/" in model_id:
        prefix = model_id.split("/")[0].lower()
        return PROVIDER_MAPPING.get(prefix, prefix)
    # Fallback: try to guess from model name
    model_lower = model_id.lower()
    if "gpt" in model_lower or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "gemini"
    if "llama" in model_lower:
        return "together_ai"
    return "unknown"


def fuzzy_match_score(model_id: str, scores_dict: dict[str, float]) -> float | None:
    """Fuzzy match a model ID to find its score."""
    # Normalize the model ID
    model_lower = model_id.lower()

    # Remove provider prefix
    if "/" in model_lower:
        model_name = model_lower.split("/")[-1]
    else:
        model_name = model_lower

    # Try exact match first
    for key, score in scores_dict.items():
        if key.lower() in model_name or model_name in key.lower():
            return score

    # Try partial matches
    for key, score in scores_dict.items():
        key_normalized = key.lower().replace("-", "").replace("_", "").replace(".", "")
        model_normalized = model_name.replace("-", "").replace("_", "").replace(".", "")
        if key_normalized in model_normalized or model_normalized in key_normalized:
            return score

    return None


def convert_litellm_to_pz_format(model_id: str, litellm_data: dict[str, Any]) -> dict[str, Any]:
    """Convert LiteLLM model data to PZ format."""
    # Extract mode to determine model type
    mode = litellm_data.get("mode", "chat")
    is_text_model = mode in ["chat", "completion"]
    is_embedding_model = mode == "embedding"

    pz_entry = {
        # Cost data (100% accuracy from LiteLLM)
        "usd_per_input_token": litellm_data.get("input_cost_per_token"),
        "usd_per_output_token": litellm_data.get("output_cost_per_token"),
        "usd_per_cache_read_token": litellm_data.get("cache_read_input_token_cost", 0),
        "usd_per_audio_input_token": litellm_data.get("input_cost_per_audio_token", 0),

        # Capability data (100% accuracy from LiteLLM)
        "is_reasoning_model": litellm_data.get("supports_reasoning", False),
        "is_text_model": is_text_model,
        "is_vision_model": litellm_data.get("supports_vision", False),
        "is_audio_model": litellm_data.get("supports_audio_input", False),
        "is_embedding_model": is_embedding_model,
        "supports_prompt_caching": litellm_data.get("supports_prompt_caching", False),

        # Provider
        "provider": extract_provider(model_id),

        # Latency (fuzzy match acceptable)
        "seconds_per_output_token": None,

        # Quality score (fuzzy match acceptable)
        "MMLU_Pro_score": None,

        # Metadata
        "sources": [LITELLM_URL],
    }

    # Try to match MMLU-Pro score
    mmlu_score = fuzzy_match_score(model_id, MMLU_PRO_SCORES)
    if mmlu_score is not None:
        pz_entry["MMLU_Pro_score"] = mmlu_score

    # Try to match latency data
    tps = fuzzy_match_score(model_id, LATENCY_DATA)
    if tps is not None:
        pz_entry["seconds_per_output_token"] = round(1.0 / tps, 6)

    return pz_entry


def update_model(
    model_id: str,
    existing_data: dict[str, Any],
    litellm_data: dict[str, Any],
    force_update: bool = False,
) -> dict[str, Any] | None:
    """Update or create a model entry."""
    # Find LiteLLM entry for this model
    litellm_entry = None

    # Try exact match first
    if model_id in litellm_data:
        litellm_entry = litellm_data[model_id]
    else:
        # Try without provider prefix
        if "/" in model_id:
            model_name = model_id.split("/", 1)[1]
            if model_name in litellm_data:
                litellm_entry = litellm_data[model_name]

    if litellm_entry is None and not force_update:
        print(f"  WARNING: No LiteLLM data found for {model_id}")
        return None

    if litellm_entry:
        new_entry = convert_litellm_to_pz_format(model_id, litellm_entry)
    else:
        # Create minimal entry for models not in LiteLLM
        new_entry = {
            "usd_per_input_token": None,
            "usd_per_output_token": None,
            "seconds_per_output_token": None,
            "MMLU_Pro_score": fuzzy_match_score(model_id, MMLU_PRO_SCORES),
            "is_reasoning_model": False,
            "is_text_model": True,
            "is_vision_model": False,
            "is_audio_model": False,
            "is_embedding_model": False,
            "supports_prompt_caching": False,
            "provider": extract_provider(model_id),
            "sources": None,
            "note": "Model not found in LiteLLM database - costs may need manual entry",
        }

    # If model exists, preserve certain fields that might have been manually set
    if model_id in existing_data:
        existing = existing_data[model_id]

        # Preserve manual overrides for MMLU and latency if they exist
        if existing.get("MMLU_Pro_score") is not None and new_entry.get("MMLU_Pro_score") is None:
            new_entry["MMLU_Pro_score"] = existing["MMLU_Pro_score"]

        if existing.get("seconds_per_output_token") is not None and new_entry.get("seconds_per_output_token") is None:
            new_entry["seconds_per_output_token"] = existing["seconds_per_output_token"]

        # Preserve custom fields like is_llama_model, is_gpt_5_model, etc.
        for key in existing:
            if key.startswith("is_") and key not in new_entry:
                new_entry[key] = existing[key]

        # Preserve notes
        if existing.get("note") and not new_entry.get("note"):
            new_entry["note"] = existing["note"]

        # Preserve sources if we didn't add new ones
        if existing.get("sources") and new_entry.get("sources") == [LITELLM_URL]:
            new_entry["sources"] = existing["sources"]

    return new_entry


def update_all_models(existing_data: dict[str, Any], litellm_data: dict[str, Any]) -> dict[str, Any]:
    """Update all existing models with latest data."""
    updated_data = {}

    for model_id in existing_data:
        print(f"Updating {model_id}...")
        updated = update_model(model_id, existing_data, litellm_data)
        if updated:
            updated_data[model_id] = updated
        else:
            # Keep existing entry if no update available
            updated_data[model_id] = existing_data[model_id]

    return updated_data


def add_new_model(
    model_id: str,
    existing_data: dict[str, Any],
    litellm_data: dict[str, Any],
) -> dict[str, Any]:
    """Add a new model to the database."""
    if model_id in existing_data:
        print(f"Model {model_id} already exists. Use update to modify.")
        return existing_data

    print(f"Adding new model: {model_id}")
    new_entry = update_model(model_id, existing_data, litellm_data, force_update=True)

    if new_entry:
        existing_data[model_id] = new_entry
        print(f"  Added {model_id}")
    else:
        print(f"  Failed to add {model_id}")

    return existing_data


def list_available_models(litellm_data: dict[str, Any], provider_filter: str | None = None) -> None:
    """List available models from LiteLLM that can be added."""
    print("\nAvailable models from LiteLLM:")
    print("-" * 60)

    models_by_provider: dict[str, list[str]] = {}

    for model_id in sorted(litellm_data.keys()):
        provider = extract_provider(model_id)
        if provider_filter and provider != provider_filter:
            continue
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model_id)

    for provider in sorted(models_by_provider.keys()):
        print(f"\n{provider.upper()}:")
        for model in models_by_provider[provider][:20]:  # Limit to 20 per provider
            mode = litellm_data[model].get("mode", "unknown")
            cost = litellm_data[model].get("input_cost_per_token", "N/A")
            print(f"  {model} (mode: {mode}, input_cost: {cost})")
        if len(models_by_provider[provider]) > 20:
            print(f"  ... and {len(models_by_provider[provider]) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Update pz_models_information.json with external data sources"
    )
    parser.add_argument(
        "--add-model",
        type=str,
        help="Add a new model by its LiteLLM model ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to file",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List available models from LiteLLM",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Filter by provider when listing available models",
    )

    args = parser.parse_args()

    # Fetch LiteLLM data
    litellm_data = fetch_litellm_data()
    if not litellm_data:
        print("Failed to fetch LiteLLM data. Exiting.")
        return

    # List available models if requested
    if args.list_available:
        list_available_models(litellm_data, args.provider)
        return

    # Load existing data
    existing_data = load_existing_data()
    print(f"Loaded {len(existing_data)} existing models")

    if args.add_model:
        # Add a new model
        updated_data = add_new_model(args.add_model, existing_data, litellm_data)
    else:
        # Update all existing models
        updated_data = update_all_models(existing_data, litellm_data)

    # Preview or save
    if args.dry_run:
        print("\n--- DRY RUN: Changes that would be made ---")
        print(json.dumps(updated_data, indent=2))
    else:
        save_data(updated_data)
        print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to automatically update pz_models_information.json with data from external sources.

Data Sources:
- LiteLLM proxy /model/info endpoint: Dynamic model info (100% accuracy, prioritized)
- LiteLLM model_prices_and_context_window.json: Cost and capability data (fallback)
- MMLU-Pro leaderboard: Quality scores (fuzzy matching acceptable)
- Artificial Analysis: Latency data (fuzzy matching acceptable)

Usage:
    python scripts/update_model_info.py MODEL_ID [MODEL_ID ...] [--use-endpoint]
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

import requests
import yaml

# Add src to path to import from palimpzest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from palimpzest.utils.model_info_helpers import (
    LATENCY_TPS_DATA,
    MMLU_PRO_SCORES,
    derive_model_flags,
    fuzzy_match_score,
)

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

# API key environment variable mapping
API_KEY_MAPPING = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "gemini": "GEMINI_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "hosted_vllm": "VLLM_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "xai": "XAI_API_KEY",
}

# Field mapping from LiteLLM endpoint to PZ format
FIELD_MAPPING = [
    ("usd_per_input_token", "input_cost_per_token", None),
    ("usd_per_output_token", "output_cost_per_token", None),
    ("usd_per_audio_input_token", "input_cost_per_audio_token", None),
    ("usd_per_audio_output_token", "output_cost_per_audio_token", None),
    ("usd_per_image_output_token", "output_cost_per_image_token", None),
    ("usd_per_cache_read_token", "cache_read_input_token_cost", None),
    ("usd_per_cache_creation_token", "cache_creation_input_token_cost", None),
    ("supports_prompt_caching", "supports_prompt_caching", False),
]

# Boolean capability fields derived from endpoint
CAPABILITY_MAPPING = [
    ("is_vision_model", "supports_vision", False),
    ("is_audio_model", "supports_audio_input", False),
    ("is_reasoning_model", "supports_reasoning", False),
]

# MMLU_PRO_SCORES, LATENCY_TPS_DATA, and fuzzy_match_score are imported from
# palimpzest.utils.model_info_helpers

# Alias for backwards compatibility in this script
LATENCY_DATA = LATENCY_TPS_DATA


# =============================================================================
# LiteLLM Proxy Endpoint Functions
# =============================================================================

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def extract_provider(model_id: str) -> str:
    """Extract provider from model ID."""
    if "/" in model_id:
        prefix = model_id.split("/")[0].lower()
        return PROVIDER_MAPPING.get(prefix, prefix)

    model_lower = model_id.lower()
    
    # OpenAI
    if any(x in model_lower for x in ["gpt", "o1-", "o3-", "dall-e", "whisper"]):
        return "openai"
    
    # Anthropic
    if "claude" in model_lower:
        return "anthropic"
    
    # Google (Vertex AI / Gemini)
    if "gemini" in model_lower or "bison" in model_lower:
        return "vertex_ai"
    
    # Meta / Together / Llama
    if "llama" in model_lower:
        return "together_ai"
    
    # Mistral
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"

    # DeepSeek
    if "deepseek" in model_lower:
        return "deepseek"

    return "unknown"


def get_api_key_env_var(provider: str) -> str | None:
    return API_KEY_MAPPING.get(provider)


def generate_config_yaml(model_ids: list[str]) -> str:
    config_id = 0
    config_filename = f"litellm_config_{config_id}.yaml"
    while not os.path.exists(config_filename):
        config_id += 1

    config_list = []
    for model_id in model_ids:
        provider = extract_provider(model_id)
        env_var_name = get_api_key_env_var(provider)
        api_key_val = f"os.environ/{env_var_name}" if env_var_name else None

        entry = {
            "model_name": model_id,
            "litellm_params": {
                "model": model_id,
                "api_key": api_key_val,
            },
        }
        config_list.append(entry)

    yaml_structure = {"model_list": config_list}
    with open(config_filename, "w") as f:
        yaml.dump(yaml_structure, f, default_flow_style=False, sort_keys=False)

    return config_filename


def fetch_dynamic_model_info(model_ids: list[str]) -> dict[str, Any]:
    if not model_ids:
        return {}

    port = get_free_port()
    proxy_url = f"http://127.0.0.1:{port}"
    config_filename = generate_config_yaml(model_ids)
    server_env = os.environ.copy()
    process = None
    dynamic_model_info = {}

    print(f"Starting LiteLLM proxy on port {port} for {len(model_ids)} models...")

    try:
        process = subprocess.Popen(
            ["litellm", "--config", config_filename, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=server_env,
        )

        server_ready = False
        max_retries = 30
        for i in range(max_retries):
            if process.poll() is not None:
                _, stderr = process.communicate()
                print(f"  LiteLLM process died unexpectedly: {stderr.decode()}")
                break
            try:
                requests.get(f"{proxy_url}/health/readiness", timeout=1)
                server_ready = True
                print(f"  Server ready after {i + 1} attempts")
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                time.sleep(0.5)

        if not server_ready:
            print("  Timeout: LiteLLM server failed to start within the limit.")
            return {}

        try:
            response = requests.get(f"{proxy_url}/model/info", timeout=10)
            response.raise_for_status()
            model_data = response.json()

            if "data" in model_data and len(model_data["data"]) > 0:
                for item in model_data["data"]:
                    model_name = item.get("model_name")
                    model_info = item.get("model_info", {})
                    dynamic_model_info[model_name] = model_info
                    print(f"  Retrieved info for: {model_name}")
            else:
                print("  WARNING: No model data returned from endpoint")
        except Exception as e:
            print(f"  Error fetching model info: {e}")

    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        if os.path.exists(config_filename):
            os.remove(config_filename)

    return dynamic_model_info


# =============================================================================
# Data Fetching Functions
# =============================================================================

def fetch_litellm_data() -> dict[str, Any]:
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
    if os.path.exists(PZ_MODELS_PATH):
        with open(PZ_MODELS_PATH) as f:
            return json.load(f)
    return {}


def save_data(data: dict[str, Any]) -> None:
    with open(PZ_MODELS_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  [System] Successfully saved to {PZ_MODELS_PATH}")


# =============================================================================
# Matching and Conversion Functions
# =============================================================================

# fuzzy_match_score is imported from palimpzest.utils.model_info_helpers


def derive_model_flags_with_provider(model_id: str, provider: str) -> dict[str, bool]:
    """Wrapper around derive_model_flags that also adds provider-specific flags."""
    flags = derive_model_flags(model_id)
    if provider == "hosted_vllm":
        flags["is_vllm_model"] = True
    return flags


# =============================================================================
# Interactive Review Functions
# =============================================================================

def prompt_for_value(field_name: str, current_value: Any, value_type: str = "any") -> Any:
    while True:
        user_input = input(f"    > Enter new value for '{field_name}' (or press Enter to keep current): ").strip()
        if user_input == "":
            return current_value
        
        try:
            if user_input.lower() == "none":
                return None
            if value_type == "float":
                return float(user_input)
            elif value_type == "int":
                return int(user_input)
            elif value_type == "bool":
                return user_input.lower() in ("true", "yes", "1", "y")
            else:
                try:
                    return json.loads(user_input)
                except json.JSONDecodeError:
                    return user_input
        except ValueError as e:
            print(f"    Invalid input: {e}. Try again.")


def review_field(
    field_name: str,
    value: Any,
    from_endpoint: bool,
    interactive: bool = True,
    value_type: str = "any"
) -> tuple[Any, bool]:
    """
    Review a single field.
    Logic:
    1. If from_endpoint is True and value not None -> VERIFIED (return immediately)
    2. If interactive -> Ask User (1. Correct, 2. Incorrect)
    """
    if from_endpoint and value is not None:
        # Verified automatically by endpoint
        return value, False

    if not interactive:
        return value, False

    print(f"\n  [Review] {field_name}: {value}")
    if from_endpoint and value is None:
         print("    (Source: Endpoint returned Null)")
    else:
         print("    (Source: Derived/Static/Fallback)")

    while True:
        choice = input("    1. Yes, information is correct\n    2. No, enter different value\n    Choice [1]: ").strip()
        if choice == "" or choice == "1":
            return value, False
        elif choice == "2":
            new_value = prompt_for_value(field_name, value, value_type)
            return new_value, True
        else:
            print("    Invalid choice.")


def convert_and_review_model(
    model_id: str,
    litellm_static: dict[str, Any] | None,
    litellm_dynamic: dict[str, Any] | None,
    existing_entry: dict[str, Any] | None,
    interactive: bool = True,
) -> dict[str, Any]:
    """
    1. Aggregates all data into a Draft Entry.
    2. Displays the Draft Entry (User can see Current State).
    3. Iterates fields to Verify (Prioritizing endpoint).
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {model_id}")
    print(f"{'='*60}")

    # --- PHASE 1: Build Draft Entry & Source Map ---
    
    endpoint_fields: set[str] = set()
    raw_data: dict[str, Any] = {}

    # 1. Base: Static Data
    if litellm_static:
        raw_data.update(litellm_static)
    
    # 2. Overlay: Dynamic Data (Priority)
    if litellm_dynamic:
        for key, val in litellm_dynamic.items():
            if val is not None:
                raw_data[key] = val
                endpoint_fields.add(key)

    # 3. Construct Candidate dictionary
    candidate = {}
    source_map = {} # Map field -> is_from_endpoint

    # Provider
    prov = raw_data.get("litellm_provider") or extract_provider(model_id)
    candidate["provider"] = prov
    source_map["provider"] = "litellm_provider" in endpoint_fields

    # Costs & Caching
    for pz_field, litellm_field, default in FIELD_MAPPING:
        val = raw_data.get(litellm_field, default)
        candidate[pz_field] = val
        source_map[pz_field] = litellm_field in endpoint_fields

    # Capabilities
    for pz_field, litellm_field, default in CAPABILITY_MAPPING:
        val = raw_data.get(litellm_field, default)
        # Special logic for audio
        if pz_field == "is_audio_model":
            audio_in = raw_data.get("supports_audio_input", False)
            audio_out = raw_data.get("supports_audio_output", False)
            val = audio_in or audio_out
            source_map[pz_field] = ("supports_audio_input" in endpoint_fields or 
                                    "supports_audio_output" in endpoint_fields)
        else:
            source_map[pz_field] = litellm_field in endpoint_fields
        candidate[pz_field] = val

    # Modes
    mode = raw_data.get("mode", "chat")
    mode_src = "mode" in endpoint_fields
    candidate["is_text_model"] = mode in ["chat", "completion"]
    source_map["is_text_model"] = mode_src
    candidate["is_embedding_model"] = mode == "embedding"
    source_map["is_embedding_model"] = mode_src

    # Flags (Always derived, never endpoint)
    flags = derive_model_flags_with_provider(model_id, candidate["provider"])
    for k, v in flags.items():
        candidate[k] = v
        source_map[k] = False

    # Scores / Latency (Fuzzy or Existing)
    mmlu = fuzzy_match_score(model_id, MMLU_PRO_SCORES)
    if mmlu is None and existing_entry:
        mmlu = existing_entry.get("MMLU_Pro_score")
    candidate["MMLU_Pro_score"] = mmlu
    source_map["MMLU_Pro_score"] = False

    tps = fuzzy_match_score(model_id, LATENCY_DATA)
    sec_per_tok = round(1.0 / tps, 6) if tps else None
    if sec_per_tok is None and existing_entry:
        sec_per_tok = existing_entry.get("seconds_per_output_token")
    candidate["seconds_per_output_token"] = sec_per_tok
    source_map["seconds_per_output_token"] = False
    
    # Audio Cache Read (check existing)
    acr = existing_entry.get("usd_per_audio_cache_read_token") if existing_entry else None
    if acr is not None:
        candidate["usd_per_audio_cache_read_token"] = acr
        source_map["usd_per_audio_cache_read_token"] = False

    # Note
    if existing_entry and existing_entry.get("note"):
        candidate["note"] = existing_entry["note"]
        source_map["note"] = False

    # Sources
    src_list = [LITELLM_URL]
    if existing_entry and existing_entry.get("sources"):
        existing_srcs = existing_entry["sources"]
        if isinstance(existing_srcs, list):
            src_list = list(set(src_list + existing_srcs))
        elif existing_srcs:
            src_list = list(set(src_list + [existing_srcs]))
    candidate["sources"] = src_list

    # --- PHASE 2: Display Current State ---
    
    print("\n--- Current State (Draft) ---")
    display_dict = {}
    for k, v in candidate.items():
        if k == "sources":
            continue
        src_label = "ENDPOINT" if source_map.get(k, False) and v is not None else "DERIVED/STATIC"
        display_dict[k] = f"{v}  [{src_label}]"
    
    print(json.dumps(display_dict, indent=2))
    print("-" * 30)

    # --- PHASE 3: Verification Loop ---

    final_entry = {}
    final_entry["sources"] = candidate["sources"]

    # Iterate over specific keys to ensure order and types
    
    # Provider
    final_entry["provider"], _ = review_field(
        "provider", candidate["provider"], source_map["provider"], interactive, "str"
    )

    # All cost/cache fields
    for k in [f[0] for f in FIELD_MAPPING] + ["usd_per_audio_cache_read_token"]:
        if k in candidate:
            vtype = "float" if "usd_" in k else "bool"
            final_entry[k], _ = review_field(
                k, candidate[k], source_map.get(k, False), interactive, vtype
            )

    # Capabilities & Modes
    bool_keys = [f[0] for f in CAPABILITY_MAPPING] + ["is_text_model", "is_embedding_model"] + list(flags.keys())
    for k in bool_keys:
        if k in candidate:
            final_entry[k], _ = review_field(
                k, candidate[k], source_map.get(k, False), interactive, "bool"
            )

    # Stats
    final_entry["MMLU_Pro_score"], _ = review_field(
        "MMLU_Pro_score", candidate["MMLU_Pro_score"], False, interactive, "float"
    )
    final_entry["seconds_per_output_token"], _ = review_field(
        "seconds_per_output_token", candidate["seconds_per_output_token"], False, interactive, "float"
    )

    # Note
    if "note" in candidate:
        final_entry["note"], _ = review_field(
            "note", candidate["note"], False, interactive, "str"
        )

    # Cleanup Nulls
    cleaned_entry = {k: v for k, v in final_entry.items() if v is not None}
    
    return cleaned_entry


def update_model(
    model_id: str,
    existing_data: dict[str, Any],
    litellm_static: dict[str, Any],
    litellm_dynamic: dict[str, Any] | None = None,
    interactive: bool = True,
) -> dict[str, Any] | None:
    static_entry = None
    if model_id in litellm_static:
        static_entry = litellm_static[model_id]
    else:
        if "/" in model_id:
            model_name = model_id.split("/", 1)[1]
            if model_name in litellm_static:
                static_entry = litellm_static[model_name]

    dynamic_entry = litellm_dynamic.get(model_id) if litellm_dynamic else None
    
    if static_entry is None and dynamic_entry is None:
        print(f"\n  WARNING: No LiteLLM data found for {model_id}")
    
    existing_entry = existing_data.get(model_id)

    new_entry = convert_and_review_model(
        model_id,
        static_entry,
        dynamic_entry,
        existing_entry,
        interactive=interactive,
    )
    return new_entry


def process_models(
    model_ids: list[str],
    existing_data: dict[str, Any],
    litellm_static: dict[str, Any],
    use_endpoint: bool = False,
    interactive: bool = True,
    skip_existing: bool = False,
) -> None:
    """
    Process models and (if interactive is True) ask user whether to write each one to file.
    """
    litellm_dynamic = None
    if use_endpoint:
        litellm_dynamic = fetch_dynamic_model_info(model_ids)

    # We work on the existing_data dictionary directly so we can save incrementally
    current_data_state = existing_data.copy()

    for model_id in model_ids:
        # Check if model exists and if we should skip it
        if skip_existing and model_id in current_data_state:
            print(f"\n  [System] Model '{model_id}' already exists in file. Skipping.")
            continue

        new_entry = update_model(
            model_id, current_data_state, litellm_static, litellm_dynamic,
            interactive=interactive
        )

        if new_entry:
            # Display Final Result
            print("\n" + "-"*30)
            print(f"FINAL JSON FOR: {model_id}")
            print(json.dumps(new_entry, indent=2))
            print("-" * 30)

            # Ask user to write to file
            should_save = True
            if interactive:
                confirm = input(f"Write '{model_id}' to json file? [y/N]: ").strip().lower()
                should_save = confirm == 'y'

            if should_save:
                current_data_state[model_id] = new_entry
                save_data(current_data_state)
            else:
                print(f"  [System] Skipped saving {model_id}.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Update pz_models_information.json with external data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_ids", nargs="*", help="Model IDs to update")
    parser.add_argument("--use-endpoint", action="store_true", help="Fetch dynamic info")
    parser.add_argument("--non-interactive", action="store_true", help="Skip review and auto-save")
    parser.add_argument("--update-all", action="store_true", help="Update all existing")

    args = parser.parse_args()

    litellm_static = fetch_litellm_data()
    if not litellm_static:
        return

    existing_data = load_existing_data()

    skip_existing = False
    if args.update_all:
        model_ids = list(existing_data.keys())
    elif args.model_ids:
        model_ids = args.model_ids
        skip_existing = True
    else:
        parser.print_help()
        return

    interactive = not args.non_interactive

    # Run the main processing loop
    process_models(
        model_ids,
        existing_data,
        litellm_static,
        use_endpoint=args.use_endpoint,
        interactive=interactive,
        skip_existing=skip_existing,
    )

    print("\nAll operations complete.")

if __name__ == "__main__":
    main()
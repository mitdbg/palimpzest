import os, re, json
from typing import Dict, Any, Optional
from palimpzest.constants import CuratedModel

# Global cache for benchmark metrics
_KNOWN_MODEL_METRICS = {}

def _load_benchmark_metrics():
    """
    Loads and flattens the model_mmlu_latency.json file into a dictionary
    mapping model slugs to their metrics.
    """
    global _KNOWN_MODEL_METRICS
    
    # Avoid reloading if already populated
    if _KNOWN_MODEL_METRICS:
        return

    # Assume json file is in the same directory as this script
    json_path = os.path.join(os.path.dirname(__file__), 'model_mmlu_latency.json')
    
    if not os.path.exists(json_path):
        # Fallback or warning could go here; for now we just return empty
        print(f"Warning: Benchmark file not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Flatten the nested JSON structure: Provider -> Model -> Metrics
        # Target format: "model-slug": {"tps": 123.4, "mmlu": 88.5}
        for provider, models in data.items():
            for model_id, specs in models.items():
                slug = model_id.lower()
                
                # Parse metrics with safety checks
                tps = specs.get("output_tokens_per_second")
                mmlu = specs.get("MMLU_Pro_score")
                
                _KNOWN_MODEL_METRICS[slug] = {
                    "tps": float(tps) if tps is not None else None,
                    "mmlu": float(mmlu) if mmlu is not None else None
                }
                
    except Exception as e:
        print(f"Error loading benchmark metrics: {e}")

# Load metrics immediately upon module import
_load_benchmark_metrics()


def get_model_provider(model_name: str) -> str:
    """
    Determines the model provider based on the model name string.
    
    Resolution Order:
    1. Explicit 'provider/model' syntax (e.g., 'anthropic/claude-3')
    2. Known model family prefixes (e.g., 'gpt-4' -> openai)
    3. Provider substring markers (e.g., 'vertex_ai')
    """
    if not model_name:
        return "unknown"
    name_clean = model_name.lower().strip()
    # explicit namespace
    if "/" in name_clean:
        return name_clean.split("/", 1)[0]
    # map model family to providers
    family_map = {
        ("gpt-", "o1-", "dall-e", "text-embedding", "whisper"): "openai",
        ("claude",): "anthropic",
        ("gemini", "gemma", "palm"): "gemini",
        ("llama",): "meta", 
        ("mistral", "mixtral"): "mistral",
        ("command",): "cohere",
    }
    for prefixes, provider in family_map.items():
        if name_clean.startswith(prefixes):
            return provider

    # check for specific provider marker
    provider_markers = ["openai", "anthropic", "vertex_ai", "together_ai", "gemini", "hosted_vllm"]

    for marker in provider_markers:
        if marker in name_clean:
            return marker
    
    return "unknown"


def get_api_key_env_var(model_name: str):
    model_provider = get_model_provider(model_name)
    
    # Special handling for Google/Vertex: Check multiple candidates
    if model_provider in ["gemini", "vertex_ai"]:
        # 1. Check if the legacy/specific GEMINI key exists
        if os.getenv("GEMINI_API_KEY"):
            return "os.environ/GEMINI_API_KEY"
        # 2. Check if the standard GOOGLE key exists
        if os.getenv("GOOGLE_API_KEY"):
            return "os.environ/GOOGLE_API_KEY"
        # 3. Default fallback (standardize on GOOGLE_API_KEY)
        return "os.environ/GOOGLE_API_KEY"

    # Standard 1-to-1 mapping for other providers
    provider_to_env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "CO_API_KEY",
        "groq": "GROQ_API_KEY",
        "huggingface": "HF_TOKEN",
        "deepseek": "DEEPSEEK_API_KEY",
        "fireworks": "FIREWORKS_API_KEY"
    }
    
    return provider_to_env_var.get(model_provider)


def _find_closest_benchmark_metric(model_slug: str) -> Optional[Dict[str, float]]:
    """
    Attempts to find MMLU and Latency metrics from _KNOWN_MODEL_METRICS
    using exact matches, substring matches, and sibling model inference.
    """
    # Ensure metrics are loaded
    if not _KNOWN_MODEL_METRICS:
        _load_benchmark_metrics()
        
    slug = model_slug.lower()

    # 1. Exact Match
    if slug in _KNOWN_MODEL_METRICS:
        return _KNOWN_MODEL_METRICS[slug]

    # 2. Date-invariant Match (e.g. gpt-4o-2024-05-13 -> gpt-4o)
    matches = [k for k in _KNOWN_MODEL_METRICS.keys() if k.startswith(slug) or slug in k]
    if matches:
        # Pick the first match; ideally this could be refined to pick the most recent/best match
        best_match = matches[0]
        return _KNOWN_MODEL_METRICS[best_match]

    # 3. Sibling Inference (Base <-> Instruct)
    is_instruct = "instruct" in slug or "chat" in slug
    base_slug = slug.replace("-instruct", "").replace("-chat", "").strip("-")
    
    if is_instruct:
        # User asks for Instruct, we look for Base
        if base_slug in _KNOWN_MODEL_METRICS:
            base_metrics = _KNOWN_MODEL_METRICS[base_slug]
            # Heuristic: Instruct usually +10% MMLU, Speed similar (0.95x)
            return {
                "mmlu": base_metrics["mmlu"] * 1.1 if base_metrics["mmlu"] else None, 
                "tps": (base_metrics["tps"] * 0.95) if base_metrics["tps"] else None
            }
    else:
        # User asks for Base, we look for Instruct
        instruct_slug = f"{slug}-instruct"
        if instruct_slug in _KNOWN_MODEL_METRICS:
            inst_metrics = _KNOWN_MODEL_METRICS[instruct_slug]
            # Heuristic: Base usually -10% MMLU, Speed similar (1.05x)
            return {
                "mmlu": inst_metrics["mmlu"] * 0.9 if inst_metrics["mmlu"] else None,
                "tps": (inst_metrics["tps"] * 1.05) if inst_metrics["tps"] else None
            }

    return None

def predict_model_specs(full_model_id: str) -> Dict[str, Any]:
    """
    Predicts pricing (USD/1M tokens), latency (sec/token), and MMLU-Pro score
    based on the semantic naming of the model. 
    
    Logic decouples the provider (e.g., 'perplexity', 'together_ai') and looks 
    at the model slug (e.g., 'sonar-reasoning') to determine the tier.
    """
    
    # 1. Normalize: Remove provider prefix to focus on the model name
    model_slug = full_model_id.split('/')[-1].lower()
    
    # Defaults (Baseline: approx. GPT-3.5 level)
    prediction = {
        "usd_per_1m_input": 0.50,
        "usd_per_1m_output": 1.50,
        "usd_per_1m_audio_input": None, # Only set for multimodal
        "seconds_per_output_token": 0.02, # ~50 tokens/sec
        "mmlu_pro_score": 40.0,
        "tier": "Standard"
    }

    # 1. REASONING / "THINKING" MODELS (Highest Cost, Highest Latency, Best Score)
    # Keywords: o1, o3, o4, r1 (DeepSeek), reasoning, thinking
    if re.search(r'\b(o1|o3|o4|r1|reasoning|thinking)\b', model_slug):
        prediction["tier"] = "Reasoning (Heavy)"
        prediction["mmlu_pro_score"] = 85.0
        prediction["seconds_per_output_token"] = 0.08  # Slower (~12 tok/s)
        prediction["usd_per_1m_input"] = 15.00
        prediction["usd_per_1m_output"] = 60.00
        
        # Adjustment for "Mini/Fast" reasoning models
        if re.search(r'(mini|fast|distill)', model_slug):
            prediction["tier"] = "Reasoning (Efficient)"
            prediction["usd_per_1m_input"] = 3.00
            prediction["usd_per_1m_output"] = 12.00
            prediction["seconds_per_output_token"] = 0.03

    # 2. FUTURE FLAGSHIPS (GPT-5, Llama 4, Gemini 3, Opus 4/4.5)
    # Keywords: gpt-5, llama-4, gemini-3, opus-4, claude-4
    elif re.search(r'(gpt-5|llama-4|gemini-3|opus-4|sonnet-4|grok-4|mistral-large-3)', model_slug):
        prediction["tier"] = "Next-Gen Flagship"
        prediction["mmlu_pro_score"] = 92.0 # Theoretical future score
        prediction["seconds_per_output_token"] = 0.04 
        prediction["usd_per_1m_input"] = 5.00
        prediction["usd_per_1m_output"] = 15.00

    # 3. CURRENT FLAGSHIPS (GPT-4, Opus 3/3.5, Gemini 1.5/2.5 Pro, Llama 3.1 405B)
    # Keywords: gpt-4, opus, gemini.*pro, large, 405b, command-r-plus, grok-3
    elif re.search(r'(gpt-4|opus|gemini.*pro|large|405b|command-r-plus|grok-3)', model_slug):
        prediction["tier"] = "Current Flagship"
        prediction["mmlu_pro_score"] = 75.0
        prediction["seconds_per_output_token"] = 0.03
        prediction["usd_per_1m_input"] = 2.50
        prediction["usd_per_1m_output"] = 10.00
        
        # Special check for "Turbo/Flash" versions of flagships (Cheaper/Faster)
        if re.search(r'(turbo|flash|lite|fast)', model_slug):
            prediction["tier"] += " (Optimized)"
            prediction["usd_per_1m_input"] = 0.15
            prediction["usd_per_1m_output"] = 0.60
            prediction["seconds_per_output_token"] = 0.01 # Fast (~100 tok/s)

    # 4. BALANCED / HIGH-MID (Sonnet, Llama 70B, Grok 2, Mistral Medium)
    # Keywords: sonnet, 70b, 90b, medium, grok-2, command-r (standard)
    elif re.search(r'(sonnet|70b|90b|medium|grok-2|command-r)', model_slug):
        prediction["tier"] = "Balanced"
        prediction["mmlu_pro_score"] = 65.0
        prediction["seconds_per_output_token"] = 0.015
        prediction["usd_per_1m_input"] = 3.00
        prediction["usd_per_1m_output"] = 15.00
        
        # Anthropic Sonnet 3.5/3.7 is exceptionally good, bump score
        if "sonnet" in model_slug:
             prediction["mmlu_pro_score"] = 78.0

    # 5. ECONOMY / SMALL (Haiku, Mini, 8B, 7B, 3B, Flash-Lite)
    # Keywords: haiku, mini, nano, 8b, 7b, small, micro
    elif re.search(r'(haiku|mini|nano|small|micro|\b[1-9]b\b|1[0-4]b)', model_slug):
        prediction["tier"] = "Economy"
        prediction["mmlu_pro_score"] = 45.0
        prediction["seconds_per_output_token"] = 0.008 # Extremely fast
        prediction["usd_per_1m_input"] = 0.15
        prediction["usd_per_1m_output"] = 0.60
        
        if "nano" in model_slug or "1b" in model_slug or "3b" in model_slug:
             prediction["tier"] = "Edge/Nano"
             prediction["usd_per_1m_input"] = 0.05
             prediction["usd_per_1m_output"] = 0.20

    # AUDIO / MULTIMODAL DETECTION
    # If the model is known to handle native audio, assign audio pricing
    # Keywords: 4o, omni, gemini (usually multimodal), audio
    if re.search(r'(4o|omni|gemini|audio)', model_slug):
        # Audio input is generally 2-5x text input price
        prediction["usd_per_1m_audio_input"] = prediction["usd_per_1m_input"] * 4.0

    # BENCHMARK DATA OVERRIDE
    # Attempt to fetch ground-truth MMLU/Latency from known dataset
    bench_data = _find_closest_benchmark_metric(model_slug)
    if bench_data:
        # Override MMLU if available
        if bench_data.get("mmlu") is not None:
            prediction["mmlu_pro_score"] = bench_data["mmlu"]
        
        # Override Latency if available
        # Note: input is tokens/sec, we need seconds/token (1/x)
        tps = bench_data.get("tps")
        if tps and tps > 0:
            prediction["seconds_per_output_token"] = 1.0 / tps

    return prediction

# helper function to select the list of available models based on the 
def get_available_model_from_env(include_embedding: bool = False):
    available_models = []
    # Check for Vertex default credentials path if env var is missing
    default_gcloud_creds = os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
    has_vertex_file_creds = os.path.exists(default_gcloud_creds)

    for model in CuratedModel:
        model_id = model.value
        if not include_embedding and model.is_embedding_model():
            continue
        env_var_name = get_api_key_env_var(model_id)
        is_available = False
        if env_var_name and os.getenv(env_var_name):
            is_available = True
        elif get_model_provider(model_id) == "vertex_ai" and has_vertex_file_creds:
            is_available = True
        if is_available:
            available_models.append(model_id)
    return available_models

def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None) -> list[CuratedModel]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") not in [None, ""]:
        openai_models = [model for model in CuratedModel if model.is_openai_model()]
        if not include_embedding:
            openai_models = [
                model for model in openai_models if not model.is_embedding_model()
            ]
        models.extend(openai_models)

    if os.getenv("TOGETHER_API_KEY") not in [None, ""]:
        together_models = [model for model in CuratedModel if model.is_together_model()]
        if not include_embedding:
            together_models = [
                model for model in together_models if not model.is_embedding_model()
            ]
        models.extend(together_models)

    if os.getenv("ANTHROPIC_API_KEY") not in [None, ""]:
        anthropic_models = [model for model in CuratedModel if model.is_anthropic_model()]
        if not include_embedding:
            anthropic_models = [
                model for model in anthropic_models if not model.is_embedding_model()
            ]
        models.extend(anthropic_models)

    gemini_credentials_path = (
        os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
        if gemini_credentials_path is None
        else gemini_credentials_path
    )
    if os.getenv("GEMINI_API_KEY") not in [None, ""] or (use_vertex and os.path.exists(gemini_credentials_path)):
        vertex_models = [model for model in CuratedModel if model.is_vertex_model()]
        google_ai_studio_models = [model for model in CuratedModel if model.is_google_ai_studio_model()]
        if not include_embedding:
            vertex_models = [
                model for model in vertex_models if not model.is_embedding_model()
            ]
        if use_vertex:
            models.extend(vertex_models)
        else:
            models.extend(google_ai_studio_models)

    if api_base is not None:
        vllm_models = [model for model in CuratedModel if model.is_vllm_model()]
        if not include_embedding:
            vllm_models = [
                model for model in vllm_models if not model.is_embedding_model()
            ]
        models.extend(vllm_models)

    return models
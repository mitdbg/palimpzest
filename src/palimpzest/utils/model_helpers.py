import os, re, json
import yaml, time, requests, subprocess, os, socket, json, random
from typing import Dict, Any, Optional
from palimpzest.core.models import PlanCost
from palimpzest.policy import Policy

def get_api_key_env_var(model_provider: str):
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

# ---------------------------------------------------------------------------
# PREFETCHING MODEL SPECS UTILITIES
# ---------------------------------------------------------------------------

LITELLM_MODEL_METRICS = {}
CURATED_MODEL_METRICS = {}

def load_known_metrics():
    global LITELLM_MODEL_METRICS, CURATED_MODEL_METRICS
    if not LITELLM_MODEL_METRICS:
        model_prices_and_context_window_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        response = requests.get(model_prices_and_context_window_url)
        LITELLM_MODEL_METRICS = response.json()
    if not CURATED_MODEL_METRICS:
        curated_model_metrics_path = os.path.join(os.path.dirname(__file__), 'curated_model_info.json')
        with open(curated_model_metrics_path, 'r') as f:
            CURATED_MODEL_METRICS = json.load(f)

load_known_metrics()

def get_known_model_info(full_model_id):
    global LITELLM_MODEL_METRICS, CURATED_MODEL_METRICS
    # Initialize the target dictionary with None
    unified_info = {
        "is_reasoning_model": None,
        "is_vision_model": None,
        "is_text_model": None,
        "is_audio_model": None,
        "is_embedding_model": None,
        "usd_per_input_token": None,
        "usd_per_output_token": None,
        "usd_per_audio_input_token": None,
        "output_tokens_per_second": None,
        "overall_score": None,
        "litellm_provider": None
    }
    
    # Normalize model name (remove provider prefix)
    if "/" in full_model_id:
        model_name = full_model_id.split("/", 1)[1]
    else:
        model_name = full_model_id

    # search logic: check full_model_id first, then model_name
    data_source_1 = LITELLM_MODEL_METRICS.get(full_model_id) or LITELLM_MODEL_METRICS.get(model_name)
    if data_source_1:
        mode = data_source_1.get("mode", "")
        unified_info["is_reasoning_model"] = data_source_1.get("supports_reasoning")
        unified_info["is_vision_model"] = data_source_1.get("supports_vision")
        unified_info["is_text_model"] = (mode == "chat" or mode == "completion")
        unified_info["is_embedding_model"] = (mode == "embedding")
        unified_info["is_audio_model"] = data_source_1.get("supports_audio_input")
        unified_info["usd_per_input_token"] = data_source_1.get("input_cost_per_token")
        unified_info["usd_per_output_token"] = data_source_1.get("output_cost_per_token")
        unified_info["usd_per_audio_input_token"] = data_source_1.get("input_cost_per_audio_token")
        unified_info["litellm_provider"] = data_source_1.get("litellm_provider")

    # search logic: check model_name only
    data_source_2 = CURATED_MODEL_METRICS.get(model_name)

    if data_source_2:
        mapping_2 = {
            "is_reasoning_model": "is_reasoning_model",
            "is_vision_model": "is_vision_model",
            "is_text_model": "is_text_model",
            "is_audio_model": "is_audio_model",
            "is_embedding_model": "is_embedding_model",
            "usd_per_input_token": "usd_per_input_token",
            "usd_per_output_token": "usd_per_output_token",
            "usd_per_audio_input_token": "usd_per_audio_input_token",
            "output_tokens_per_second": "output_tokens_per_second",
            "overall_score": "MMLU_Pro_score"
        }
        for target_key, source_key in mapping_2.items():
            # ONLY fill if currently None (do not overwrite)
            if unified_info[target_key] is None:
                val = data_source_2.get(source_key)
                if val is not None:
                    unified_info[target_key] = val

    return unified_info


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

def _find_closest_benchmark_metric(model_slug: str) -> Optional[Dict[str, float]]:
    """
    Looks for benchmark data (MMLU, TPS) in CURATED_MODEL_METRICS using fuzzy matching.
    """
    global CURATED_MODEL_METRICS
    if not CURATED_MODEL_METRICS:
        return None
        
    slug = model_slug.lower()

    # 1. Exact Match
    if slug in CURATED_MODEL_METRICS:
        data = CURATED_MODEL_METRICS[slug]
        return {
            "mmlu": data.get("MMLU_Pro_score"),
            "tps": data.get("output_tokens_per_second")
        }

    # 2. Date/Version-invariant Match (e.g. gpt-4o matches gpt-4o-2024-05-13)
    # matching keys starting with slug or containing slug
    matches = [k for k in CURATED_MODEL_METRICS.keys() if k.lower().startswith(slug) or slug in k.lower()]
    if matches:
        # Pick the first match (or sort by length/similarity if needed)
        best_match = matches[0]
        data = CURATED_MODEL_METRICS[best_match]
        return {
            "mmlu": data.get("MMLU_Pro_score"),
            "tps": data.get("output_tokens_per_second")
        }

    # 3. Sibling Inference (Base <-> Instruct)
    is_instruct = "instruct" in slug or "chat" in slug
    base_slug = slug.replace("-instruct", "").replace("-chat", "").strip("-")
    
    # Try finding the base version in the curated list
    # (Note: keys in curated list are often case sensitive, so we iterate)
    curated_keys_lower = {k.lower(): k for k in CURATED_MODEL_METRICS.keys()}
    
    if is_instruct and base_slug in curated_keys_lower:
        base_key = curated_keys_lower[base_slug]
        base_data = CURATED_MODEL_METRICS[base_key]
        base_mmlu = base_data.get("MMLU_Pro_score")
        base_tps = base_data.get("output_tokens_per_second")
        
        return {
            "mmlu": base_mmlu * 1.1 if base_mmlu else None, # Instruct often scores higher
            "tps": base_tps * 0.95 if base_tps else None   # Instruct slightly slower
        }
        
    return None

def _generate_heuristic_specs(model_slug: str) -> Dict[str, Any]:
    """Generates regex-based fallback estimates."""
    prediction = {
        "usd_per_1m_input": 0.50,
        "usd_per_1m_output": 1.50,
        "usd_per_1m_audio_input": None,
        "seconds_per_output_token": 0.02,
        "mmlu_pro_score": 40.0,
        "is_reasoning_model": False,
        "is_vision_model": False,
        "is_audio_model": False,
        "is_embedding_model": False
    }

    # Reasoning
    if re.search(r'\b(o1|o3|o4|r1|reasoning|thinking)\b', model_slug):
        prediction["is_reasoning_model"] = True
        prediction["mmlu_pro_score"] = 85.0
        prediction["seconds_per_output_token"] = 0.08
        prediction["usd_per_1m_input"] = 15.00
        prediction["usd_per_1m_output"] = 60.00
        if re.search(r'(mini|fast|distill)', model_slug):
            prediction["usd_per_1m_input"] = 3.00
            prediction["usd_per_1m_output"] = 12.00
            prediction["seconds_per_output_token"] = 0.03

    # Flagships (Future/Current)
    elif re.search(r'(gpt-5|llama-4|gemini-3|opus-4)', model_slug):
        prediction["mmlu_pro_score"] = 92.0
        prediction["seconds_per_output_token"] = 0.04 
        prediction["usd_per_1m_input"] = 5.00
        prediction["usd_per_1m_output"] = 15.00
    elif re.search(r'(gpt-4|opus|gemini.*pro|large|405b|grok-3)', model_slug):
        prediction["mmlu_pro_score"] = 75.0
        prediction["seconds_per_output_token"] = 0.03
        prediction["usd_per_1m_input"] = 2.50
        prediction["usd_per_1m_output"] = 10.00
        if re.search(r'(turbo|flash|lite)', model_slug):
            prediction["usd_per_1m_input"] = 0.15
            prediction["usd_per_1m_output"] = 0.60
            prediction["seconds_per_output_token"] = 0.01

    # Balanced/Economy
    elif re.search(r'(sonnet|70b|90b|medium|grok-2)', model_slug):
        prediction["mmlu_pro_score"] = 65.0
        prediction["seconds_per_output_token"] = 0.015
        prediction["usd_per_1m_input"] = 3.00
        prediction["usd_per_1m_output"] = 15.00
    elif re.search(r'(haiku|mini|nano|small|8b|7b)', model_slug):
        prediction["mmlu_pro_score"] = 45.0
        prediction["seconds_per_output_token"] = 0.008
        prediction["usd_per_1m_input"] = 0.15
        prediction["usd_per_1m_output"] = 0.60
        if "nano" in model_slug or "1b" in model_slug:
             prediction["usd_per_1m_input"] = 0.05
             prediction["usd_per_1m_output"] = 0.20

    # Modality
    if re.search(r'(4o|omni|gemini|audio)', model_slug):
        prediction["is_audio_model"] = True
        prediction["usd_per_1m_audio_input"] = prediction["usd_per_1m_input"] * 4.0
    if re.search(r'(vision|4o|gemini|claude-3|pixtral|llama-3.2)', model_slug):
        prediction["is_vision_model"] = True
    if "embedding" in model_slug:
        prediction["is_embedding_model"] = True

    return prediction

def get_model_specs(full_model_id: str) -> Dict[str, Any]:
    """
    Predicts pricing, latency, and scores.
    Uses strict lookup -> fuzzy benchmark lookup -> heuristics.
    """
    # 1. Fetch Objective Info (Strict Matches)
    specs = get_known_model_info(full_model_id)
    
    # Initialize Metadata: True if value is None (meaning it needs estimation)
    metadata = {k: (v is None) for k, v in specs.items()}
    model_slug = full_model_id.split('/')[-1].lower()

    # 2. Fill Provider
    if specs["litellm_provider"] is None:
        specs["litellm_provider"] = get_model_provider(full_model_id)

    # 3. Fuzzy Benchmark Lookup (Overlay if missing)
    # If we missed score or speed in the strict lookup, try the fuzzy/sibling lookup in CURATED
    if specs["overall_score"] is None or specs["output_tokens_per_second"] is None:
        bench_data = _find_closest_benchmark_metric(model_slug)
        if bench_data:
            if specs["overall_score"] is None and bench_data.get("mmlu") is not None:
                specs["overall_score"] = bench_data["mmlu"]
                # Note: We keep metadata=True because this wasn't in the strict 'get_known_model_info'
            
            if specs["output_tokens_per_second"] is None and bench_data.get("tps") is not None:
                specs["output_tokens_per_second"] = bench_data["tps"]

    # 4. Heuristic Fallback (Overlay if still missing)
    heuristics = _generate_heuristic_specs(model_slug)

    # Booleans
    for key in ["is_reasoning_model", "is_vision_model", "is_text_model", "is_audio_model", "is_embedding_model"]:
        if specs[key] is None:
            specs[key] = heuristics.get(key, False)
            if key == "is_text_model" and specs[key] is None:
                 specs[key] = True # Default to text

    # Pricing
    if specs["usd_per_input_token"] is None:
        specs["usd_per_input_token"] = heuristics["usd_per_1m_input"] / 1_000_000.0
    if specs["usd_per_output_token"] is None:
        specs["usd_per_output_token"] = heuristics["usd_per_1m_output"] / 1_000_000.0
    if specs["usd_per_audio_input_token"] is None:
        if heuristics["usd_per_1m_audio_input"]:
            specs["usd_per_audio_input_token"] = heuristics["usd_per_1m_audio_input"] / 1_000_000.0

    # Performance / Score (if even fuzzy lookup failed)
    if specs["output_tokens_per_second"] is None:
        sec_per_tok = heuristics["seconds_per_output_token"]
        specs["output_tokens_per_second"] = 1.0 / sec_per_tok if sec_per_tok > 0 else 0
        
    if specs["overall_score"] is None:
        specs["overall_score"] = heuristics["mmlu_pro_score"]

    specs["metadata"] = metadata
    return specs

# ---------------------------------------------------------------------------
# DYNAMIC FETCHING UTILITIES
# ---------------------------------------------------------------------------

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def _generate_config_yaml(available_models):
    rand_id = random.randint(100000, 999999)
    config_filename = f"litellm_config_{rand_id}.yaml"
    config_list = []
    for model_str in available_models:
        env_var_name = get_api_key_env_var(model_str)
        api_key_val = f"os.environ/{env_var_name}" if env_var_name else None
        entry = {
            "model_name": model_str,
            "litellm_params": {
                "model": model_str,
                "api_key": api_key_val
            }
        }
        config_list.append(entry)
    yaml_structure = {"model_list": config_list}
    with open(config_filename, 'w') as f:
        yaml.dump(yaml_structure, f, default_flow_style=False, sort_keys=False)
    return config_filename

def fetch_dynamic_model_info(available_models):
    import palimpzest.constants as constants

    port = get_free_port()
    proxy_url = f"http://127.0.0.1:{port}" 
    config_filename = _generate_config_yaml(available_models)
    server_env = os.environ.copy()
    process = None
    dynamic_model_info = {}

    try:
        process = subprocess.Popen(
            ["litellm", "--config", config_filename, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=server_env
        )
        server_ready = False
        max_retries = 20
        for _ in range(max_retries):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"LiteLLM process died unexpectedly: {stderr.decode()}")
                break
            try:
                requests.get(f"{proxy_url}/health/readiness", timeout=1)
                server_ready = True
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                time.sleep(0.5)  
        if not server_ready:
            print("Timeout: LiteLLM server failed to start within the limit.")
            return {}
        try: 
            response = requests.get(f"{proxy_url}/model/info")
            response.raise_for_status()
            model_data = response.json()
        
            if "data" in model_data and len(model_data["data"]) > 0:
                for item in model_data["data"]:
                    model_name = item.get("model_name")
                    dynamic_model_info[model_name] = item.get("model_info", {})
        except Exception:
            pass
        if not dynamic_model_info:
            print(f"WARNING: LiteLLM server started but returned no model info.")
    
    finally: # cleanup
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        if os.path.exists(config_filename):
            os.remove(config_filename)

    constants.DYNAMIC_MODEL_INFO.update(dynamic_model_info)
    return dynamic_model_info

# ---------------------------------------------------------------------------
# SUBSAMPLE AVAILABLE MODELS UTILITIES
# ---------------------------------------------------------------------------

# helper function to select the list of available models based on the 
def get_available_model_from_env(include_embedding: bool = False):
    from palimpzest.constants import Model

    available_models = []
    # Check for Vertex default credentials path if env var is missing
    default_gcloud_creds = os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
    has_vertex_file_creds = os.path.exists(default_gcloud_creds)

    for model in Model:
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

def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None):
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    from palimpzest.constants import Model
    models = []
    if os.getenv("OPENAI_API_KEY") not in [None, ""]:
        openai_models = [model for model in Model if model.is_openai_model()]
        if not include_embedding:
            openai_models = [
                model for model in openai_models if not model.is_embedding_model()
            ]
        models.extend(openai_models)

    if os.getenv("TOGETHER_API_KEY") not in [None, ""]:
        together_models = [model for model in Model if model.is_together_model()]
        if not include_embedding:
            together_models = [
                model for model in together_models if not model.is_embedding_model()
            ]
        models.extend(together_models)

    if os.getenv("ANTHROPIC_API_KEY") not in [None, ""]:
        anthropic_models = [model for model in Model if model.is_anthropic_model()]
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
        vertex_models = [model for model in Model if model.is_vertex_model()]
        google_ai_studio_models = [model for model in Model if model.is_google_ai_studio_model()]
        if not include_embedding:
            vertex_models = [
                model for model in vertex_models if not model.is_embedding_model()
            ]
        if use_vertex:
            models.extend(vertex_models)
        else:
            models.extend(google_ai_studio_models)

    if api_base is not None:
        vllm_models = [model for model in Model if model.is_vllm_model()]
        if not include_embedding:
            vllm_models = [
                model for model in vllm_models if not model.is_embedding_model()
            ]
        models.extend(vllm_models)

    return models

def get_optimal_models( policy: Policy, include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None):
    """
    Selects the top models from the available list based on the user's policy.
    
    This function:
    1. Discovers available models (using env vars and params).
    2. Filters models that violate policy constraints (e.g. min quality).
    3. Calculates a weighted score for each model using the policy's weights.
    4. Returns the top 5 models with the highest score.
    """
    from palimpzest.constants import MODEL_CARDS
    # 1. Gather Available Models
    available_models = get_models(
        include_embedding=include_embedding, 
        use_vertex=use_vertex, 
        gemini_credentials_path=gemini_credentials_path, 
        api_base=api_base
    )
    # Convert enums to string IDs for processing
    if not available_models:
        return []

    # 2. Gather Metrics and Apply Constraints
    candidates = []
    for model in available_models:
        # Retrieve or predict metrics
        quality_score = model.get_overall_score()
        cost = model.get_usd_per_output_token()
        time_val = model.get_seconds_per_output_token()
        
        if quality_score is None: quality_score = 0
        if cost is None: cost = float("inf")
        if time_val is None: time_val = float("inf")

        # Create proxy plan for constraint checking
        normalized_quality = quality_score / 100.0
        proxy_plan = PlanCost(cost=0.0, time=0.0, quality=normalized_quality)
        
        if not policy.constraint(proxy_plan):
            continue

        candidates.append({
            "id": model, 
            "quality": quality_score, 
            "cost": cost, 
            "time": time_val
        })

    if not candidates:
        return []

    # 3. Normalize Metrics (Min-Max Normalization)
    quals = [c["quality"] for c in candidates]
    costs = [c["cost"] for c in candidates]
    times = [c["time"] for c in candidates]
    
    min_q, max_q = min(quals), max(quals)
    min_c, max_c = min(costs), max(costs)
    min_t, max_t = min(times), max(times)
    
    def normalize(val, min_v, max_v, invert=False):
        if max_v == min_v:
            return 1.0
        norm = (val - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    # 4. Calculate Scores
    weights = policy.get_dict()
    w_q = weights.get("quality", 0.0)
    w_c = weights.get("cost", 0.0)
    w_t = weights.get("time", 0.0)
    
    scored_candidates = []
    for cand in candidates:
        n_q = normalize(cand["quality"], min_q, max_q, invert=False)
        n_c = normalize(cand["cost"], min_c, max_c, invert=True)
        n_t = normalize(cand["time"], min_t, max_t, invert=True)
        
        score = (w_q * n_q) + (w_c * n_c) + (w_t * n_t)
        
        scored_candidates.append((score, cand["id"]))

    # 5. Select Top 5
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_models_ids = [mid for score, mid in scored_candidates[:5]]
    
    # Return list of Model objects
    top_models = [mid for mid in top_models_ids]
    
    return top_models
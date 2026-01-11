import requests, os, json, re
from typing import Dict, Any, Optional

LITELLM_MODEL_METRICS = {}
CURATED_MODEL_METRICS = {}

def load_known_metrics():
    global LITELLM_MODEL_METRICS, CURATED_MODEL_METRICS
    if not LITELLM_MODEL_METRICS:
        try:
            model_prices_and_context_window_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
            response = requests.get(model_prices_and_context_window_url, timeout=5)
            if response.status_code == 200:
                LITELLM_MODEL_METRICS = response.json()
        except Exception:
            pass # Fail gracefully if offline
            
    if not CURATED_MODEL_METRICS:
        curated_model_metrics_path = os.path.join(os.path.dirname(__file__), 'curated_model_info.json')
        if os.path.exists(curated_model_metrics_path):
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
        # "litellm_provider": None
    }
    
    # Normalize model name (remove provider prefix)
    if "/" in full_model_id:
        model_name = full_model_id.split("/", 1)[1]
    else:
        model_name = full_model_id

    # search logic: check full_model_id first, then model_name
    data_source_1 = LITELLM_MODEL_METRICS.get(full_model_id) or LITELLM_MODEL_METRICS.get(model_name)
    if data_source_1:
        # FIX: Only set is_text_model/is_embedding_model if "mode" is actually present.
        # Otherwise keep it None so heuristics can fill it in.
        if "mode" in data_source_1:
            mode = data_source_1["mode"]
            unified_info["is_text_model"] = (mode == "chat" or mode == "completion")
            unified_info["is_embedding_model"] = (mode == "embedding")
        
        unified_info["is_reasoning_model"] = data_source_1.get("supports_reasoning")
        unified_info["is_vision_model"] = data_source_1.get("supports_vision")
        unified_info["is_audio_model"] = data_source_1.get("supports_audio_input")
        unified_info["usd_per_input_token"] = data_source_1.get("input_cost_per_token")
        unified_info["usd_per_output_token"] = data_source_1.get("output_cost_per_token")
        unified_info["usd_per_audio_input_token"] = data_source_1.get("input_cost_per_audio_token")
        # unified_info["litellm_provider"] = data_source_1.get("litellm_provider")

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

    # 2. Date/Version-invariant Match
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
        "is_text_model": True, # default to true
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
    if re.search(r'(audio)', model_slug):
        prediction["is_audio_model"] = True
        prediction["usd_per_1m_audio_input"] = prediction["usd_per_1m_input"] * 4.0
    
    if re.search(r'(vision|4o|gemini|claude-3|pixtral|llama-3.2)', model_slug):
        prediction["is_vision_model"] = True
    if "embedding" in model_slug:
        prediction["is_embedding_model"] = True
        prediction["is_text_model"] = False

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

    # 3. Fuzzy Benchmark Lookup (Overlay if missing)
    if specs["overall_score"] is None or specs["output_tokens_per_second"] is None:
        bench_data = _find_closest_benchmark_metric(model_slug)
        if bench_data:
            if specs["overall_score"] is None and bench_data.get("mmlu") is not None:
                specs["overall_score"] = bench_data["mmlu"]
            
            if specs["output_tokens_per_second"] is None and bench_data.get("tps") is not None:
                specs["output_tokens_per_second"] = bench_data["tps"]

    # 4. Heuristic Fallback (Overlay if still missing)
    heuristics = _generate_heuristic_specs(model_slug)

    # Booleans
    for key in ["is_reasoning_model", "is_vision_model", "is_text_model", "is_audio_model", "is_embedding_model"]:
        if specs[key] is None:
            specs[key] = heuristics.get(key, False)

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
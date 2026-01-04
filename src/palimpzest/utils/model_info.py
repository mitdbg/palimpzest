import yaml, time, requests, subprocess, os
from palimpzest.core.models import PlanCost
from palimpzest.policy import Policy
from palimpzest.constants import MODEL_CARDS, CuratedModel
from palimpzest.utils.model_helpers import predict_model_specs, get_model_provider, get_api_key_env_var

CONFIG_FILENAME = "litellm_config.yaml"
PROXY_PORT = 4000
PROXY_URL = f"http://0.0.0.0:{PROXY_PORT}"
DYNAMIC_MODEL_INFO = {}

def _generate_config_yaml(available_models, filename):
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
    with open(filename, 'w') as f:
        yaml.dump(yaml_structure, f, default_flow_style=False)

def _wait_for_server(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if requests.get(f"{url}/health").status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False

def fetch_dynamic_model_info(available_models):
    global DYNAMIC_MODEL_INFO
    _generate_config_yaml(available_models, CONFIG_FILENAME)
    process = subprocess.Popen(
        ["litellm", "--config", CONFIG_FILENAME, "--port", str(PROXY_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    dynamic_model_info = {}
    try:
        if not _wait_for_server(PROXY_PORT):
            raise Exception("Sever failed to start")
        response = requests.get(
            f"{PROXY_URL}/model/info", 
            headers={"Authorization": "Bearer sk-1234"}
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            for item in data["data"]:
                model_name = item.get("model_name")
                dynamic_model_info[model_name] = item.get("model_info", {})
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    DYNAMIC_MODEL_INFO = dynamic_model_info
    return dynamic_model_info

class Model(str):
    """
    Model describes the underlying LLM which should be used to perform some operation
    which requires invoking an LLM. It does NOT specify whether the model need be executed
    remotely or locally (if applicable).
    """
    def __new__(cls, value):
        return super(Model, cls).__new__(cls, value)
    
    def __init__(self):
        self.prediction = predict_model_specs(self)
    
    @property
    def value(self):
        return self
    
    def __repr__(self):
        return f"{self}"

    def is_llama_model(self):
        return "llama" in self.lower()

    def is_clip_model(self):
        return "clip" in self.lower()

    def is_together_model(self):
         return get_model_provider() == "together_ai"
    
    def is_anthropic_model(self):
        return get_model_provider() == "anthropic"
    
    def is_openai_model(self):
        return get_model_provider() == "openai" or self.is_text_embedding_model()

    def is_vertex_model(self):
        return get_model_provider() == "vertex_ai"

    def is_google_ai_studio_model(self):
        return get_model_provider() == "gemini"

    def is_text_embedding_model(self):
        return "text-embedding" in self.lower()
    
    def is_vllm_model(self):
        return "hosted_vllm" in self.lower()

    def is_o_model(self):
        return self in {CuratedModel.o4_MINI.value}
    
    def is_gpt_5_model(self):
        gpt_5_models = {
            CuratedModel.GPT_5.value,
            CuratedModel.GPT_5_MINI.value,
            CuratedModel.GPT_5_NANO.value
        }
        return self.value in gpt_5_models

    def is_reasoning_model(self):
        info = DYNAMIC_MODEL_INFO(self.value)
        if "supports_reasoning" in info and info["supports_reasoning"] is not None:
            return info["supports_reasoning"]
        # configured list
        known_reasoning_models = {
            CuratedModel.GPT_5.value,
            CuratedModel.GPT_5_MINI.value,
            CuratedModel.GPT_5_NANO.value,
            CuratedModel.o4_MINI.value,
            CuratedModel.GEMINI_3_0_PRO.value,
            CuratedModel.GEMINI_3_0_FLASH.value,
            CuratedModel.GEMINI_2_5_PRO.value,
            CuratedModel.GEMINI_2_5_FLASH.value,
            CuratedModel.GEMINI_3_0_FLASH.value,
            CuratedModel.GEMINI_3_0_PRO.value,
            CuratedModel.GOOGLE_GEMINI_2_5_PRO.value,
            CuratedModel.GOOGLE_GEMINI_2_5_FLASH.value,
            CuratedModel.GOOGLE_GEMINI_2_5_FLASH_LITE.value,
            CuratedModel.CLAUDE_3_7_SONNET.value,
        }
        return self.value in known_reasoning_models
    
    def is_vision_model(self):
        info = DYNAMIC_MODEL_INFO[self.value]
        if "supports_vision" in info and info["supports_vision"] is not None:
            return info["supports_vision"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_vision_model()
        except Exception:
            return False
    
    def is_audio_model(self):
        info = DYNAMIC_MODEL_INFO[self.value]
        if "supports_audio_input" in info and info["supports_audio_input"] is not None:
            return info["supports_audio_input"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_audio_model()
        except Exception:
            return False # default
    
    def is_text_model(self):
        info = DYNAMIC_MODEL_INFO[self.value]
        if "mode" in info:
            return info["mode"] in ["chat", "completion"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_text_model(self)
        except Exception:
            return False # default
    
    def is_embedding_model(self):
        info = DYNAMIC_MODEL_INFO(self)
        if "mode" in info:
            return info["mode"] == "embedding"
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_embedding_model(self)
        except Exception:
            return False # default

    def is_text_image_multimodal_model(self):
        return self.is_vision_model() and self.is_text_model()

    def is_text_audio_multimodal_model(self):
        return self.is_audio_model() and self.is_text_model()
    
    def get_usd_per_input_token(self):
        info = DYNAMIC_MODEL_INFO(self)
        if "input_cost_per_token" in info and info["input_cost_per_token"] is not None:
            return info["input_cost_per_token"]
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["usd_per_input_token"]
        return self.prediction["usd_per_1m_input"]/1e6
    
    def get_usd_per_output_token(self):
        info = DYNAMIC_MODEL_INFO(self)
        if "output_cost_per_token" in info and info["output_cost_per_token"] is not None:
            return info["output_cost_per_token"]
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["usd_per_output_token"]
        return self.prediction["usd_per_1m_input"]/1e6
    
    def get_seconds_per_output_token(self):
        # LiteLLM endpoint doesn't provide information on the latency
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["seconds_per_output_token"]
        return self.prediction["seconds_per_output_token"]

    def get_usd_per_audio_input_token(self):
        assert self.is_audio_model(), "model must be an audio model to retrieve audio input token cost"
        info = DYNAMIC_MODEL_INFO(self)
        if "input_cost_per_audio_token" in info and info["input_cost_per_audio_token"] is not None:
            return info["input_cost_per_audio_token"]
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["usd_per_audio_input_token"]
        if self.prediction["usd_per_1m_audio_input"] is not None:
            return self.prediction["usd_per_1m_audio_input"]/1e6
    
    def get_overall_score(self):
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["overall"]
        return self.prediction["mmlu_pro_score"]
    

def get_optimal_models(policy: Policy) -> list[Model]:
    """
    Selects the top models from the available list based on the user's policy.
    
    This function:
    1. Filters models that violate policy constraints (e.g. min quality).
    2. Calculates a weighted score for each model using the policy's weights 
       (Quality, Cost, Time).
    3. Returns the top 5 models with the highest score.
    """
    model_ids = get_available_model_from_env()

    if not model_ids:
        return []

    # 1. Gather Metrics and Apply Constraints
    candidates = []
    for mid in model_ids:
        # Retrieve or predict metrics
        card = MODEL_CARDS.get(mid)
        if not card: 
            specs = predict_model_specs(mid)
            quality_score = specs.get("mmlu_pro_score", 0)
            cost = specs.get("usd_per_1m_output", float("inf")) / 1e6
            time_val = specs.get("seconds_per_output_token", float("inf"))
        else:
            quality_score = card.get("overall", 0)
            cost = card.get("usd_per_output_token", float("inf"))
            time_val = card.get("seconds_per_output_token", float("inf"))
        
        if quality_score is None: quality_score = 0
        if cost is None: cost = float("inf")
        if time_val is None: time_val = float("inf")

        # Create proxy plan for constraint checking
        # (Cost/Time set to 0.0 as we only check unit-invariant constraints like Quality here)
        normalized_quality = quality_score / 100.0
        proxy_plan = PlanCost(cost=0.0, time=0.0, quality=normalized_quality)
        
        if not policy.constraint(proxy_plan):
            continue

        candidates.append({
            "id": mid, 
            "quality": quality_score, 
            "cost": cost, 
            "time": time_val
        })

    if not candidates:
        return []

    # 2. Normalize Metrics (Min-Max Normalization)
    # We want to map everything to [0, 1] to apply weights fairly.
    
    # Extract ranges
    quals = [c["quality"] for c in candidates]
    costs = [c["cost"] for c in candidates]
    times = [c["time"] for c in candidates]
    
    min_q, max_q = min(quals), max(quals)
    min_c, max_c = min(costs), max(costs)
    min_t, max_t = min(times), max(times)
    
    # Helper for safe normalization
    def normalize(val, min_v, max_v, invert=False):
        if max_v == min_v:
            return 1.0 # If all values are same, treat as 'best'
        norm = (val - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    # 3. Calculate Scores
    # Weights from policy (e.g., {'cost': 1.0, 'time': 0.0, 'quality': 0.0})
    weights = policy.get_dict()
    w_q = weights.get("quality", 0.0)
    w_c = weights.get("cost", 0.0)
    w_t = weights.get("time", 0.0)
    
    scored_candidates = []
    for cand in candidates:
        # Quality is Benefit (Higher is Better)
        n_q = normalize(cand["quality"], min_q, max_q, invert=False)
        
        # Cost and Time are Costs (Lower is Better), so we invert the normalization
        # so that 1.0 represents the cheapest/fastest (best) option.
        n_c = normalize(cand["cost"], min_c, max_c, invert=True)
        n_t = normalize(cand["time"], min_t, max_t, invert=True)
        
        # Weighted Score (Simple Additive Weighting)
        score = (w_q * n_q) + (w_c * n_c) + (w_t * n_t)
        
        scored_candidates.append((score, cand["id"]))

    # 4. Select Top 5
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_models_ids = [mid for score, mid in scored_candidates[:5]]
    top_models = [Model(id) in top_models_ids]
    
    return top_models

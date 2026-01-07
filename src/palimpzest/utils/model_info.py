import yaml, time, requests, subprocess, os, socket, json, random
from palimpzest.core.models import PlanCost
from palimpzest.policy import Policy
from palimpzest.constants import MODEL_CARDS, CuratedModel
from palimpzest.utils.model_helpers import predict_model_specs, get_model_provider, get_api_key_env_var, get_models

DYNAMIC_MODEL_INFO = {}

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
    global DYNAMIC_MODEL_INFO
    
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

    finally:
        # 6. Robust Cleanup
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        if os.path.exists(config_filename):
            os.remove(config_filename)

    DYNAMIC_MODEL_INFO = dynamic_model_info
    return dynamic_model_info

class Model(str):
    """
    Supports any model id. Returns information about the model previously in MODEL_CARDS
    """
    def __new__(cls, value):
        if isinstance(value, Model):
            return value
        return super(Model, cls).__new__(cls, value)
    
    def __init__(self, value):
        if isinstance(value, Model):
            value = value.value
        self.prediction = predict_model_specs(self)
    
    @property
    def value(self):
        return str(self)
    
    def __repr__(self):
        return f"{self}"

    def is_llama_model(self):
        return "llama" in self.lower()

    def is_clip_model(self):
        return "clip" in self.lower()

    def is_together_model(self):
         return get_model_provider(self.value) == "together_ai"
    
    def is_anthropic_model(self):
        return get_model_provider(self.value) == "anthropic"
    
    def is_openai_model(self):
        return get_model_provider(self.value) == "openai" or self.is_text_embedding_model()

    def is_vertex_model(self):
        return get_model_provider(self.value) == "vertex_ai"

    def is_google_ai_studio_model(self):
        return get_model_provider(self.value) == "gemini"

    def is_text_embedding_model(self):
        return "text-embedding" in self.lower()
    
    def is_vllm_model(self):
        return "hosted_vllm" in self.lower()

    def is_o_model(self):
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_o_model()
        except Exception:
            return False
    
    def is_gpt_5_model(self):
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_gpt_5_model()
        except Exception:
            return False
        
    def is_reasoning_model(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "supports_reasoning" in info and info["supports_reasoning"] is not None:
            return info["supports_reasoning"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_reasoning_model()
        except Exception:
            return False
    
    def is_vision_model(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "supports_vision" in info and info["supports_vision"] is not None:
            return info["supports_vision"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_vision_model()
        except Exception:
            return False
    
    def is_audio_model(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "supports_audio_input" in info and info["supports_audio_input"] is not None:
            return info["supports_audio_input"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_audio_model()
        except Exception:
            return False # default
    
    def is_text_model(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "mode" in info:
            return info["mode"] in ["chat", "completion"]
        # configured list
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_text_model()
        except Exception:
            return False # default
    
    def is_embedding_model(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "mode" in info:
            return info["mode"] == "embedding"
        try:
            configured_model = CuratedModel(self.value)
            return configured_model.is_embedding_model()
        except Exception:
            return False # default

    def is_text_image_multimodal_model(self):
        return self.is_vision_model() and self.is_text_model()

    def is_text_audio_multimodal_model(self):
        return self.is_audio_model() and self.is_text_model()
    
    def get_usd_per_input_token(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
        if "input_cost_per_token" in info and info["input_cost_per_token"] is not None:
            return info["input_cost_per_token"]
        if self.value in MODEL_CARDS:
            return MODEL_CARDS[self]["usd_per_input_token"]
        return self.prediction["usd_per_1m_input"]/1e6
    
    def get_usd_per_output_token(self):
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
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
        info = DYNAMIC_MODEL_INFO.get(self.value, {})
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
    

def get_optimal_models(
    policy: Policy, 
    include_embedding: bool = False, 
    use_vertex: bool = False, 
    gemini_credentials_path: str | None = None, 
    api_base: str | None = None
) -> list[Model]:
    """
    Selects the top models from the available list based on the user's policy.
    
    This function:
    1. Discovers available models (using env vars and params).
    2. Filters models that violate policy constraints (e.g. min quality).
    3. Calculates a weighted score for each model using the policy's weights.
    4. Returns the top 5 models with the highest score.
    """
    # 1. Gather Available Models
    available_model_enums = get_models(
        include_embedding=include_embedding, 
        use_vertex=use_vertex, 
        gemini_credentials_path=gemini_credentials_path, 
        api_base=api_base
    )
    # Convert enums to string IDs for processing
    model_ids = [m.value for m in available_model_enums]

    if not model_ids:
        return []

    # 2. Gather Metrics and Apply Constraints
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
    top_models = [Model(mid) for mid in top_models_ids]
    
    return top_models
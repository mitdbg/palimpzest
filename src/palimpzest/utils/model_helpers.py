import yaml, time, requests, subprocess, os, socket, random, os
from palimpzest.core.models import PlanCost
from palimpzest.policy import Policy
from palimpzest.constants import Model, ModelProvider, DYNAMIC_MODEL_INFO

# helper function to select the list of available models based on the 
def get_available_model_from_env(include_embedding: bool = False):
    available_models = []
    for model in Model:
        # Skip embedding models if not requested
        if not include_embedding and model.is_embedding_model():
            continue
        if model.api_key is not None:
            available_models.append(model.value)
        elif model.provider == ModelProvider.HOSTED_VLLM:
            available_models.append(model.value)
    return available_models

def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None):
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    
    # We iterate over all defined Models and let their internal logic decide if they match the criteria
    for model in Model:
        # 1. Filter by embedding preference
        if not include_embedding and model.is_embedding_model():
            continue
        # 2. Filter by availability (API Key existence)
        has_key = model.api_key is not None
        # Special handling for user-provided Vertex credentials path (overriding default)
        if model.provider == ModelProvider.VERTEX_AI and gemini_credentials_path:
            if os.path.exists(gemini_credentials_path):
                has_key = True
        # 3. Add models based on Provider groups (replicating original logic structure)
        if model.provider == ModelProvider.OPENAI and has_key:
            models.append(model)
        elif model.provider == ModelProvider.TOGETHER_AI and has_key:
            models.append(model)
        elif model.provider == ModelProvider.ANTHROPIC and has_key:
            models.append(model)
        elif model.provider == ModelProvider.VERTEX_AI and has_key and use_vertex:
            models.append(model)
        elif model.provider == ModelProvider.GOOGLE and has_key and not use_vertex:
            models.append(model)
        elif model.provider == ModelProvider.HOSTED_VLLM and api_base is not None:
            models.append(model)
            
    return models

def get_optimal_models( policy: Policy, include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None):
    """
    Selects the top models from the available list based on the user's policy.
    """
    # 1. Gather Available Models
    available_models = get_models(
        include_embedding=include_embedding, 
        use_vertex=use_vertex, 
        gemini_credentials_path=gemini_credentials_path, 
        api_base=api_base
    )
    
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

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def _generate_config_yaml(models: list[Model]):
    rand_id = random.randint(100000, 999999)
    config_filename = f"litellm_config_{rand_id}.yaml"
    config_list = []
    for model in models:
        # Use Model property instead of helper function
        # Ensure 'model' is an instance of Model enum or class
        if isinstance(model, str):
            model_obj = Model(model)
        else:
            model_obj = model
            
        env_var_name = model_obj.api_key_env_var
        api_key_val = f"os.environ/{env_var_name}" if env_var_name else None
        
        entry = {
            "model_name": model_obj.value,
            "litellm_params": {
                "model": model_obj.value,
                "api_key": api_key_val
            }
        }
        config_list.append(entry)
    yaml_structure = {"model_list": config_list}
    with open(config_filename, 'w') as f:
        yaml.dump(yaml_structure, f, default_flow_style=False, sort_keys=False)
    return config_filename

def fetch_dynamic_model_info(available_models: list[Model]):

    # only fetch dynamic info for models that have estimated value
    models = [model for model in available_models if model.use_endpoint]

    port = get_free_port()
    proxy_url = f"http://127.0.0.1:{port}" 
    config_filename = _generate_config_yaml(models)
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

    DYNAMIC_MODEL_INFO.update(dynamic_model_info)
    return dynamic_model_info
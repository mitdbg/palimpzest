import yaml, time, requests, subprocess

from palimpzest.constants import MODEL_CARDS, CuratedModel
from palimpzest.utils.model_helpers import predict_model_specs

CONFIG_FILENAME = "litellm_config.yaml"
PROXY_PORT = 4000
PROXY_URL = f"http://0.0.0.0:{PROXY_PORT}"
DYNAMIC_MODEL_INFO = {}

def _get_model_provider(model_name: str):
    model_name = model_name.lower().strip()
    if "/" in model_name:
        provider = model_name.split("/", 1)[0]
        return provider
    # fallback logic
    if model_name.startswith(("gpt-", "o1-", "dall-e", "text-embedding")):
        return "openai"
    if model_name.startswith("claude"):
        return "anthropic"
    # existing logic
    if "openai" in model_name.lower(): return "openai"
    if "anthropic" in model_name.lower(): return "anthropic"
    if "vertex_ai" in model_name.lower(): return "vertex_ai"
    if "gemini/" in model_name.lower(): return "gemini" # Google AI studio
    if "together_ai" in model_name.lower(): return "together_ai"
    # default
    return "unknown" # TODO: throw exception

# TODO: consider making model provider names into a dictionary
def _infer_api_key(model_name):
    model_provider = _get_model_provider(model_name)
    if model_provider == "openai": return "os.environ/OPENAI_API_KEY"
    elif model_provider == "anthropic": return "os.environ/ANTHROPIC_API_KEY"
    elif model_provider == "gemini": return "os.environ/GEMINI_API_KEY"
    elif model_provider == "together_ai": return "os.environ/TOGETHER_API_KEY"
    elif model_provider == "vertex_ai": return "os.environ/GOOGLE_APPLICATION_CREDENTIALS"
    # Expand the list of API keys
    return ""

def _generate_config_yaml(available_models, filename):
    config_list = []
    for model_str in available_models:
        entry = {
            "model_name": model_str,
            "litellm_params": {
                "model": model_str,
                "api_key": _infer_api_key(model_str)
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
    # TODO: exception logic
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
    
    def __init__(self, value, prediction=None):
        self.prediction = predict_model_specs(str)
        # prediction = {
        #     "usd_per_1m_input": 0.50,
        #     "usd_per_1m_output": 1.50,
        #     "usd_per_1m_audio_input": None, # Only set for multimodal
        #     "seconds_per_output_token": 0.02, # ~50 tokens/sec
        #     "mmlu_pro_score": 40.0,
        #     "tier": "Standard"
        # }
    
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
         return _get_model_provider() == "together_ai"
    
    def is_anthropic_model(self):
        return _get_model_provider() == "anthropic"
    
    def is_openai_model(self):
        return _get_model_provider() == "openai" or self.is_text_embedding_model()

    def is_vertex_model(self):
        return _get_model_provider() == "vertex_ai"

    def is_google_ai_studio_model(self):
        return _get_model_provider() == "gemini"

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
            CuratedModel.GEMINI_2_5_PRO.value,
            CuratedModel.GEMINI_2_5_FLASH.value,
            CuratedModel.GOOGLE_GEMINI_2_5_PRO.value,
            CuratedModel.GOOGLE_GEMINI_2_5_FLASH.value,
            CuratedModel.GOOGLE_GEMINI_2_5_FLASH_LITE.value,
            CuratedModel.CLAUDE_3_7_SONNET.value,
        }
        return self.value in known_reasoning_models
    
    # TODO: I think SONNET and HAIKU are vision-capable too
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

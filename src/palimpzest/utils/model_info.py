import yaml, time, requests, subprocess

from palimpzest.constants import MODEL_CARDS, CuratedModel
from palimpzest.utils.model_helpers import predict_model_specs

CONFIG_FILENAME = "litellm_config.yaml"
PROXY_PORT = 4000
PROXY_URL = f"http://0.0.0.0:{PROXY_PORT}"
DYNAMIC_MODEL_INFO = {}

def _get_model_provider(model_name: str) -> str:
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
    # 1. Handle explicit namespace (e.g., 'openai/gpt-4', 'aws/titan')
    # This is the standard convention for LangChain, LiteLLM, etc.
    if "/" in name_clean:
        return name_clean.split("/", 1)[0]
    # 2. Map known model families (prefixes) to providers
    # (Order doesn't strictly matter here as keys are unique enough, but specific > generic)
    family_map = {
        ("gpt-", "o1-", "dall-e", "text-embedding", "whisper"): "openai",
        ("claude",): "anthropic",
        ("gemini", "gemma", "palm"): "gemini", # Mapped to gemini (or google)
        ("llama",): "meta", 
        ("mistral", "mixtral"): "mistral",
        ("command",): "cohere",
    }
    for prefixes, provider in family_map.items():
        if name_clean.startswith(prefixes):
            return provider

    # 3. Fallback: Check for specific provider markers anywhere in the string
    # Useful for non-standard names like "hosted_vllm_llama_3"
    provider_markers = {
        "openai": "openai",
        "anthropic": "anthropic",
        "vertex_ai": "vertex_ai",
        "together_ai": "together_ai",
        "gemini": "gemini", 
        "azure": "azure",
        "aws": "aws",
    }

    for marker, provider in provider_markers.items():
        if marker in name_clean:
            return provider

    return "unknown"

import os

def _infer_api_key(model_name: str) -> str:
    model_provider = _get_model_provider(model_name)
    
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
    
    env_var = provider_to_env_var.get(model_provider)
    if env_var:
        return f"os.environ/{env_var}"
    
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
    
    def __init__(self):
        self.prediction = predict_model_specs(str)
    
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

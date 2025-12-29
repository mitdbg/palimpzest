import os, re

from palimpzest.constants import CuratedModel
from palimpzest.utils.model_info import Model
from typing import Dict, Any


def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None) -> list[Model]:
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
        vllm_models = [model for model in Model if model.is_vllm_model()]
        if not include_embedding:
            vllm_models = [
                model for model in vllm_models if not model.is_embedding_model()
            ]
        models.extend(vllm_models)

    return models

def predict_model_specs(full_model_id: str) -> Dict[str, Any]:
    """
    Predicts pricing (USD/1M tokens), latency (sec/token), and MMLU-Pro score
    based on the semantic naming of the model. 
    
    Logic decouples the provider (e.g., 'perplexity', 'together_ai') and looks 
    at the model slug (e.g., 'sonar-reasoning') to determine the tier.
    """
    
    # 1. Normalize: Remove provider prefix to focus on the model name
    # e.g., "together_ai/meta-llama/Llama-4-Maverick" -> "llama-4-maverick"
    model_slug = full_model_id.split('/')[-1].lower()
    
    # Defaults (Baseline: ~GPT-3.5 level)
    prediction = {
        "usd_per_1m_input": 0.50,
        "usd_per_1m_output": 1.50,
        "usd_per_1m_audio_input": None, # Only set for multimodal
        "seconds_per_output_token": 0.02, # ~50 tokens/sec
        "mmlu_pro_score": 40.0,
        "tier": "Standard"
    }

    # ==============================================================================
    # 1. REASONING / "THINKING" MODELS (Highest Cost, Highest Latency, Best Score)
    # Keywords: o1, o3, o4, r1 (DeepSeek), reasoning, thinking
    # ==============================================================================
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

    # ==============================================================================
    # 2. FUTURE FLAGSHIPS (GPT-5, Llama 4, Gemini 3, Opus 4/4.5)
    # Keywords: gpt-5, llama-4, gemini-3, opus-4, claude-4
    # ==============================================================================
    elif re.search(r'(gpt-5|llama-4|gemini-3|opus-4|sonnet-4|grok-4|mistral-large-3)', model_slug):
        prediction["tier"] = "Next-Gen Flagship"
        prediction["mmlu_pro_score"] = 92.0 # Theoretical future score
        prediction["seconds_per_output_token"] = 0.04 
        prediction["usd_per_1m_input"] = 5.00
        prediction["usd_per_1m_output"] = 15.00

    # ==============================================================================
    # 3. CURRENT FLAGSHIPS (GPT-4, Opus 3/3.5, Gemini 1.5/2.5 Pro, Llama 3.1 405B)
    # Keywords: gpt-4, opus, gemini.*pro, large, 405b, command-r-plus, grok-3
    # ==============================================================================
    elif re.search(r'(gpt-4|opus|gemini.*pro|large|405b|command-r-plus|grok-3)', model_slug):
        prediction["tier"] = "Current Flagship"
        prediction["mmlu_pro_score"] = 75.0
        prediction["seconds_per_output_token"] = 0.03
        prediction["usd_per_1m_input"] = 2.50
        prediction["usd_per_1m_output"] = 10.00
        
        # Special check for "Turbo/Flash" versions of flagships (Cheaper/Faster)
        if re.search(r'(turbo|flash|lite)', model_slug):
            prediction["tier"] += " (Optimized)"
            prediction["usd_per_1m_input"] = 0.15
            prediction["usd_per_1m_output"] = 0.60
            prediction["seconds_per_output_token"] = 0.01 # Fast (~100 tok/s)

    # ==============================================================================
    # 4. BALANCED / HIGH-MID (Sonnet, Llama 70B, Grok 2, Mistral Medium)
    # Keywords: sonnet, 70b, 90b, medium, grok-2, command-r (standard)
    # ==============================================================================
    elif re.search(r'(sonnet|70b|90b|medium|grok-2|command-r)', model_slug):
        prediction["tier"] = "Balanced"
        prediction["mmlu_pro_score"] = 65.0
        prediction["seconds_per_output_token"] = 0.015
        prediction["usd_per_1m_input"] = 3.00
        prediction["usd_per_1m_output"] = 15.00
        
        # Anthropic Sonnet 3.5/3.7 is exceptionally good, bump score
        if "sonnet" in model_slug:
             prediction["mmlu_pro_score"] = 78.0

    # ==============================================================================
    # 5. ECONOMY / SMALL (Haiku, Mini, 8B, 7B, 3B, Flash-Lite)
    # Keywords: haiku, mini, nano, 8b, 7b, small, micro
    # ==============================================================================
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

    # ==============================================================================
    # AUDIO / MULTIMODAL DETECTION
    # If the model is known to handle native audio, assign audio pricing
    # ==============================================================================
    # Keywords: 4o, omni, gemini (usually multimodal), audio
    if re.search(r'(4o|omni|gemini|audio)', model_slug):
        # Audio input is generally 2-5x text input price
        prediction["usd_per_1m_audio_input"] = prediction["usd_per_1m_input"] * 4.0

    return prediction

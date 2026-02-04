import os

from palimpzest.constants import Model


def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None, api_base: str | None = None) -> list[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
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


def use_reasoning_prompt(reasoning_effort: str | None) -> bool:
    """
    Determine whether to use the reasoning prompt based on the provided reasoning effort.
    By default, we use the reasoning prompt everywhere unless the reasoning_effort is in [None, "disable", "minimal", "low"].
    """
    return reasoning_effort not in [None, "disable", "minimal", "low"]


def resolve_reasoning_effort(model: Model, reasoning_effort: str | None) -> str | None:
    """
    Resolve the reasoning effort setting based on the model and provided reasoning effort.
    """
    # check that model is a reasoning model, throw an assertion error otherwise
    assert model.is_reasoning_model(), f"Model {model} is not a reasoning model. Should only use resolve_reasoning_effort with reasoning models."

    # if reasoning_effort is set to "default", set it to None to use model defaults
    if reasoning_effort == "default":
        reasoning_effort = None

    # translate reasoning_effort into model-specific settings
    if model.is_vertex_model() or model.is_google_ai_studio_model():
        if reasoning_effort is None and model in [Model.GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_PRO]:
            reasoning_effort = "low"
        elif reasoning_effort is None:
            reasoning_effort = "disable"
    elif model.is_openai_model():
        reasoning_effort = "low" if reasoning_effort in [None, "disable", "minimal", "low"] else reasoning_effort

    return reasoning_effort

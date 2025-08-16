import os

from palimpzest.constants import Model


def get_models(include_embedding: bool = False) -> list[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        openai_models = [model for model in Model if model.is_openai_model()]
        if not include_embedding:
            openai_models = [
                model for model in openai_models if not model.is_embedding_model()
            ]
        models.extend(openai_models)

    if os.getenv("TOGETHER_API_KEY") is not None:
        together_models = [model for model in Model if model.is_together_model()]
        if not include_embedding:
            together_models = [
                model for model in together_models if not model.is_embedding_model()
            ]
        models.extend(together_models)

    if os.getenv("ANTHROPIC_API_KEY") is not None:
        anthropic_models = [model for model in Model if model.is_anthropic_model()]
        if not include_embedding:
            anthropic_models = [
                model for model in anthropic_models if not model.is_embedding_model()
            ]
        models.extend(anthropic_models)

    if os.getenv("GEMINI_API_KEY") is not None:
        vertex_models = [model for model in Model if model.is_vertex_model()]
        if not include_embedding:
            vertex_models = [
                model for model in vertex_models if not model.is_embedding_model()
            ]
        models.extend(vertex_models)

    return models

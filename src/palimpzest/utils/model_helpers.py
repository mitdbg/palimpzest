import os

from palimpzest.constants import Model


def get_vision_models() -> list[Model]:
    """
    Return the set of vision models which the system has access to based on the set of environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        openai_vision_models = [
            model for model in Model
            if model.is_openai_model() and model.is_vision_model()
        ]
        models.extend(openai_vision_models)

    if os.getenv("TOGETHER_API_KEY") is not None:
        together_vision_models = [
            model for model in Model
            if model.is_together_model() and model.is_vision_model()
        ]
        models.extend(together_vision_models)

    return models


def get_models(include_vision: bool = False, include_embedding: bool = False) -> list[Model]:
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
        if not include_vision:
            together_models = [
                model for model in together_models if not model.is_vision_model()
            ]
        if not include_embedding:
            together_models = [
                model for model in together_models if not model.is_embedding_model()
            ]
        models.extend(together_models)

    if include_vision:
        vision_models = get_vision_models()
        models.extend(vision_models)

    return models

# The order is the priority of the model
TEXT_MODEL_PRIORITY = [
    # Model.o1,
    Model.GPT_4o,
    Model.GPT_4o_MINI,
    Model.LLAMA3_3_70B,
    Model.MIXTRAL,
    Model.DEEPSEEK_V3,
    Model.LLAMA3_2_3B,
    Model.LLAMA3_1_8B,
    Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
]

VISION_MODEL_PRIORITY = [
    Model.GPT_4o,
    Model.GPT_4o_MINI,
    Model.LLAMA3_2_90B_V,
]
def get_champion_model(available_models, vision=False):
    # Select appropriate priority list based on task
    model_priority = VISION_MODEL_PRIORITY if vision else TEXT_MODEL_PRIORITY

    # Return first available model from priority list
    for model in model_priority:
        if model in available_models:
            return model

    # If no suitable model found, raise informative error
    task_type = "vision" if vision else "text"
    raise Exception(
        f"No {task_type} models available to create physical plans!\n"
        "You must set at least one of the following environment variables:\n"
        "[OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]\n"
        f"Available models: {available_models}"
    )


def get_fallback_model(available_models, vision=False):
    return get_champion_model(available_models, vision)


def get_code_champion_model(available_models):
    # NOTE: for now, assume same champion as get_champion_model()
    return get_champion_model(available_models, vision=False)


def get_champion_model_name(available_models, vision=False):
    return get_champion_model(available_models, vision).value

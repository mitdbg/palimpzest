import os

from palimpzest.constants import Model


def get_vision_models() -> list[Model]:
    """
    Return the set of vision models which the system has access to based on the set of environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4o_V, Model.GPT_4o_MINI_V])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.LLAMA3_V])

    return models


def get_models(include_vision: bool = False) -> list[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4o, Model.GPT_4o_MINI])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.LLAMA3, Model.MIXTRAL])

    if include_vision:
        vision_models = get_vision_models()
        models.extend(vision_models)

    return models


# The order is the priority of the model
TEXT_MODEL_PRIORITY = [Model.GPT_4o, Model.GPT_4o_MINI, Model.LLAMA3, Model.MIXTRAL]

VISION_MODEL_PRIORITY = [Model.GPT_4o_V, Model.GPT_4o_MINI_V, Model.LLAMA3_V]


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

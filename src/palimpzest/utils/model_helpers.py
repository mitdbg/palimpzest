import os
from typing import List, Optional

from palimpzest.constants import Model


def get_vision_models() -> List[Model]:
    """
    Return the set of vision models which the system has access to based on the set of environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4o_V, Model.GPT_4o_MINI_V])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.LLAMA3_V])

    return models


def get_models(include_vision: Optional[bool] = False) -> List[Model]:
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


def get_champion_model(available_models, vision=False):
    champion_model = None

    # non-vision
    if not vision and Model.GPT_4o in available_models:
        champion_model = Model.GPT_4o
    elif not vision and Model.GPT_4o_MINI in available_models:
        champion_model = Model.GPT_4o_MINI
    elif not vision and Model.LLAMA3 in available_models:
        champion_model = Model.LLAMA3
    elif not vision and Model.MIXTRAL in available_models:
        champion_model = Model.MIXTRAL

    # vision
    elif vision and Model.GPT_4o_V in available_models:
        champion_model = Model.GPT_4o_V
    elif vision and Model.GPT_4o_MINI_V in available_models:
        champion_model = Model.GPT_4o_MINI_V
    elif vision and Model.LLAMA3_V in available_models:
        champion_model = Model.LLAMA3_V

    else:
        raise Exception(
            "No models available to create physical plans! You must set at least one of the following environment"
            " variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]\n"
            f"available_models: {available_models}"
        )

    return champion_model


def get_conventional_fallback_model(available_models, vision=False):
    return get_champion_model(available_models, vision)


def get_code_champion_model(available_models):
    # NOTE: for now, assume same champion as get_champion_model()
    return get_champion_model(available_models, vision=False)


def get_champion_model_name(available_models, vision=False):
    return get_champion_model(available_models, vision).value

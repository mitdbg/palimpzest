from palimpzest.constants import Model

from typing import List, Optional

import os


def getVisionModels() -> List[Model]:
    """
    Return the set of vision models which the system has access to based on the set of environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4o_V, Model.GPT_4o_MINI_V])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.LLAMA3_V])

    return models


def getModels(include_vision: Optional[bool] = False) -> List[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4o, Model.GPT_4o_MINI])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.LLAMA3, Model.MIXTRAL])

    if include_vision:
        vision_models = getVisionModels()
        models.extend(vision_models)

    return models


def getChampionModel(available_models, vision=False):
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
        raise Exception(f"No models available to create physical plans! available_models: {available_models}")

    return champion_model


def getConventionalFallbackModel(available_models, vision=False):
    return getChampionModel(available_models, vision)


def getCodeChampionModel(available_models):
    # NOTE: for now, assume same champion as getChampionModel()
    return getChampionModel(available_models, vision=False)


def getChampionModelName(available_models, vision=False):
    return getChampionModel(available_models, vision=vision).value

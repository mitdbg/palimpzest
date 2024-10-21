import os
from typing import List, Optional

from palimpzest.constants import Model


def getVisionModels() -> List[Model]:
    """
    Return the set of vision models which the system has access to based on the set of environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_4V])

    # NOTE: not using free Gemini vision model at the moment due to quality issues

    return models


def getModels(include_vision: Optional[bool] = False) -> List[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    if os.getenv("OPENAI_API_KEY") is not None:
        models.extend([Model.GPT_3_5, Model.GPT_4])

    if os.getenv("TOGETHER_API_KEY") is not None:
        models.extend([Model.MIXTRAL])

    # if os.getenv("GOOGLE_API_KEY") is not None:
    #     models.extend([Model.GEMINI_1])

    if include_vision:
        vision_models = getVisionModels()
        models.extend(vision_models)

    return models

def getChampionModel():
    champion_model = None
    if os.environ.get("OPENAI_API_KEY", None) is not None:
        champion_model = Model.GPT_4
    elif os.environ.get("TOGETHER_API_KEY", None) is not None:
        champion_model = Model.MIXTRAL
    elif os.environ.get("GOOGLE_API_KEY", None) is not None:
        champion_model = Model.GEMINI_1
    else:
        raise Exception("No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]")

    return champion_model

def getConventionalFallbackModel():
    fallback_model = None
    if os.environ.get("OPENAI_API_KEY", None) is not None:
        fallback_model = Model.GPT_3_5
    elif os.environ.get("TOGETHER_API_KEY", None) is not None:
        fallback_model = Model.MIXTRAL
    elif os.environ.get("GOOGLE_API_KEY", None) is not None:
        fallback_model = Model.GEMINI_1
    else:
        raise Exception("No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]")

    return fallback_model

def getCodeChampionModel():
    # NOTE: for now, assume same champion as getChampionModel()
    return getChampionModel()

def getChampionModelName():
    return getChampionModel().value

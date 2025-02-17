import warnings
from typing import Any, Dict, List

import together
from together.legacy.base import API_KEY_WARNING, deprecated


class Models:
    @classmethod
    @deprecated  # type: ignore
    def list(
        cls,
    ) -> List[Dict[str, Any]]:
        """Legacy model list function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return [item.model_dump(exclude_none=True) for item in client.models.list()]

    @classmethod
    @deprecated  # type: ignore
    def info(
        cls,
        model: str,
    ) -> Dict[str, Any]:
        """Legacy model info function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        model_list = client.models.list()

        for item in model_list:
            if item.id == model:
                return item.model_dump(exclude_none=True)

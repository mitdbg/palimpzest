import warnings
from typing import Any, Dict

import together
from together.legacy.base import API_KEY_WARNING, deprecated


class Embeddings:
    @classmethod
    @deprecated  # type: ignore
    def create(
        cls,
        input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Legacy embeddings function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.embeddings.create(input=input, **kwargs).model_dump(
            exclude_none=True
        )

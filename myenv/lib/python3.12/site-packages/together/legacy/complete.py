from __future__ import annotations

import warnings
from typing import Any, AsyncGenerator, Dict, Iterator

import together
from together.legacy.base import API_KEY_WARNING, deprecated
from together.types import CompletionChunk, CompletionResponse


class Complete:
    @classmethod
    @deprecated  # type: ignore
    def create(
        cls,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Legacy completion function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        result = client.completions.create(prompt=prompt, stream=False, **kwargs)

        assert isinstance(result, CompletionResponse)

        return result.model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def create_streaming(
        cls,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Legacy streaming completion function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return (
            token.model_dump(exclude_none=True)  # type: ignore
            for token in client.completions.create(prompt=prompt, stream=True, **kwargs)
        )


class Completion:
    @classmethod
    @deprecated  # type: ignore
    def create(
        cls,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResponse | Iterator[CompletionChunk]:
        """Completion function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.completions.create(prompt=prompt, **kwargs)


class AsyncComplete:
    @classmethod
    @deprecated  # type: ignore
    async def create(
        cls,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[CompletionChunk, None]:
        """Async completion function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.AsyncTogether(api_key=api_key)

        return await client.completions.create(prompt=prompt, **kwargs)

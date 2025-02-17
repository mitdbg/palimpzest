from __future__ import annotations

from typing import List, Any

from together.abstract import api_requestor
from together.together_response import TogetherResponse
from together.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    TogetherClient,
    TogetherRequest,
)


class Embeddings:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def create(
        self,
        *,
        input: str | List[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Method to generate completions based on a given prompt using a specified model.

        Args:
            input (str | List[str]): A string or list of strings to embed
            model (str): The name of the model to query.

        Returns:
            EmbeddingResponse: Object containing embeddings
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = EmbeddingRequest(
            input=input,
            model=model,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url="embeddings",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return EmbeddingResponse(**response.data)


class AsyncEmbeddings:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        input: str | List[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Async method to generate completions based on a given prompt using a specified model.

        Args:
            input (str | List[str]): A string or list of strings to embed
            model (str): The name of the model to query.

        Returns:
            EmbeddingResponse: Object containing embeddings
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = EmbeddingRequest(
            input=input,
            model=model,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url="embeddings",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return EmbeddingResponse(**response.data)

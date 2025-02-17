from __future__ import annotations

from typing import List

from together.abstract import api_requestor
from together.together_response import TogetherResponse
from together.types import (
    ModelObject,
    TogetherClient,
    TogetherRequest,
)


class Models:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def list(
        self,
    ) -> List[ModelObject]:
        """
        Method to return list of models on the API

        Returns:
            List[ModelObject]: List of model objects
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="GET",
                url="models",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)
        assert isinstance(response.data, list)

        return [ModelObject(**model) for model in response.data]


class AsyncModels:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def list(
        self,
    ) -> List[ModelObject]:
        """
        Async method to return list of models on API

        Returns:
            List[ModelObject]: List of model objects
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="GET",
                url="models",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)
        assert isinstance(response.data, list)

        return [ModelObject(**model) for model in response.data]

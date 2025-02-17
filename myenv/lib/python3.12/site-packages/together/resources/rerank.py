from __future__ import annotations

from typing import List, Dict, Any

from together.abstract import api_requestor
from together.together_response import TogetherResponse
from together.types import (
    RerankRequest,
    RerankResponse,
    TogetherClient,
    TogetherRequest,
)


class Rerank:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: int | None = None,
        return_documents: bool = False,
        rank_fields: List[str] | None = None,
        **kwargs: Any,
    ) -> RerankResponse:
        """
        Method to generate completions based on a given prompt using a specified model.

        Args:
            model (str): The name of the model to query.
            query (str): The input query or list of queries to rerank.
            documents (List[str] | List[Dict[str, Any]]): List of documents to be reranked.
            top_n (int | None): Number of top results to return.
            return_documents (bool): Flag to indicate whether to return documents.
            rank_fields (List[str] | None): Fields to be used for ranking the documents.

        Returns:
            RerankResponse: Object containing reranked scores and documents
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = RerankRequest(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            rank_fields=rank_fields,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url="rerank",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return RerankResponse(**response.data)


class AsyncRerank:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: int | None = None,
        return_documents: bool = False,
        rank_fields: List[str] | None = None,
        **kwargs: Any,
    ) -> RerankResponse:
        """
        Async method to generate completions based on a given prompt using a specified model.

        Args:
            model (str): The name of the model to query.
            query (str): The input query or list of queries to rerank.
            documents (List[str] | List[Dict[str, Any]]): List of documents to be reranked.
            top_n (int | None): Number of top results to return.
            return_documents (bool): Flag to indicate whether to return documents.
            rank_fields (List[str] | None): Fields to be used for ranking the documents.

        Returns:
            RerankResponse: Object containing reranked scores and documents
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = RerankRequest(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            rank_fields=rank_fields,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url="rerank",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return RerankResponse(**response.data)

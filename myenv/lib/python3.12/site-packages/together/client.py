from __future__ import annotations

import os
from typing import Dict

from together import resources
from together.constants import BASE_URL, MAX_RETRIES, TIMEOUT_SECS
from together.error import AuthenticationError
from together.types import TogetherClient
from together.utils import enforce_trailing_slash


class Together:
    completions: resources.Completions
    chat: resources.Chat
    embeddings: resources.Embeddings
    files: resources.Files
    images: resources.Images
    models: resources.Models
    fine_tuning: resources.FineTuning
    rerank: resources.Rerank

    # client options
    client: TogetherClient

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        supplied_headers: Dict[str, str] | None = None,
    ) -> None:
        """Construct a new synchronous together client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `TOGETHER_API_KEY`
        - `base_url` from `TOGETHER_BASE_URL`
        """

        # get api key
        if not api_key:
            api_key = os.environ.get("TOGETHER_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "The api_key client option must be set either by passing api_key to the client or by setting the "
                "TOGETHER_API_KEY environment variable"
            )

        # get base url
        if not base_url:
            base_url = os.environ.get("TOGETHER_BASE_URL")

        if not base_url:
            base_url = BASE_URL

        if timeout is None:
            timeout = TIMEOUT_SECS

        if max_retries is None:
            max_retries = MAX_RETRIES

        # TogetherClient object
        self.client = TogetherClient(
            api_key=api_key,
            base_url=enforce_trailing_slash(base_url),
            timeout=timeout,
            max_retries=max_retries,
            supplied_headers=supplied_headers,
        )

        self.completions = resources.Completions(self.client)
        self.chat = resources.Chat(self.client)
        self.embeddings = resources.Embeddings(self.client)
        self.files = resources.Files(self.client)
        self.images = resources.Images(self.client)
        self.models = resources.Models(self.client)
        self.fine_tuning = resources.FineTuning(self.client)
        self.rerank = resources.Rerank(self.client)


class AsyncTogether:
    completions: resources.AsyncCompletions
    chat: resources.AsyncChat
    embeddings: resources.AsyncEmbeddings
    files: resources.AsyncFiles
    images: resources.AsyncImages
    models: resources.AsyncModels
    fine_tuning: resources.AsyncFineTuning
    rerank: resources.AsyncRerank

    # client options
    client: TogetherClient

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        supplied_headers: Dict[str, str] | None = None,
    ) -> None:
        """Construct a new async together client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `TOGETHER_API_KEY`
        - `base_url` from `TOGETHER_BASE_URL`
        """

        # get api key
        if not api_key:
            api_key = os.environ.get("TOGETHER_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "The api_key client option must be set either by passing api_key to the client or by setting the "
                "TOGETHER_API_KEY environment variable"
            )

        # get base url
        if not base_url:
            base_url = os.environ.get("TOGETHER_BASE_URL")

        if not base_url:
            base_url = BASE_URL

        if timeout is None:
            timeout = TIMEOUT_SECS

        if max_retries is None:
            max_retries = MAX_RETRIES

        # TogetherClient object
        self.client = TogetherClient(
            api_key=api_key,
            base_url=enforce_trailing_slash(base_url),
            timeout=timeout,
            max_retries=max_retries,
            supplied_headers=supplied_headers,
        )

        self.completions = resources.AsyncCompletions(self.client)
        self.chat = resources.AsyncChat(self.client)
        self.embeddings = resources.AsyncEmbeddings(self.client)
        self.files = resources.AsyncFiles(self.client)
        self.images = resources.AsyncImages(self.client)
        self.models = resources.AsyncModels(self.client)
        self.fine_tuning = resources.AsyncFineTuning(self.client)
        self.rerank = resources.AsyncRerank(self.client)


Client = Together

AsyncClient = AsyncTogether

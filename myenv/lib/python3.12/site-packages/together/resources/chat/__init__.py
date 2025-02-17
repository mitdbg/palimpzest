from functools import cached_property

from together.resources.chat.completions import AsyncChatCompletions, ChatCompletions
from together.types import (
    TogetherClient,
)


class Chat:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    @cached_property
    def completions(self) -> ChatCompletions:
        return ChatCompletions(self._client)


class AsyncChat:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    @cached_property
    def completions(self) -> AsyncChatCompletions:
        return AsyncChatCompletions(self._client)

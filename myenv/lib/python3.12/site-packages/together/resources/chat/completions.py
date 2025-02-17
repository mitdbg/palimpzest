from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Iterator, List

from together.abstract import api_requestor
from together.together_response import TogetherResponse
from together.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TogetherClient,
    TogetherRequest,
)


class ChatCompletions:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        min_p: float | None = None,
        logit_bias: Dict[str, float] | None = None,
        seed: int | None = None,
        stream: bool = False,
        logprobs: int | None = None,
        echo: bool | None = None,
        n: int | None = None,
        safety_model: str | None = None,
        response_format: Dict[str, str | Dict[str, Any]] | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, str | Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunk]:
        """
        Method to generate completions based on a given prompt using a specified model.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the format
                `[{"role": together.types.chat_completions.MessageRole, "content": TEXT}, ...]`
            model (str): The name of the model to query.
            max_tokens (int, optional): The maximum number of tokens to generate.
                Defaults to 512.
            stop (List[str], optional): List of strings at which to stop generation.
                Defaults to None.
            temperature (float, optional): A decimal number that determines the degree of randomness in the response.
                Defaults to None.
            top_p (float, optional): The top_p (nucleus) parameter is used to dynamically adjust the number
                    of choices for each predicted token based on the cumulative probabilities.
                Defaults to None.
            top_k (int, optional): The top_k parameter is used to limit the number of choices for the
                    next predicted word or token.
                Defaults to None.
            repetition_penalty (float, optional): A number that controls the diversity of generated text
                    by reducing the likelihood of repeated sequences. Higher values decrease repetition.
                Defaults to None.
            presence_penalty (float, optional): A number that controls the likelihood of tokens based on if they have
                    appeared in the text. Positive values decrease the likelihood of repeated tokens or phrases.
                    Must be in the range [-2, 2].
                Defaults to None.
            frequency_penalty (float, optional): A number that controls the likelihood of tokens based on the frequency
                    of their appearance in the text. Positive decrease the likelihood of repeated tokens or phrases.
                    Must be in the range [-2, 2].
                Defaults to None.
            min_p (float, optional): A number that controls the minimum percentage value that a token must reach to
                be considered during sampling.
                Must be in the range [0, 1].
                Defaults to None.
            logit_bias (Dict[str, float], optional): A dictionary of tokens and their bias values that modify the
                likelihood of specific tokens being sampled. Bias values must be in the range [-100, 100].
                Defaults to None.
            seed (int, optional): A seed value to use for reproducibility.
            stream (bool, optional): Flag indicating whether to stream the generated completions.
                Defaults to False.
            logprobs (int, optional): Number of top-k logprobs to return
                Defaults to None.
            echo (bool, optional): Echo prompt in output. Can be used with logprobs to return prompt logprobs.
                Defaults to None.
            n (int, optional): Number of completions to generate. Setting to None will return a single generation.
                Defaults to None.
            safety_model (str, optional): A moderation model to validate tokens. Choice between available moderation
                    models found [here](https://docs.together.ai/docs/inference-models#moderation-models).
                Defaults to None.
            response_format (Dict[str, Any], optional): An object specifying the format that the model must output.
                Defaults to None.
            tools (Dict[str, str | Dict[str, str | Dict[str, Any]]], optional): A list of tools the model may call.
                    Currently, only functions are supported as a tool.
                    Use this to provide a list of functions the model may generate JSON inputs for.
                Defaults to None
            tool_choice: Controls which (if any) function is called by the model. auto means the model can pick
                    between generating a message or calling a function. Specifying a particular function
                    via {"type": "function", "function": {"name": "my_function"}} forces the model to call that function.
                    Sets to `auto` if None.
                Defaults to None.

        Returns:
            ChatCompletionResponse | Iterator[ChatCompletionChunk]: Object containing the completions
            or an iterator over completion chunks.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = ChatCompletionRequest(
            model=model,
            messages=messages,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            min_p=min_p,
            logit_bias=logit_bias,
            seed=seed,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            n=n,
            safety_model=safety_model,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url="chat/completions",
                params=parameter_payload,
            ),
            stream=stream,
        )

        if stream:
            # must be an iterator
            assert not isinstance(response, TogetherResponse)
            return (ChatCompletionChunk(**line.data) for line in response)
        assert isinstance(response, TogetherResponse)
        return ChatCompletionResponse(**response.data)


class AsyncChatCompletions:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        min_p: float | None = None,
        logit_bias: Dict[str, float] | None = None,
        seed: int | None = None,
        stream: bool = False,
        logprobs: int | None = None,
        echo: bool | None = None,
        n: int | None = None,
        safety_model: str | None = None,
        response_format: Dict[str, Any] | None = None,
        tools: Dict[str, str | Dict[str, str | Dict[str, Any]]] | None = None,
        tool_choice: str | Dict[str, str | Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None] | ChatCompletionResponse:
        """
        Async method to generate completions based on a given prompt using a specified model.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the format
                `[{"role": together.types.chat_completions.MessageRole, "content": TEXT}, ...]`
            model (str): The name of the model to query.
            max_tokens (int, optional): The maximum number of tokens to generate.
                Defaults to 512.
            stop (List[str], optional): List of strings at which to stop generation.
                Defaults to None.
            temperature (float, optional): A decimal number that determines the degree of randomness in the response.
                Defaults to None.
            top_p (float, optional): The top_p (nucleus) parameter is used to dynamically adjust the number
                    of choices for each predicted token based on the cumulative probabilities.
                Defaults to None.
            top_k (int, optional): The top_k parameter is used to limit the number of choices for the
                    next predicted word or token.
                Defaults to None.
            repetition_penalty (float, optional): A number that controls the diversity of generated text
                    by reducing the likelihood of repeated sequences. Higher values decrease repetition.
                Defaults to None.
            presence_penalty (float, optional): A number that controls the likelihood of tokens based on if they have
                    appeared in the text. Positive values decrease the likelihood of repeated tokens or phrases.
                    Must be in the range [-2, 2].
                Defaults to None.
            frequency_penalty (float, optional): A number that controls the likelihood of tokens based on the frequency
                    of their appearance in the text. Positive decrease the likelihood of repeated tokens or phrases.
                    Must be in the range [-2, 2].
                Defaults to None.
            min_p (float, optional): A number that controls the minimum percentage value that a token must reach to
                be considered during sampling.
                Must be in the range [0, 1].
                Defaults to None.
            logit_bias (Dict[str, float], optional): A dictionary of tokens and their bias values that modify the
                likelihood of specific tokens being sampled. Bias values must be in the range [-100, 100].
                Defaults to None.
            seed (int, optional): A seed value to use for reproducibility.
            stream (bool, optional): Flag indicating whether to stream the generated completions.
                Defaults to False.
            logprobs (int, optional): Number of top-k logprobs to return
                Defaults to None.
            echo (bool, optional): Echo prompt in output. Can be used with logprobs to return prompt logprobs.
                Defaults to None.
            n (int, optional): Number of completions to generate. Setting to None will return a single generation.
                Defaults to None.
            safety_model (str, optional): A moderation model to validate tokens. Choice between available moderation
                    models found [here](https://docs.together.ai/docs/inference-models#moderation-models).
                Defaults to None.
            response_format (Dict[str, Any], optional): An object specifying the format that the model must output.
                Defaults to None.
            tools (Dict[str, str | Dict[str, str | Dict[str, Any]]], optional): A list of tools the model may call.
                    Currently, only functions are supported as a tool.
                    Use this to provide a list of functions the model may generate JSON inputs for.
                Defaults to None
            tool_choice: Controls which (if any) function is called by the model. auto means the model can pick
                    between generating a message or calling a function. Specifying a particular function
                    via {"type": "function", "function": {"name": "my_function"}} forces the model to call that function.
                    Sets to `auto` if None.
                Defaults to None.

        Returns:
            AsyncGenerator[ChatCompletionChunk, None] | ChatCompletionResponse: Object containing the completions
            or an iterator over completion chunks.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = ChatCompletionRequest(
            model=model,
            messages=messages,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            min_p=min_p,
            logit_bias=logit_bias,
            seed=seed,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            n=n,
            safety_model=safety_model,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url="chat/completions",
                params=parameter_payload,
            ),
            stream=stream,
        )

        if stream:
            # must be an iterator
            assert not isinstance(response, TogetherResponse)
            return (ChatCompletionChunk(**line.data) async for line in response)
        assert isinstance(response, TogetherResponse)
        return ChatCompletionResponse(**response.data)

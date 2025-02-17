from __future__ import annotations

from typing import Any

from together.abstract import api_requestor
from together.together_response import TogetherResponse
from together.types import (
    ImageRequest,
    ImageResponse,
    TogetherClient,
    TogetherRequest,
)


class Images:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = 20,
        seed: int | None = None,
        n: int | None = 1,
        height: int | None = 1024,
        width: int | None = 1024,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Method to generate images based on a given prompt using a specified model.

        Args:
            prompt (str): A description of the desired images. Maximum length varies by model.

            model (str, optional): The model to use for image generation.

            steps (int, optional): Number of generation steps. Defaults to 20

            seed (int, optional): Seed used for generation. Can be used to reproduce image generations.
                Defaults to None.

            n (int, optional): Number of image results to generate. Defaults to 1.

            height (int, optional): Height of the image to generate in number of pixels. Defaults to 1024

            width (int, optional): Width of the image to generate in number of pixels. Defaults to 1024

            negative_prompt (str, optional): The prompt or prompts not to guide the image generation.
                Defaults to None

            image_base64: (str, optional): Reference image used for generation. Defaults to None.

        Returns:
            ImageResponse: Object containing image data
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = ImageRequest(
            prompt=prompt,
            model=model,
            steps=steps,
            seed=seed,
            n=n,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url="images/generations",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return ImageResponse(**response.data)


class AsyncImages:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = 20,
        seed: int | None = None,
        n: int | None = 1,
        height: int | None = 1024,
        width: int | None = 1024,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Async method to generate images based on a given prompt using a specified model.

        Args:
            prompt (str): A description of the desired images. Maximum length varies by model.

            model (str, optional): The model to use for image generation.

            steps (int, optional): Number of generation steps. Defaults to 20

            seed (int, optional): Seed used for generation. Can be used to reproduce image generations.
                Defaults to None.

            n (int, optional): Number of image results to generate. Defaults to 1.

            height (int, optional): Height of the image to generate in number of pixels. Defaults to 1024

            width (int, optional): Width of the image to generate in number of pixels. Defaults to 1024

            negative_prompt (str, optional): The prompt or prompts not to guide the image generation.
                Defaults to None

            image_base64: (str, optional): Reference image used for generation. Defaults to None.

        Returns:
            ImageResponse: Object containing image data
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = ImageRequest(
            prompt=prompt,
            model=model,
            steps=steps,
            seed=seed,
            n=n,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            **kwargs,
        ).model_dump(exclude_none=True)

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url="images/generations",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return ImageResponse(**response.data)

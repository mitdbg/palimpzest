# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=bad-continuation, line-too-long, protected-access
"""Classes for working with vision models."""

import base64
import dataclasses

import pathlib
import typing
from typing import Iterator, List, Literal, Optional, Union

from google.generativeai import client
from google.generativeai.types import image_types

from google.generativeai import files
from google.generativeai import protos
from google.generativeai import operations

ImageAspectRatio = Literal["1:1", "9:16", "16:9", "4:3", "3:4"]
IMAGE_ASPECT_RATIOS = ImageAspectRatio.__args__  # type: ignore

VideoAspectRatio = Literal["9:16", "16:9"]
VIDEO_ASPECT_RATIOS = VideoAspectRatio.__args__  # type: ignore

OutputMimeType = Literal["image/png", "image/jpeg"]
OUTPUT_MIME_TYPES = OutputMimeType.__args__  # type: ignore

SafetyFilterLevel = Literal["block_low_and_above", "block_medium_and_above", "block_only_high"]
SAFETY_FILTER_LEVELS = SafetyFilterLevel.__args__  # type: ignore

PersonGeneration = Literal["dont_allow", "allow_adult"]
PERSON_GENERATIONS = PersonGeneration.__args__  # type: ignore


class ImageGenerationModel:
    """Generates images from text prompt.

    Examples::

        model = ImageGenerationModel.from_pretrained("imagegeneration@002")
        response = model.generate_images(
            prompt="Astronaut riding a horse",
            # Optional:
            number_of_images=1,
        )
        response[0].show()
        response[0].save("image1.png")
    """

    def __init__(self, model_id: str):
        if not model_id.startswith("models"):
            model_id = f"models/{model_id}"
        self.model_name = model_id
        self._client = None

    @classmethod
    def from_pretrained(cls, model_name: str = "imagen-3.0-generate-001"):
        """For vertex compatibility"""
        return cls(model_name)

    def _generate_images(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        number_of_images: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        guidance_scale: Optional[float] = None,
        output_mime_type: Optional[OutputMimeType] = None,
        compression_quality: Optional[float] = None,
        language: Optional[str] = None,
        safety_filter_level: Optional[SafetyFilterLevel] = None,
        person_generation: Optional[PersonGeneration] = None,
    ) -> "ImageGenerationResponse":
        """Generates images from text prompt.

        Args:
            prompt: Text prompt for the image.
            negative_prompt: A description of what you want to omit in the generated
              images.
            number_of_images: Number of images to generate. Range: 1..8.
            width: Width of the image. One of the sizes must be 256 or 1024.
            height: Height of the image. One of the sizes must be 256 or 1024.
            aspect_ratio: Aspect ratio for the image. Supported values are:
                * 1:1 - Square image
                * 9:16 - Portait image
                * 16:9 - Landscape image
                * 4:3 - Landscape, desktop ratio.
                * 3:4 - Portrait, desktop ratio
            guidance_scale: Controls the strength of the prompt. Suggested values
              are - * 0-9 (low strength) * 10-20 (medium strength) * 21+ (high
              strength)
            output_mime_type: Which image format should the output be saved as.
              Supported values: * image/png: Save as a PNG image * image/jpeg: Save
              as a JPEG image
            compression_quality: Level of compression if the output mime type is
              selected to be image/jpeg. Float between 0 to 100
            language: Language of the text prompt for the image. Default: None.
              Supported values are `"en"` for English, `"hi"` for Hindi, `"ja"` for
              Japanese, `"ko"` for Korean, and `"auto"` for automatic language
              detection.
            safety_filter_level: Adds a filter level to Safety filtering. Supported
              values are:
              * "block_most" : Strongest filtering level, most strict blocking
              * "block_some" : Block some problematic prompts and responses
              * "block_few" : Block fewer problematic prompts and responses
            person_generation: Allow generation of people by the model Supported
              values are:
              * "dont_allow" : Block generation of people
              * "allow_adult" : Generate adults, but not children

        Returns:
            An `ImageGenerationResponse` object.
        """
        if self._client is None:
            self._client = client.get_default_prediction_client()
        # Note: Only a single prompt is supported by the service.
        instance = {"prompt": prompt}
        shared_generation_parameters = {
            "prompt": prompt,
            # b/295946075 The service stopped supporting image sizes.
            # "width": width,
            # "height": height,
            "number_of_images_in_batch": number_of_images,
        }

        parameters = {}
        max_size = max(width or 0, height or 0) or None
        if aspect_ratio is not None:
            if aspect_ratio not in IMAGE_ASPECT_RATIOS:
                raise ValueError(f"aspect_ratio not in {IMAGE_ASPECT_RATIOS}")
            parameters["aspectRatio"] = aspect_ratio
        elif max_size:
            # Note: The size needs to be a string
            parameters["sampleImageSize"] = str(max_size)
            if height is not None and width is not None and height != width:
                parameters["aspectRatio"] = f"{width}:{height}"

        parameters["sampleCount"] = number_of_images
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
            shared_generation_parameters["negative_prompt"] = negative_prompt

        if guidance_scale is not None:
            parameters["guidanceScale"] = guidance_scale
            shared_generation_parameters["guidance_scale"] = guidance_scale

        if language is not None:
            parameters["language"] = language
            shared_generation_parameters["language"] = language

        parameters["outputOptions"] = {}
        if output_mime_type is not None:
            if output_mime_type not in OUTPUT_MIME_TYPES:
                raise ValueError(f"output_mime_type not in {OUTPUT_MIME_TYPES}")
            parameters["outputOptions"]["mimeType"] = output_mime_type
            shared_generation_parameters["mime_type"] = output_mime_type

        if compression_quality is not None:
            parameters["outputOptions"]["compressionQuality"] = compression_quality
            shared_generation_parameters["compression_quality"] = compression_quality

        if safety_filter_level is not None:
            if safety_filter_level not in SAFETY_FILTER_LEVELS:
                raise ValueError(f"safety_filter_level not in {SAFETY_FILTER_LEVELS}")
            parameters["safetySetting"] = safety_filter_level
            shared_generation_parameters["safety_filter_level"] = safety_filter_level

        if person_generation is not None:
            parameters["personGeneration"] = person_generation
            shared_generation_parameters["person_generation"] = person_generation

        response = self._client.predict(
            model=self.model_name, instances=[instance], parameters=parameters
        )

        generated_images: List[image_types.GeneratedImage] = []
        for idx, prediction in enumerate(response.predictions):
            generation_parameters = dict(shared_generation_parameters)
            generation_parameters["index_of_image_in_batch"] = idx
            encoded_bytes = prediction.get("bytesBase64Encoded")
            generated_image = image_types.GeneratedImage(
                image_bytes=base64.b64decode(encoded_bytes) if encoded_bytes else None,
                generation_parameters=generation_parameters,
            )
            generated_images.append(generated_image)

        return ImageGenerationResponse(images=generated_images)

    def generate_images(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        number_of_images: int = 1,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        guidance_scale: Optional[float] = None,
        language: Optional[str] = None,
        safety_filter_level: Optional[SafetyFilterLevel] = None,
        person_generation: Optional[PersonGeneration] = None,
    ) -> "ImageGenerationResponse":
        """Generates images from text prompt.

        Args:
            prompt: Text prompt for the image.
            negative_prompt: A description of what you want to omit in the generated
                images.
            number_of_images: Number of images to generate. Range: 1..8.
            aspect_ratio: Changes the aspect ratio of the generated image Supported
                values are:
                * "1:1" : 1:1 aspect ratio
                * "9:16" : 9:16 aspect ratio
                * "16:9" : 16:9 aspect ratio
                * "4:3" : 4:3 aspect ratio
                * "3:4" : 3:4 aspect_ratio
            guidance_scale: Controls the strength of the prompt. Suggested values are:
                * 0-9 (low strength)
                * 10-20 (medium strength)
                * 21+ (high strength)
            language: Language of the text prompt for the image. Default: None.
                Supported values are `"en"` for English, `"hi"` for Hindi, `"ja"`
                for Japanese, `"ko"` for Korean, and `"auto"` for automatic language
                detection.
            safety_filter_level: Adds a filter level to Safety filtering. Supported
                values are:
                * "block_most" : Strongest filtering level, most strict
                blocking
                * "block_some" : Block some problematic prompts and responses
                * "block_few" : Block fewer problematic prompts and responses
            person_generation: Allow generation of people by the model Supported
                values are:
                * "dont_allow" : Block generation of people
                * "allow_adult" : Generate adults, but not children
        Returns:
            An `ImageGenerationResponse` object.
        """
        return self._generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            guidance_scale=guidance_scale,
            language=language,
            safety_filter_level=safety_filter_level,
            person_generation=person_generation,
        )


@dataclasses.dataclass
class ImageGenerationResponse:
    """Image generation response.

    Attributes:
        images: The list of generated images.
    """

    __module__ = "vertexai.preview.vision_models"

    images: List[image_types.GeneratedImage]

    def __iter__(self) -> typing.Iterator[image_types.GeneratedImage]:
        """Iterates through the generated images."""
        yield from self.images

    def __getitem__(self, idx: int) -> image_types.GeneratedImage:
        """Gets the generated image by index."""
        return self.images[idx]


class VideoGenerationModel:
    """Generates videos from a text prompt.

    Examples:

    ```
    model = VideoGenerationModel("veo-001-preview-0815")
    response = model.generate_videos(
        prompt="Astronaut riding a horse",
    )
    response[0].save("video.mp4")
    ```
    """

    def __init__(self, model_id: str):
        if not model_id.startswith("models"):
            model_id = f"models/{model_id}"
        self.model_name = model_id
        self._client = None

    @classmethod
    def from_pretrained(cls, model_name: str):
        """For vertex compatibility"""
        return cls(model_name)

    def _generate_videos(
        self,
        prompt: str,
        *,
        image: Union[pathlib.Path, image_types.ImageType] = None,
        aspect_ratio: Optional[VideoAspectRatio] = None,
        person_generation: Optional[PersonGeneration] = None,
        number_of_videos: int = 4,
    ) -> "VideoGenerationResponse":
        """Generates videos from a text prompt.

        Args:
            prompt: Text prompt for the video.
            aspect_ratio: Aspect ratio for the video. Supported values are:
                * 9:16 - Portait
                * 16:9 - Landscape
            person_generation: Allow generation of people by the model.
              Supported values are:
              * "dont_allow" : Block generation of people
              * "allow_adult" : Generate adults, but not children
            number_of_videos: Number of videos to generate. Maximum 4.
        Returns:
            An `VideoGenerationResponse` object.
        """
        if self._client is None:
            self._client = client.get_default_prediction_client()
        # Note: Only a single prompt is supported by the service.

        instance = {}
        if prompt is not None:
            instance.update({"prompt": prompt})
        if image is not None:
            img = image_types.to_image(image)
            instance.update(
                {
                    "image": {
                        "bytesBase64Encoded": base64.b64encode(img._loaded_bytes).decode(),
                        "mimeType": img._mime_type,
                    }
                }
            )

        parameters = {}

        if aspect_ratio is not None:
            if aspect_ratio not in VIDEO_ASPECT_RATIOS:
                raise ValueError(f"aspect_ratio not in {VIDEO_ASPECT_RATIOS}")
            parameters["aspectRatio"] = aspect_ratio

        if person_generation is not None:
            parameters["personGeneration"] = person_generation

        parameters["sampleCount"] = number_of_videos

        operation = self._client.predict_long_running(
            model=self.model_name, instances=[instance], parameters=parameters
        )
        operation = GenerateVideoOperation.from_core_operation(operation)

        return operation

    def generate_videos(
        self,
        prompt: str,
        *,
        image: Union[pathlib.Path, image_types.ImageType] = None,
        aspect_ratio: Optional[VideoAspectRatio] = None,
        person_generation: Optional[PersonGeneration] = None,
        number_of_videos: int = 4,
    ) -> "VideoGenerationResponse":
        """Generates videos from a text prompt.

        Args:
            prompt: Text prompt for the video.
            aspect_ratio: Changes the aspect ratio of the generated video Supported
                values are:
                * "9:16" : 9:16 aspect ratio
                * "16:9" : 16:9 aspect ratio
            person_generation: Allow generation of people by the model.
                Supported values are:
                * "dont_allow" : Block generation of people
                * "allow_adult" : Generate adults, but not children
            number_of_videos: Number of videos to generate. Maximum 4.
        Returns:
            An `VideoGenerationResponse` object.
        """
        return self._generate_videos(
            prompt=prompt,
            image=image,
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            number_of_videos=number_of_videos,
        )


@dataclasses.dataclass
class VideoGenerationResponse:
    """Video generation response.

    Attributes:
        videos: The list of generated videos.
    """

    videos: List["image_types.Video"]
    rai_media_filtered_reasons: Optional[list[str]] = None

    def __iter__(self) -> typing.Iterator["image_types.Video"]:
        """Iterates through the generated videos."""
        yield from self.videos

    def __getitem__(self, idx: int) -> "image_types.Video":
        """Gets the generated video by index."""
        try:
            return self.videos[idx]
        except IndexError as e:
            if bool(self.rai_media_filtered_reasons):
                e.args = (
                    f"{e.args[0]}\n\n"
                    f"Note: Some videos were filtered out: {self.rai_media_filtered_reasons=}",
                )
            raise e


class GenerateVideoOperation(operations.BaseOperation):
    def set_result(self, result):
        response = result.generate_video_response
        samples = result.generate_video_response.generated_samples

        videos = []
        for sample in samples:
            video = image_types.GeneratedVideo(uri=sample.video.uri)
            videos.append(video)
        result = VideoGenerationResponse(
            videos, rai_media_filtered_reasons=list(response.rai_media_filtered_reasons)
        )
        super().set_result(result)

    def wait_bar(self, **kwargs) -> Iterator[protos.PredictLongRunningMetadata]:
        """A tqdm wait bar, yields `Operation` statuses until complete.

        Args:
            **kwargs: passed through to `tqdm.auto.tqdm(..., **kwargs)`

        Yields:
            Operation statuses as `protos.PredictLongRunningMetadata` objects.
        """
        return super().wait_bar(**kwargs)

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any
from urllib.request import urlopen

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pydantic.fields import FieldInfo

from palimpzest.constants import (
    NAIVE_EST_FILTER_SELECTIVITY,
    NAIVE_EST_NUM_INPUT_TOKENS,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import (
    IMAGE_FIELD_TYPES,
    IMAGE_LIST_FIELD_TYPES,
    ImageBase64,
    ImageFilepath,
    ImageURL,
    create_schema_from_fields,
)
from palimpzest.core.models import GenerationStats, OperatorCostEstimates
from palimpzest.prompts.prompt_factory import _detect_image_media_type
from palimpzest.query.operators.filter import LLMFilter

_FORMAT_TO_EXT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "GIF": ".gif",
    "WEBP": ".webp",
}
_FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
_KNOWN_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_IMAGE_FILEPATH_TYPES = {
    ImageFilepath,
    ImageFilepath | None,
    ImageFilepath | Any,
}
_IMAGE_BASE64_TYPES = {
    ImageBase64,
    ImageBase64 | None,
    ImageBase64 | Any,
}
_IMAGE_URL_TYPES = {
    ImageURL,
    ImageURL | None,
    ImageURL | Any,
}
_IMAGE_FILEPATH_LIST_TYPES = {
    list[ImageFilepath],
    list[ImageFilepath] | None,
    list[ImageFilepath] | Any,
}
_IMAGE_BASE64_LIST_TYPES = {
    list[ImageBase64],
    list[ImageBase64] | None,
    list[ImageBase64] | Any,
}
_IMAGE_URL_LIST_TYPES = {
    list[ImageURL],
    list[ImageURL] | None,
    list[ImageURL] | Any,
}


class RescaledImageFilter(LLMFilter):
    def __init__(
        self,
        rescale_factor: float,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.FILTER,
        reasoning_effort: str = "default",
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            prompt_strategy=prompt_strategy,
            reasoning_effort=reasoning_effort,
            *args,
            **kwargs,
        )
        self.rescale_factor = rescale_factor
        if self.rescale_factor <= 0:
            raise ValueError("rescale_factor must be > 0!")

    def __str__(self):
        op = super().__str__()
        op += f"    Rescale Factor: {self.rescale_factor}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"rescale_factor": self.rescale_factor, **id_params}
        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"rescale_factor": self.rescale_factor, **op_params}
        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates):
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        if self.is_image_op():
            base_image_tokens = 765 / 10  # 1024x1024 image is 765 tokens
            est_num_input_tokens = base_image_tokens / (self.rescale_factor ** 2)

        est_num_output_tokens = 1.25

        model_conversion_time_per_record = (
            self.model.get_seconds_per_output_token() * est_num_output_tokens
        )

        usd_per_input_token = (
            self.model.get_usd_per_audio_input_token()
            if self.is_audio_op()
            else self.model.get_usd_per_input_token()
        )
        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + self.model.get_usd_per_output_token() * est_num_output_tokens
        )

        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        quality = (self.model.get_overall_score() / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        input_fields = self.get_input_fields()
        new_fields = []
        new_candidate: DataRecord
        for field_name in input_fields:
            field_info = candidate.get_field_type(field_name)
            if field_info.annotation not in IMAGE_FIELD_TYPES:
                new_fields.append(field_name)
                continue

            field_value = candidate[field_name]
            if field_value is None:
                continue

            if field_info.annotation in IMAGE_LIST_FIELD_TYPES:
                new_field_schema = create_schema_from_fields([
                    {
                        "name": "image_b64_list",
                        "description": "List of Base64-encoded images",
                        "type": list[ImageBase64],
                    }
                ])

                image_b64_list = []
                for idx, item in enumerate(field_value):
                    if item is None:
                        image_b64_list.append(None)
                        continue
                    image_b64_list.append(self.rescale_image(field_info.annotation, item, f"{field_name}_{idx}"))

                new_candidate = DataRecord.from_parent(
                    schema=new_field_schema,
                    data_item={"image_b64_list": image_b64_list},
                    parent_record=candidate,
                )
                new_fields.append(f"image_b64_list")
            else:
                rescaled_b64 = self.rescale_image(field_info.annotation, field_value, field_name)
                new_field_schema = create_schema_from_fields([
                    {
                        "name": "image_b64",
                        "description": "Base64-encoded image",
                        "type": ImageBase64,
                    }
                ])           
                new_candidate = DataRecord.from_parent(
                    schema=new_field_schema,
                    data_item={"image_b64": rescaled_b64},
                    parent_record=candidate,
                )
                new_fields.append(f"image_b64")
            # TODO what to do with media_type
            # change the field image_path in the candidate 

            gen_kwargs = {
                "project_cols": new_fields,
                "filter_condition": self.filter_obj.filter_condition,
            }
            fields = {
                "passed_operator": FieldInfo(
                    annotation=bool,
                    description="Whether the record passed the filter operation",
                )
            }
            field_answers, _, generation_stats, _ = self.generator(new_candidate, fields, **gen_kwargs)

        return field_answers, generation_stats

    def rescale_image(self, annotation: object, field_value: str, name_stub: str) -> str:
        image: Image.Image

        if annotation in _IMAGE_FILEPATH_TYPES or annotation in _IMAGE_FILEPATH_LIST_TYPES:
            with open(field_value, "rb") as f:
                raw_bytes = f.read()
            image = Image.open(io.BytesIO(raw_bytes))

        elif annotation in _IMAGE_BASE64_TYPES or annotation in _IMAGE_BASE64_LIST_TYPES:
            # TODO check if field_value is already base64-encoded or if it's raw bytes that need to be encoded
            b64_image = base64.b64encode(field_value)
            image = Image.open(io.BytesIO(b64_image))

        elif annotation in _IMAGE_URL_TYPES or annotation in _IMAGE_URL_LIST_TYPES:
            url = field_value
            if url.startswith("data:"):
                header, data = url.split(",", 1)
                if ";base64" in header:
                    b64_image = base64.b64decode(data)
                    image = Image.open(io.BytesIO(b64_image))
                else: 
                    raise ValueError("Unsupported data URL encoding (only base64 is supported)")
            else:
                with urlopen(url) as response:
                    image_bytes = response.read()
                image = Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError(f"Unsupported image field type: {annotation}")

        width, height = image.size
        new_width = int(width / self.rescale_factor)
        new_height = int(height / self.rescale_factor)
        if new_width < 1 or new_height < 1:
            raise ValueError("rescale_factor produces an invalid image size")
        rescaled = image.resize((new_width, new_height))
        buffer = io.BytesIO()
        rescaled.save(buffer, format="PNG")
        rescaled_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return rescaled_b64


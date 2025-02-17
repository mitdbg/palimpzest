from __future__ import annotations

from typing import List, Literal

from together.types.abstract import BaseModel


class ImageRequest(BaseModel):
    # input or list of inputs
    prompt: str
    # model to query
    model: str
    # num generation steps
    steps: int | None = 20
    # seed
    seed: int | None = None
    # number of results to return
    n: int | None = 1
    # pixel height
    height: int | None = 1024
    # pixel width
    width: int | None = 1024
    # negative prompt
    negative_prompt: str | None = None


class ImageChoicesData(BaseModel):
    # response index
    index: int
    # base64 image response
    b64_json: str | None = None
    # URL hosting image
    url: str | None = None


class ImageResponse(BaseModel):
    # job id
    id: str | None = None
    # query model
    model: str | None = None
    # object type
    object: Literal["list"] | None = None
    # list of embedding choices
    data: List[ImageChoicesData] | None = None

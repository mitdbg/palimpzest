from __future__ import annotations

from typing import List, Literal

from together.types.abstract import BaseModel
from together.types.common import (
    ObjectType,
)


class EmbeddingRequest(BaseModel):
    # input or list of inputs
    input: str | List[str]
    # model to query
    model: str


class EmbeddingChoicesData(BaseModel):
    # response index
    index: int
    # object type
    object: ObjectType
    # embedding response
    embedding: List[float] | None = None


class EmbeddingResponse(BaseModel):
    # job id
    id: str | None = None
    # query model
    model: str | None = None
    # object type
    object: Literal["list"] | None = None
    # list of embedding choices
    data: List[EmbeddingChoicesData] | None = None

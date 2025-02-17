from __future__ import annotations

from enum import Enum
from typing import Literal

from together.types.abstract import BaseModel
from together.types.common import ObjectType


class ModelType(str, Enum):
    CHAT = "chat"
    LANGUAGE = "language"
    CODE = "code"
    IMAGE = "image"
    EMBEDDING = "embedding"
    MODERATION = "moderation"
    RERANK = "rerank"


class PricingObject(BaseModel):
    input: float | None = None
    output: float | None = None
    hourly: float | None = None
    base: float | None = None
    finetune: float | None = None


class ModelObject(BaseModel):
    # model id
    id: str
    # object type
    object: Literal[ObjectType.Model]
    created: int | None = None
    # model type
    type: ModelType | None = None
    # pretty name
    display_name: str | None = None
    # model creator organization
    organization: str | None = None
    # link to model resource
    link: str | None = None
    license: str | None = None
    context_length: int | None = None
    pricing: PricingObject

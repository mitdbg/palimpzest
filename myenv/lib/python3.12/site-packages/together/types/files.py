from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import Field

from together.types.abstract import BaseModel
from together.types.common import (
    ObjectType,
)


class FilePurpose(str, Enum):
    FineTune = "fine-tune"


class FileType(str, Enum):
    jsonl = "jsonl"
    parquet = "parquet"


class FileRequest(BaseModel):
    """
    Files request type
    """

    # training file ID
    training_file: str
    # base model string
    model: str
    # number of epochs to train for
    n_epochs: int
    # training learning rate
    learning_rate: float
    # number of checkpoints to save
    n_checkpoints: int | None = None
    # training batch size
    batch_size: int | None = None
    # up to 40 character suffix for output model name
    suffix: str | None = None
    # weights & biases api key
    wandb_api_key: str | None = None


class FileResponse(BaseModel):
    """
    Files API response type
    """

    id: str
    object: Literal[ObjectType.File]
    # created timestamp
    created_at: int | None = None
    type: FileType | None = None
    purpose: FilePurpose | None = None
    filename: str | None = None
    # file byte size
    bytes: int | None = None
    # JSONL line count
    line_count: int | None = Field(None, alias="LineCount")
    processed: bool | None = Field(None, alias="Processed")


class FileList(BaseModel):
    # object type
    object: Literal["list"] | None = None
    # list of fine-tune job objects
    data: List[FileResponse] | None = None


class FileDeleteResponse(BaseModel):
    # file id
    id: str
    # object type
    object: Literal[ObjectType.File]
    # is deleted
    deleted: bool


class FileObject(BaseModel):
    # object type
    object: Literal["local"] | None = None
    # fine-tune job id
    id: str | None = None
    # local path filename
    filename: str | None = None
    # size in bytes
    size: int | None = None

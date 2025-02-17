from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import ConfigDict
from tqdm.utils import CallbackIOWrapper

from together.types.abstract import BaseModel


# Generation finish reason
class FinishReason(str, Enum):
    Length = "length"
    StopSequence = "stop"
    EOS = "eos"
    ToolCalls = "tool_calls"
    Error = "error"


class UsageData(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ObjectType(str, Enum):
    Completion = "text.completion"
    CompletionChunk = "completion.chunk"
    ChatCompletion = "chat.completion"
    ChatCompletionChunk = "chat.completion.chunk"
    Embedding = "embedding"
    FinetuneEvent = "fine-tune-event"
    File = "file"
    Model = "model"


class LogprobsPart(BaseModel):
    # token list
    tokens: List[str | None] | None = None
    # token logprob list
    token_logprobs: List[float | None] | None = None


class PromptPart(BaseModel):
    # prompt string
    text: str | None = None
    # list of prompt logprobs
    logprobs: LogprobsPart | None = None


class DeltaContent(BaseModel):
    content: str | None = None


class TogetherRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str
    url: str
    headers: Dict[str, str] | None = None
    params: Dict[str, Any] | CallbackIOWrapper | None = None
    files: Dict[str, Any] | None = None
    allow_redirects: bool = True
    override_headers: bool = False

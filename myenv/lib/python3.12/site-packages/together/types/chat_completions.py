from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Dict, List

from pydantic import model_validator
from typing_extensions import Self

from together.types.abstract import BaseModel
from together.types.common import (
    DeltaContent,
    FinishReason,
    LogprobsPart,
    ObjectType,
    PromptPart,
    UsageData,
)


class MessageRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"


class ResponseFormatType(str, Enum):
    JSON_OBJECT = "json_object"


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCalls(BaseModel):
    id: str | None = None
    type: str | None = None
    function: FunctionCall | None = None


class ChatCompletionMessageContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str


class ChatCompletionMessageContent(BaseModel):
    type: ChatCompletionMessageContentType
    text: str | None = None
    image_url: ChatCompletionMessageContentImageURL | None = None


class ChatCompletionMessage(BaseModel):
    role: MessageRole
    content: str | List[ChatCompletionMessageContent] | None = None
    tool_calls: List[ToolCalls] | None = None


class ResponseFormat(BaseModel):
    type: ResponseFormatType
    schema_: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": self.schema_, "type": self.type}


class FunctionTool(BaseModel):
    description: str | None = None
    name: str
    parameters: Dict[str, Any] | None = None


class FunctionToolChoice(BaseModel):
    name: str


class Tools(BaseModel):
    type: str
    function: FunctionTool


class ToolChoice(BaseModel):
    type: str
    function: FunctionToolChoice


class ToolChoiceEnum(str, Enum):
    Auto = "auto"


class ChatCompletionRequest(BaseModel):
    # list of messages
    messages: List[ChatCompletionMessage]
    # model name
    model: str
    # stopping criteria: max tokens to generate
    max_tokens: int | None = None
    # stopping criteria: list of strings to stop generation
    stop: List[str] | None = None
    # sampling hyperparameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    min_p: float | None = None
    logit_bias: Dict[str, float] | None = None
    seed: int | None = None
    # stream SSE token chunks
    stream: bool = False
    # return logprobs
    logprobs: int | None = None
    # echo prompt.
    # can be used with logprobs to return prompt logprobs
    echo: bool | None = None
    # number of output generations
    n: int | None = None
    # moderation model
    safety_model: str | None = None
    # constraints
    response_format: ResponseFormat | None = None
    tools: List[Tools] | None = None
    tool_choice: ToolChoice | ToolChoiceEnum | None = None

    # Raise warning if repetition_penalty is used with presence_penalty or frequency_penalty
    @model_validator(mode="after")
    def verify_parameters(self) -> Self:
        if self.repetition_penalty:
            if self.presence_penalty or self.frequency_penalty:
                warnings.warn(
                    "repetition_penalty is not advisable to be used alongside presence_penalty or frequency_penalty"
                )
        return self


class ChatCompletionChoicesData(BaseModel):
    index: int | None = None
    logprobs: LogprobsPart | None = None
    seed: int | None = None
    finish_reason: FinishReason | None = None
    message: ChatCompletionMessage | None = None


class ChatCompletionResponse(BaseModel):
    # request id
    id: str | None = None
    # object type
    object: ObjectType | None = None
    # created timestamp
    created: int | None = None
    # model name
    model: str | None = None
    # choices list
    choices: List[ChatCompletionChoicesData] | None = None
    # prompt list
    prompt: List[PromptPart] | List[None] | None = None
    # token usage data
    usage: UsageData | None = None


class ChatCompletionChoicesChunk(BaseModel):
    index: int | None = None
    logprobs: float | None = None
    seed: int | None = None
    finish_reason: FinishReason | None = None
    delta: DeltaContent | None = None


class ChatCompletionChunk(BaseModel):
    # request id
    id: str | None = None
    # object type
    object: ObjectType | None = None
    # created timestamp
    created: int | None = None
    # model name
    model: str | None = None
    # delta content
    choices: List[ChatCompletionChoicesChunk] | None = None
    # finish reason
    finish_reason: FinishReason | None = None
    # token usage data
    usage: UsageData | None = None

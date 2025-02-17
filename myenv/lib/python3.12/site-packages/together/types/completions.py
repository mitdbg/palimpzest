from __future__ import annotations

import warnings
from typing import Dict, List

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


class CompletionRequest(BaseModel):
    # prompt to complete
    prompt: str
    # query model
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

    # Raise warning if repetition_penalty is used with presence_penalty or frequency_penalty
    @model_validator(mode="after")
    def verify_parameters(self) -> Self:
        if self.repetition_penalty:
            if self.presence_penalty or self.frequency_penalty:
                warnings.warn(
                    "repetition_penalty is not advisable to be used alongside presence_penalty or frequency_penalty"
                )
        return self


class CompletionChoicesData(BaseModel):
    index: int
    logprobs: LogprobsPart | None = None
    seed: int | None = None
    finish_reason: FinishReason
    text: str


class CompletionChoicesChunk(BaseModel):
    index: int | None = None
    logprobs: float | None = None
    seed: int | None = None
    finish_reason: FinishReason | None = None
    delta: DeltaContent | None = None


class CompletionResponse(BaseModel):
    # request id
    id: str | None = None
    # object type
    object: ObjectType | None = None
    # created timestamp
    created: int | None = None
    # model name
    model: str | None = None
    # choices list
    choices: List[CompletionChoicesData] | None = None
    # prompt list
    prompt: List[PromptPart] | None = None
    # token usage data
    usage: UsageData | None = None


class CompletionChunk(BaseModel):
    # request id
    id: str | None = None
    # object type
    object: ObjectType | None = None
    # created timestamp
    created: int | None = None
    # model name
    model: str | None = None
    # choices list
    choices: List[CompletionChoicesChunk] | None = None
    # token usage data
    usage: UsageData | None = None

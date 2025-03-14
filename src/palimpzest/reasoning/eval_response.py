from dataclasses import dataclass


@dataclass
class EvaluationResponse:
    pass_: bool = False
    think: str = ""
    type: str | None = None
    improvement_plan: str | None = None
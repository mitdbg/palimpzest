from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, Policy


class OptimizerConfig(BaseModel):
    """Configuration owned by the optimizer layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    optimizer_strategy: Any = Field(default="pareto")
    policy: Policy = Field(default_factory=MaxQuality)
    available_models: list[Model | str] | None = Field(default=None)
    remove_models: list[Model | str] | None = Field(default=None)

    verbose: bool = Field(default=False)
    join_parallelism: int = Field(default=64)
    reasoning_effort: str = Field(default="default")
    execution_strategy: Any = Field(default="parallel")

    allow_bonded_query: bool = Field(default=True)
    allow_model_selection: bool = Field(default=True)
    allow_rag_reduction: bool = Field(default=True)
    allow_mixtures: bool = Field(default=True)
    allow_critic: bool = Field(default=True)
    allow_split_merge: bool = Field(default=False)
    use_final_op_quality: bool = Field(default=False)

    sentinel_execution_strategy: Any | None = Field(default="auto")
    k: int = Field(default=6)
    j: int = Field(default=4)
    sample_budget: int = Field(default=100)
    sample_cost_budget: float | None = Field(default=None)
    seed: int = Field(default=42)
    exp_name: str | None = Field(default=None)
    priors: dict | None = Field(default=None)
    dont_use_priors: bool = Field(default=False)

    def with_strategy(self, optimizer_strategy: Any) -> OptimizerConfig:
        return self.model_copy(
            update={"optimizer_strategy": optimizer_strategy},
            deep=True,
        )

    def to_optimizer_kwargs(self, optimizer_strategy: Any | None = None) -> dict:
        config = self if optimizer_strategy is None else self.with_strategy(optimizer_strategy)
        return {
            "optimizer_config": config,
        }

    def to_sentinel_execution_kwargs(self) -> dict:
        return {
            "policy": self.policy,
            "k": self.k,
            "j": self.j,
            "sample_budget": self.sample_budget,
            "sample_cost_budget": self.sample_cost_budget,
            "priors": self.priors,
            "use_final_op_quality": self.use_final_op_quality,
            "seed": self.seed,
            "exp_name": self.exp_name,
            "dont_use_priors": self.dont_use_priors,
        }

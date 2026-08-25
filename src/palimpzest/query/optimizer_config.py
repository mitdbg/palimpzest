from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, Policy


class OptimizerConfig(BaseModel):
    """
    Configuration owned by the optimizer layer.

    OptimizerConfig is the single internal home for policy, normalized model
    selection, optimizer strategy, logical/physical operator toggles, and
    sentinel/MAB sampling knobs. Users may provide model identifiers as strings,
    but OptimizerConfig stores them as Model objects after validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    optimizer_strategy: Any = Field(default="pareto")
    policy: Policy = Field(default_factory=MaxQuality)
    available_models: list[Model] = Field(default_factory=list)
    remove_models: list[Model] = Field(default_factory=list)

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

    budget_allocator: Any | None = Field(default=None)
    final_uncertainty_weight: float = Field(default=0.0)

    seed: int = Field(default=42)
    exp_name: str | None = Field(default=None)
    priors: dict | None = Field(default=None)
    dont_use_priors: bool = Field(default=False)

    @field_validator("available_models", "remove_models", mode="before")
    @classmethod
    def _coerce_models(cls, models):
        if models is None:
            return []
        return [
            Model(model) if isinstance(model, str) else model
            for model in models
        ]

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

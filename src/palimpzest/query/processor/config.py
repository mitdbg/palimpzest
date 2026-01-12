from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, Policy


# TODO: Add description for each field.
class QueryProcessorConfig(BaseModel):
    """Shared context for query processors"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # execution and optimization flags
    execution_strategy: str = Field(default="parallel")              # substituted with ExecutionStrategyType
    sentinel_execution_strategy: str | None = Field(default="auto")  # substituted with SentinelExecutionStrategyType
    optimizer_strategy: str = Field(default="pareto")                # substituted with OptimizationStrategyType

    # general execution flags
    policy: Policy = Field(default_factory=MaxQuality)
    enforce_types: bool = Field(default=False)
    scan_start_idx: int = Field(default=0)
    num_samples: int | None = Field(default=None)
    verbose: bool = Field(default=False)
    progress: bool = Field(default=True)
    available_models: list[Model] | None = Field(default=None)
    remove_models: list[Model] | None = Field(default=None)
    max_workers: int | None = Field(default=64)
    join_parallelism: int = Field(default=64)
    batch_size: int | None = Field(default=None)
    reasoning_effort: str | None = Field(default="default")  # Gemini: "disable", "low", "medium", "high"
    use_vertex: bool = Field(default=False)  # Whether to use Vertex models for Gemini or Google models
    gemini_credentials_path: str | None = Field(default=None)  # Path to Gemini credentials file
    api_base: str | None = Field(default=None)  # API base URL for vLLM

    # operator flags
    allow_bonded_query: bool = Field(default=True)
    allow_model_selection: bool = Field(default=True)
    allow_rag_reduction: bool = Field(default=True)
    allow_mixtures: bool = Field(default=True)
    allow_critic: bool = Field(default=True)
    allow_split_merge: bool = Field(default=False)
    use_final_op_quality: bool = Field(default=False)

    # sentinel optimization flags
    k: int = Field(default=6)
    j: int = Field(default=4)
    sample_budget: int = Field(default=100)
    sample_cost_budget: float | None = Field(default=None)
    seed: int = Field(default=42)
    exp_name: str | None = Field(default=None)
    priors: dict | None = Field(default=None)
    dont_use_priors: bool = Field(default=False)

    def to_dict(self) -> dict:
        """Convert the config to a dict representation."""
        return self.model_dump()

    def copy(self) -> QueryProcessorConfig:
        """Create a copy of the config."""
        return QueryProcessorConfig(**self.to_dict())

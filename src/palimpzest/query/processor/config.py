from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from palimpzest.query.optimizer_config import OptimizerConfig


# TODO: Add description for each field.
class QueryProcessorConfig(BaseModel):
    """Shared context for query processors"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # execution and optimization flags
    execution_strategy: str = Field(default="parallel")              # substituted with ExecutionStrategyType
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)

    # general execution flags
    enforce_types: bool = Field(default=False)
    scan_start_idx: int = Field(default=0)
    num_samples: int | None = Field(default=None)
    verbose: bool = Field(default=False)
    progress: bool = Field(default=True)
    max_workers: int | None = Field(default=64)
    batch_size: int | None = Field(default=None)
    use_vertex: bool = Field(default=False)  # Whether to use Vertex models for Gemini or Google models
    use_azure: bool = Field(default=False)  # Whether to use Azure for OpenAI models
    gemini_credentials_path: str | None = Field(default=None)  # Path to Gemini credentials file
    azure_endpoint: str | None = Field(default=None)  # Azure endpoint URL (AZURE_API_BASE)
    azure_api_version: str | None = Field(default=None)  # Azure API version

    # TODO make it more robust than string type for optimizer selection
    # if only run() is used, then optimizer will be changed to "naive"
    optimizer: str = Field(default="abacus")  # "abacus" or "cluster"

    def to_dict(self) -> dict:
        """Convert the config to a dict representation."""
        return self.model_dump()

    def copy(self) -> QueryProcessorConfig:
        """Create a copy of the config."""
        return QueryProcessorConfig(**self.to_dict())

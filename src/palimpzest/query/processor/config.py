from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from palimpzest.query.optimizer_config import OptimizerConfig


OPTIMIZER_CONFIG_FACADE_FIELDS = set(OptimizerConfig.model_fields) - {
    "execution_strategy",
    "verbose",
}


class QueryProcessorConfig(BaseModel):
    """
    Public facade for configuring query processing.

    QueryProcessorConfig owns execution, provider, and validation settings.
    Optimizer-owned fields may be supplied directly to the main QueryProcessorConfig  ergonomics, but they are routed into a specific OptimizerConfig object before validation. After construction, those values live only on optimizer_config.
    """
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
    validator: Any | None = Field(default=None)
    use_vertex: bool = Field(default=False)  # Whether to use Vertex models for Gemini or Google models
    use_azure: bool = Field(default=False)  # Whether to use Azure for OpenAI models
    gemini_credentials_path: str | None = Field(default=None)  # Path to Gemini credentials file
    azure_endpoint: str | None = Field(default=None)  # Azure endpoint URL (AZURE_API_BASE)
    azure_api_version: str | None = Field(default=None)  # Azure API version

    # TODO make it more robust than string type for optimizer selection
    # if only run() is used, then optimizer will be changed to "naive"
    optimizer: str = Field(default="abacus")  # "abacus" or "cluster"

    @model_validator(mode="before")
    @classmethod
    def _route_optimizer_config_fields(cls, data):
        if not isinstance(data, dict):
            return data

        data = dict(data)
        optimizer_config = data.get("optimizer_config")
        if isinstance(optimizer_config, OptimizerConfig):
            optimizer_config_fields = optimizer_config.model_fields_set
            optimizer_config = optimizer_config.model_dump()
            optimizer_config = {
                field: value
                for field, value in optimizer_config.items()
                if field in optimizer_config_fields
            }
        elif optimizer_config is None:
            optimizer_config = {}
        elif isinstance(optimizer_config, dict):
            optimizer_config = dict(optimizer_config)

        if isinstance(optimizer_config, dict):
            for field in OPTIMIZER_CONFIG_FACADE_FIELDS:
                if field not in data:
                    continue
                if field in optimizer_config:
                    raise ValueError(
                        f"Specify `{field}` either as a QueryProcessorConfig argument "
                        "or inside optimizer_config, not both."
                    )
                optimizer_config[field] = data.pop(field)
            data["optimizer_config"] = optimizer_config

        return data

    def to_dict(self) -> dict:
        """Convert the config to a dict representation."""
        return self.model_dump()

    def copy(self) -> QueryProcessorConfig:
        """Create a copy of the config."""
        return QueryProcessorConfig(**self.to_dict())

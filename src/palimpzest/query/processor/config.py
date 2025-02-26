import json
from dataclasses import dataclass, field

from palimpzest.constants import Model
from palimpzest.core.data.datareaders import DataReader
from palimpzest.policy import MaxQuality, Policy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType, SentinelExecutionStrategyType
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.processor.processing_strategy_type import ProcessingStrategyType


# TODO: Separate out the config for the Optimizer, ExecutionStrategy, and QueryProcessor
# TODO: Add description for each field.
@dataclass
class QueryProcessorConfig:
    """Shared context for query processors"""
    processing_strategy: str | ProcessingStrategyType = field(default="auto")
    execution_strategy: str | ExecutionStrategyType = field(default="sequential")
    sentinel_execution_strategy: str | SentinelExecutionStrategyType | None = field(default="auto")
    optimizer_strategy: str | OptimizationStrategyType = field(default="pareto")

    val_datasource: DataReader | None = field(default=None)

    policy: Policy = field(default_factory=MaxQuality)
    scan_start_idx: int = field(default=0)
    num_samples: int = field(default=None)
    cache: bool = field(default=False)  # NOTE: until we properly implement caching, let's set the default to False
    include_baselines: bool = field(default=False)
    min_plans: int | None = field(default=None)
    verbose: bool = field(default=False)
    progress: bool = field(default=True)
    available_models: list[Model] | None = field(default=None)

    max_workers: int | None = field(default=None)
    num_workers_per_plan: int = field(default=1)

    allow_bonded_query: bool = field(default=True)
    allow_conventional_query: bool = field(default=False)
    allow_model_selection: bool = field(default=True)
    allow_code_synth: bool = field(default=False)
    allow_token_reduction: bool = field(default=False)
    allow_rag_reduction: bool = field(default=False)
    allow_mixtures: bool = field(default=True)
    allow_critic: bool = field(default=False)
    use_final_op_quality: bool = field(default=False)

    def to_dict(self) -> dict:
        """Convert the config to a dict representation."""
        return {
            "processing_strategy": self.processing_strategy,
            "execution_strategy": self.execution_strategy,
            "optimizer_strategy": self.optimizer_strategy,
            "val_datasource": self.val_datasource,
            "policy": self.policy,
            "scan_start_idx": self.scan_start_idx,
            "num_samples": self.num_samples,
            "cache": self.cache,
            "include_baselines": self.include_baselines,
            "min_plans": self.min_plans,
            "verbose": self.verbose,
            "progress": self.progress,
            "available_models": self.available_models,
            "max_workers": self.max_workers,
            "num_workers_per_plan": self.num_workers_per_plan,
            "allow_bonded_query": self.allow_bonded_query,
            "allow_conventional_query": self.allow_conventional_query,
            "allow_model_selection": self.allow_model_selection,
            "allow_code_synth": self.allow_code_synth,
            "allow_token_reduction": self.allow_token_reduction,
            "allow_rag_reduction": self.allow_rag_reduction,
            "allow_mixtures": self.allow_mixtures,
            "allow_critic": self.allow_critic,
            "use_final_op_quality": self.use_final_op_quality
        }

    def to_json_str(self):
        """Convert the config to a JSON string representation."""
        config_dict = self.to_dict()
        config_dict["val_datasource"] = (
            None if self.val_datasource is None else self.val_datasource.serialize()
        )
        config_dict["policy"] = self.policy.to_json_str()
        for strategy in ["processing_strategy", "execution_strategy", "optimizer_strategy"]:
            config_dict[strategy] = str(config_dict[strategy])

        return json.dumps(config_dict, indent=2)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

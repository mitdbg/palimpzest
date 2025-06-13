import json
from dataclasses import dataclass, field

from palimpzest.constants import Model
from palimpzest.core.data.datareaders import DataReader
from palimpzest.policy import MaxQuality, Policy


# TODO: Separate out the config for the Optimizer, ExecutionStrategy, and QueryProcessor
# TODO: Add description for each field.
@dataclass
class QueryProcessorConfig:
    """Shared context for query processors"""
    processing_strategy: str = field(default="auto")                 # substituted with ProcessingStrategyType
    execution_strategy: str = field(default="sequential")            # substituted with ExecutionStrategyType
    sentinel_execution_strategy: str | None = field(default="auto")  # substituted with SentinelExecutionStrategyType
    optimizer_strategy: str = field(default="pareto")                # substituted with OptimizationStrategyType

    val_datasource: DataReader | None = field(default=None)

    policy: Policy = field(default_factory=MaxQuality)
    scan_start_idx: int = field(default=0)
    num_samples: int = field(default=None)
    cache: bool = field(default=False)  # NOTE: until we properly implement caching, let's set the default to False
    verbose: bool = field(default=False)
    progress: bool = field(default=True)
    available_models: list[Model] | None = field(default=None)

    max_workers: int | None = field(default=None)

    allow_bonded_query: bool = field(default=True)
    allow_model_selection: bool = field(default=True)
    allow_code_synth: bool = field(default=False)
    allow_rag_reduction: bool = field(default=True)
    allow_mixtures: bool = field(default=True)
    allow_critic: bool = field(default=True)
    allow_split_merge: bool = field(default=False)
    use_final_op_quality: bool = field(default=False)

    kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the config to a dict representation."""
        return {
            "processing_strategy": self.processing_strategy,
            "execution_strategy": self.execution_strategy,
            "sentinel_execution_strategy": self.sentinel_execution_strategy,
            "optimizer_strategy": self.optimizer_strategy,
            "val_datasource": self.val_datasource,
            "policy": self.policy,
            "scan_start_idx": self.scan_start_idx,
            "num_samples": self.num_samples,
            "cache": self.cache,
            "verbose": self.verbose,
            "progress": self.progress,
            "available_models": self.available_models,
            "max_workers": self.max_workers,
            "allow_bonded_query": self.allow_bonded_query,
            "allow_model_selection": self.allow_model_selection,
            "allow_code_synth": self.allow_code_synth,
            "allow_rag_reduction": self.allow_rag_reduction,
            "allow_mixtures": self.allow_mixtures,
            "allow_critic": self.allow_critic,
            "allow_split_merge": self.allow_split_merge,
            "use_final_op_quality": self.use_final_op_quality,
            **self.kwargs,
        }

    def to_json_str(self):
        """Convert the config to a JSON string representation."""
        config_dict = self.to_dict()
        config_dict["val_datasource"] = (
            None if self.val_datasource is None else self.val_datasource.serialize()
        )
        config_dict["policy"] = self.policy.to_json_str()
        for strategy in ["processing_strategy", "execution_strategy", "sentinel_execution_strategy", "optimizer_strategy"]:
            config_dict[strategy] = str(config_dict[strategy])

        return json.dumps(config_dict, indent=2)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.kwargs.update(kwargs)

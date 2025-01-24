import json
from dataclasses import dataclass, field

from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, Policy


# TODO: Separate out the config for the Optimizer, ExecutionStrategy, and QueryProcessor
# TODO: Add description for each field.
@dataclass
class QueryProcessorConfig:
    """Shared context for query processors"""
    processing_strategy: str = field(default="no_sentinel")
    execution_strategy: str = field(default="sequential")
    optimizer_strategy: str = field(default="pareto")

    policy: Policy = field(default_factory=MaxQuality)
    scan_start_idx: int = field(default=0)
    num_samples: int = field(default=float("inf"))
    nocache: bool = field(default=True)  # NOTE: until we properly implement caching, let's set the default to True
    include_baselines: bool = field(default=False)
    min_plans: int | None = field(default=None)
    verbose: bool = field(default=False)
    available_models: list[Model] | None = field(default=None)
    
    max_workers: int | None = field(default=None)
    num_workers_per_plan: int = field(default=1)

    allow_bonded_query: bool = field(default=True)
    allow_conventional_query: bool = field(default=False)
    allow_model_selection: bool = field(default=True)
    allow_code_synth: bool = field(default=False)
    allow_token_reduction: bool = field(default=False)
    allow_rag_reduction: bool = field(default=True)
    allow_mixtures: bool = field(default=True)
    use_final_op_quality: bool = field(default=False)

    def to_json_str(self):
        return json.dumps({
            "processing_strategy": self.processing_strategy,
            "execution_strategy": self.execution_strategy,
            "optimizer_strategy": self.optimizer_strategy,
            "policy": self.policy.to_json_str(),
            "scan_start_idx": self.scan_start_idx,
            "num_samples": self.num_samples,
            "nocache": self.nocache,
            "include_baselines": self.include_baselines,
            "min_plans": self.min_plans,
            "verbose": self.verbose,
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
            "use_final_op_quality": self.use_final_op_quality,
        }, indent=2)

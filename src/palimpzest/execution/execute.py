from typing import List, Optional

from palimpzest.constants import Model, OptimizationStrategy
from palimpzest.execution import ExecutionEngine, PipelinedSingleThreadSentinelExecution
from palimpzest.policy import Policy
from palimpzest.sets import Set


class Execute:
    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int=20,
        nocache: bool=False,
        include_baselines: bool=False,
        min_plans: Optional[int] = None,
        max_workers: int=1,
        verbose: bool = False,
        available_models: Optional[List[Model]] = [],
        allow_bonded_query: Optional[bool]=True,
        allow_conventional_query: Optional[bool]=False,
        allow_model_selection: Optional[bool]=True,
        allow_code_synth: Optional[bool]=True,
        allow_token_reduction: Optional[bool]=True,
        optimization_strategy: OptimizationStrategy=OptimizationStrategy.OPTIMAL,
        execution_engine: ExecutionEngine = PipelinedSingleThreadSentinelExecution,
        *args,
        **kwargs
    ):

        return execution_engine(
            num_samples=num_samples,
            nocache=nocache,
            include_baselines=include_baselines,
            min_plans=min_plans,
            max_workers=max_workers,
            verbose=verbose,
            available_models=available_models,
            allow_bonded_query=allow_bonded_query,
            allow_conventional_query=allow_conventional_query,
            allow_code_synth=allow_code_synth,
            allow_model_selection=allow_model_selection,
            allow_token_reduction=allow_token_reduction,
            optimization_strategy=optimization_strategy,
            *args,
            **kwargs
        ).execute(
            dataset=dataset,
            policy=policy
        )

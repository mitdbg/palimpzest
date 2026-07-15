from __future__ import annotations

import logging


from palimpzest.core.data.dataset import Dataset
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.optimizer_config import OptimizerConfig
from palimpzest.query.optimizer.cost_model import BaseCostModel
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.plan import PhysicalPlan

logger = logging.getLogger(__name__)


class Optimizer:
    """
    This class represents a general interface for an Optimizer.
    The optimizer is responsible for searching the space of possible physical plans
    for a user's initial (logical) plan and selecting the one which is closest to
    optimizing the user's policy objective.
    """

    def __init__(
        self,
        cost_model: BaseCostModel,
        optimizer_config: OptimizerConfig,
    ):
        optimizer_strategy = optimizer_config.optimizer_strategy
        execution_strategy = optimizer_config.execution_strategy
        if not isinstance(optimizer_strategy, OptimizationStrategyType):
            try:
                optimizer_strategy = OptimizationStrategyType[str(optimizer_strategy).upper().replace("-", "_")]
            except KeyError as e:
                raise ValueError(f"Unsupported optimizer_strategy: {optimizer_config.optimizer_strategy}") from e
        if not isinstance(execution_strategy, ExecutionStrategyType):
            try:
                execution_strategy = ExecutionStrategyType[str(execution_strategy).upper().replace("-", "_")]
            except KeyError as e:
                raise ValueError(f"Unsupported execution_strategy: {optimizer_config.execution_strategy}") from e

        optimizer_config = optimizer_config.model_copy(
            update={
                "optimizer_strategy": optimizer_strategy,
                "execution_strategy": execution_strategy,
            },
        )

        self.optimizer_config = optimizer_config
        self.policy = optimizer_config.policy
        self.cost_model = cost_model
        available_models = list(optimizer_config.available_models or [])

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

        self.allow_model_selection = optimizer_config.allow_model_selection
        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
        if optimizer_strategy == OptimizationStrategyType.NONE:
            self.allow_bonded_query = True
            self.allow_rag_reduction = False
            self.allow_mixtures = False
            self.allow_critic = False
            self.allow_split_merge = False
            self.available_models = [available_models[0]]
        else:
            self.allow_bonded_query = optimizer_config.allow_bonded_query
            self.allow_rag_reduction = optimizer_config.allow_rag_reduction
            self.allow_mixtures = optimizer_config.allow_mixtures
            self.allow_critic = optimizer_config.allow_critic
            self.allow_split_merge = optimizer_config.allow_split_merge
            self.available_models = available_models

        # store optimization hyperparameters
        self.verbose = optimizer_config.verbose
        self.join_parallelism = optimizer_config.join_parallelism
        self.reasoning_effort = optimizer_config.reasoning_effort
        self.optimizer_strategy = optimizer_strategy
        self.execution_strategy = execution_strategy
        self.use_final_op_quality = optimizer_config.use_final_op_quality

    def update_cost_model(self, cost_model: BaseCostModel):
        self.cost_model = cost_model

    def update_strategy(self, optimizer_strategy: OptimizationStrategyType):
        # TODO check if this should be here or in Abacus only? 
        # set the optimizer_strategy
        self.optimizer_strategy = optimizer_strategy
        self.optimizer_config = self.optimizer_config.with_strategy(optimizer_strategy)

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

    def get_physical_op_params(self):
        return {
            "verbose": self.verbose,
            "available_models": self.available_models,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            "is_validation": self.optimizer_strategy
            == OptimizationStrategyType.SENTINEL,
        }


    def optimize(self, dataset: Dataset, *args, **kwargs) -> list[PhysicalPlan]:
        raise NotImplementedError("The optimize method must be implemented by subclasses of Optimizer.")

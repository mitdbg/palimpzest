from __future__ import annotations

import logging


from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.optimizer.cost_model import BaseCostModel, SampleBasedCostModel
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
        policy: Policy,
        cost_model: BaseCostModel,
        available_models: list[Model],
        join_parallelism: int = 64,
        reasoning_effort: str = "default",
        verbose: bool = False,
        allow_bonded_query: bool = True,
        allow_rag_reduction: bool = False,
        allow_mixtures: bool = True,
        allow_critic: bool = False,
        allow_split_merge: bool = False,
        optimizer_strategy: OptimizationStrategyType = OptimizationStrategyType.PARETO,
        execution_strategy: ExecutionStrategyType = ExecutionStrategyType.PARALLEL,
        use_final_op_quality: bool = False,
        **kwargs,
    ):

        self.policy = policy
        self.cost_model = cost_model

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

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
            self.allow_bonded_query = allow_bonded_query
            self.allow_rag_reduction = allow_rag_reduction
            self.allow_mixtures = allow_mixtures
            self.allow_critic = allow_critic
            self.allow_split_merge = allow_split_merge
            self.available_models = available_models

        # store optimization hyperparameters
        self.verbose = verbose
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.optimizer_strategy = optimizer_strategy
        self.execution_strategy = execution_strategy
        self.use_final_op_quality = use_final_op_quality

    def update_cost_model(self, cost_model: BaseCostModel):
        self.cost_model = cost_model

    def update_strategy(self, optimizer_strategy: OptimizationStrategyType):
        # TODO check if this should be here or in Abacus only? 
        # set the optimizer_strategy
        self.optimizer_strategy = optimizer_strategy

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

    def deepcopy_clean(self):
        # TODO should this be part of the generic Optimizer or the AbacusOptimizer?
        optimizer = self.__class__(
            policy=self.policy,
            cost_model=SampleBasedCostModel(),
            verbose=self.verbose,
            available_models=self.available_models,
            join_parallelism=self.join_parallelism,
            reasoning_effort=self.reasoning_effort,
            allow_bonded_query=self.allow_bonded_query,
            allow_rag_reduction=self.allow_rag_reduction,
            allow_mixtures=self.allow_mixtures,
            allow_critic=self.allow_critic,
            allow_split_merge=self.allow_split_merge,
            optimizer_strategy=self.optimizer_strategy,
            execution_strategy=self.execution_strategy,
            use_final_op_quality=self.use_final_op_quality,
        )
        return optimizer


    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        raise NotImplementedError("The optimize method must be implemented by subclasses of Optimizer.")
from __future__ import annotations

import logging
from copy import deepcopy

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import get_schema_field_names
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.optimizer.optimizer import Optimizer
logger = logging.getLogger(__name__)


class ClusterOptimizer(Optimizer):
    """
    Compared to the Abacus optimizer, which is based off the cascades framework, this optimizer separates the boundary between logical and physical optimization into two distinct phases. 
    Pro:
        - Logical optimization can reason over individual operators and decide whether to merge/split them and rewrite semantic parameters using prompt optimization.
        - Physical optimization can reason over the entire plan and decide which models to use for each operator and how to parallelize the execution of the plan.
    Con:
        - The logical plan search may not be able to take into account the physical plan search, which may lead to suboptimal plans being selected.

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
        # store the policy
        self.policy = policy
        # store the cost model
        self.cost_model = cost_model
        self.available_models = available_models
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose

        # TODO look at optimizer strategy ? 
        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
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
        self.optimizer_strategy = optimizer_strategy
        self.execution_strategy = execution_strategy
        self.use_final_op_quality = use_final_op_quality

        self.logical_optimizer = LogicalOptimizer()
        self.physical_optimizer = PhysicalOptimizer()

        # prune implementation rules based on boolean flags
        # TODO disable phsical implementation rules based on boolean flags

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        logger.info(f"Optimizing query plan: {dataset}")
        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.logical_optimizer.optimize(dataset_copy)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(self.groups, final_group_id, self.policy, self.use_final_op_quality)

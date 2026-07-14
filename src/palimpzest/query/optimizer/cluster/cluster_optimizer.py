from __future__ import annotations

import logging
from copy import deepcopy

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import get_schema_field_names
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalOptimizer, LogicalPlan
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.plan import PhysicalPlan
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logical_optimizer = LogicalOptimizer()
        # self.physical_optimizer = PhysicalOptimizer()

        # prune implementation rules based on boolean flags
        # TODO disable phsical implementation rules based on boolean flags

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        
        Output of clusteroptimizer is a list whose first element will be the optimal plan and the rest will be suboptimal plans. The user can choose to execute any of these plans based on their preference for quality vs. latency.
        """
        logger.info(f"Optimizing query plan: {dataset}")
        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        initial_plan = LogicalPlan.from_dataset(dataset_copy)
        logger.info(f"Initial logical plan: {initial_plan}")
        raise NotImplementedError("Logical plan optimization is not yet implemented.")
        self.logical_optimizer.optimize(dataset_copy)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(self.groups, final_group_id, self.policy, self.use_final_op_quality)

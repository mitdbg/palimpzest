from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

from palimpzest.policy import Policy
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan

logger = logging.getLogger(__name__)

class OptimizationStrategyType(str, Enum):
    """
    OptimizationStrategyType determines which (set of) plan(s) the Optimizer
    will return to the Execution layer.
    """
    GREEDY = "greedy"
    CONFIDENCE_INTERVAL = "confidence-interval"
    PARETO = "pareto" 
    SENTINEL = "sentinel"
    NONE = "none"
    AUTO = "auto"


class OptimizationStrategy(ABC):
    @abstractmethod
    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan] | list[SentinelPlan]:
        """Strategy decides how to search through the groups for optimal plan(s)"""
        pass

    @classmethod
    def get_strategy(cls, strategy_type: str) -> OptimizationStrategy:
        """Factory method to create strategy instances"""
        return OptimizerStrategyRegistry.get_strategy(strategy_type)

    def normalize_final_plans(self, plans: list[PhysicalPlan]) -> list[PhysicalPlan]:
        """
        For each plan in `plans`, this function enforces that the input schema of every
        operator is the output schema of the previous operator in the plan.

        Args:
            plans list[PhysicalPlan]: list of physical plans to normalize

        Returns:
            list[PhysicalPlan]: list of normalized physical plans
        """
        normalized_plans = []
        for plan in plans:
            normalized_ops = []
            for idx, op in enumerate(plan.operators):
                op_copy = op.copy()
                if idx == 0:
                    normalized_ops.append(op_copy)
                else:
                    op_copy.input_schema = plan.operators[-1].output_schema
                    normalized_ops.append(op_copy)
            normalized_plans.append(PhysicalPlan(operators=normalized_ops, plan_cost=plan.plan_cost))

        return normalized_plans


class GreedyStrategy(OptimizationStrategy):
    def _get_greedy_physical_plan(self, groups: dict, group_id: int) -> PhysicalPlan:
        """
        Return the best plan with respect to the user provided policy.
        """
        # get the best physical expression for this group
        best_phys_expr = groups[group_id].best_physical_expression

        # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
        # create and return the physical plan
        if len(best_phys_expr.input_group_ids) == 0:
            return PhysicalPlan(operators=[best_phys_expr.operator], plan_cost=best_phys_expr.plan_cost)

        # get the best physical plan(s) for this group's inputs
        input_group_id = best_phys_expr.input_group_ids[0] # TODO: need to handle joins
        input_best_phys_plan = self._get_greedy_physical_plan(groups, input_group_id)

        # add this operator to best physical plan and return
        return PhysicalPlan.from_ops_and_sub_plan([best_phys_expr.operator], input_best_phys_plan, best_phys_expr.plan_cost)

    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan]:
        logger.info(f"Getting greedy optimal plans for final group id: {final_group_id}")
        plans = [self._get_greedy_physical_plan(groups, final_group_id)]
        logger.info(f"Greedy optimal plans: {plans}")
        logger.info(f"Done getting greedy optimal plans for final group id: {final_group_id}")
        
        return plans


class ParetoStrategy(OptimizationStrategy):
    def _get_candidate_pareto_physical_plans(self, groups: dict, group_id: int, policy: Policy) -> list[PhysicalPlan]:
        """
        Return a list of plans which will contain all of the pareto optimal plans (and some additional
        plans which may not be pareto optimal).

        TODO: can we cache group_id --> final_pareto_optimal_plans to avoid re-computing upstream
        groups' pareto-optimal plans for each expression?
        """
        # get the pareto optimal physical expressions for this group
        pareto_optimal_phys_exprs = groups[group_id].pareto_optimal_physical_expressions

        # construct list of pareto optimal plans
        pareto_optimal_plans = []
        for phys_expr in pareto_optimal_phys_exprs:
            # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
            # create and return the physical plan
            if len(phys_expr.input_group_ids) == 0:
                for plan_cost, _ in phys_expr.pareto_optimal_plan_costs:
                    plan = PhysicalPlan(operators=[phys_expr.operator], plan_cost=plan_cost)
                    pareto_optimal_plans.append(plan)

            # otherwise, get the pareto optimal physical plan(s) for this group's inputs
            else:
                # get the pareto optimal physical plan(s) for this group's inputs
                input_group_id = phys_expr.input_group_ids[0] # TODO: need to handle joins
                pareto_optimal_phys_subplans = self._get_candidate_pareto_physical_plans(groups, input_group_id, policy)

                # iterate over the input subplans and find the one(s) which combine with this physical expression
                # to make a pareto-optimal plan
                for plan_cost, input_plan_cost in phys_expr.pareto_optimal_plan_costs:
                    for subplan in pareto_optimal_phys_subplans:
                        if (
                            subplan.plan_cost.cost == input_plan_cost.cost
                            and subplan.plan_cost.time == input_plan_cost.time
                            and subplan.plan_cost.quality == input_plan_cost.quality
                        ):
                            # TODO: The plan_cost gets summed with subplan.plan_cost;
                            #       am I defining expression.best_plan_cost to be the cost of that operator,
                            #       and expression.pareto_optimal_plan_costs to be the cost(s) of the subplan including that operator?
                            #       i.e. are my definitions inconsistent?
                            plan = PhysicalPlan.from_ops_and_sub_plan([phys_expr.operator], subplan, plan_cost)
                            pareto_optimal_plans.append(plan)

        return pareto_optimal_plans
    
    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan]:
        logger.info(f"Getting pareto optimal plans for final group id: {final_group_id}")
        # compute all of the pareto optimal physical plans
        plans = self._get_candidate_pareto_physical_plans(groups, final_group_id, policy)

        # adjust plans' plan_cost.quality to reflect only the quality of the final operator
        if use_final_op_quality:
            for plan in plans:
                plan.plan_cost.quality = plan.plan_cost.op_estimates.quality

        # filter pareto optimal plans for ones which satisfy policy constraint (if at least one of them does)
        # import pdb; pdb.set_trace()
        if any([policy.constraint(plan.plan_cost) for plan in plans]):
            plans = [plan for plan in plans if policy.constraint(plan.plan_cost)]

        # select the plan which is best for the given policy
        optimal_plan, plans = plans[0], plans[1:]
        for plan in plans:
            optimal_plan = optimal_plan if policy.choose(optimal_plan.plan_cost, plan.plan_cost) else plan

        plans = [optimal_plan]
        logger.info(f"Pareto optimal plans: {plans}")
        logger.info(f"Done getting pareto optimal plans for final group id: {final_group_id}")
        return plans
    

class SentinelStrategy(OptimizationStrategy):
    def _get_sentinel_plan(self, groups: dict, group_id: int) -> SentinelPlan:
        """
        Create and return a SentinelPlan object.
        """
        # get all the physical expressions for this group
        phys_exprs = groups[group_id].physical_expressions
        phys_op_set = [expr.operator for expr in phys_exprs]

        # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
        # create and return the physical plan
        best_phys_expr = groups[group_id].best_physical_expression
        if len(best_phys_expr.input_group_ids) == 0:
            return SentinelPlan(operator_sets=[phys_op_set])

        # TODO: need to handle joins
        # get the best physical plan(s) for this group's inputs
        best_phys_subplan = SentinelPlan(operator_sets=[])
        for input_group_id in best_phys_expr.input_group_ids:
            input_best_phys_plan = self._get_sentinel_plan(groups, input_group_id)
            best_phys_subplan = SentinelPlan.from_ops_and_sub_plan(best_phys_subplan.operator_sets, input_best_phys_plan)

        # add this operator set to best physical plan and return
        return SentinelPlan.from_ops_and_sub_plan([phys_op_set], best_phys_subplan)

    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[SentinelPlan]:
        logger.info(f"Getting sentinel optimal plans for final group id: {final_group_id}")
        plans = [self._get_sentinel_plan(groups, final_group_id)]
        logger.info(f"Sentinel optimal plans: {plans}")
        logger.info(f"Done getting sentinel optimal plans for final group id: {final_group_id}")
        return plans


class NoOptimizationStrategy(GreedyStrategy):
    """
    NoOptimizationStrategy is used to intentionally construct a PhysicalPlan without applying any
    logical transformations or optimizations. It uses the same get_optimal_plans logic as the
    GreedyOptimizationStrategy.
    """


class ConfidenceIntervalStrategy(OptimizationStrategy):
    def _get_confidence_interval_optimal_plans(self, groups: dict, group_id: int) -> list[PhysicalPlan]:
        """
        Return all physical plans whose upper bound on the primary policy metric is greater than the
        best plan's lower bound on the primary policy metric (subject to satisfying the policy constraint).

        The OptimizePhysicalExpression task guarantees that each group's `ci_best_physical_expressions`
        maintains a list of expressions with overlapping CI's on the primary policy metric (while also
        satisfying the policy constraint).

        This function computes the cross-product of all such expressions across all groups.
        """
        # get all the physical expressions which could be the best for this group
        best_phys_exprs = groups[group_id].ci_best_physical_expressions

        best_plans = []
        for phys_expr in best_phys_exprs:
            # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
            # create the physical plan and append it to the best_plans for this group
            if len(phys_expr.input_group_ids) == 0:
                plan = PhysicalPlan(operators=[phys_expr.operator], plan_cost=phys_expr.plan_cost)
                best_plans.append(plan)

            # otherwise, get the best physical plan(s) for this group's inputs
            else:
                # TODO: need to handle joins
                best_phys_subplans = [PhysicalPlan(operators=[])]
                for input_group_id in phys_expr.input_group_ids:
                    input_best_phys_plans = self._get_confidence_interval_optimal_plans(groups, input_group_id)
                    best_phys_subplans = [
                        PhysicalPlan.from_ops_and_sub_plan(subplan.operators, input_subplan, subplan.plan_cost)
                        for subplan in best_phys_subplans
                        for input_subplan in input_best_phys_plans
                    ]

                # add this operator to best physical plan and return
                for subplan in best_phys_subplans:
                    plan = PhysicalPlan.from_ops_and_sub_plan([phys_expr.operator], subplan, phys_expr.plan_cost)
                    best_plans.append(plan)

        return best_plans

    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan]:
        # TODO: fix this to properly handle multiple potential plans
        raise Exception("NotImplementedError")
        # plans = self._get_confidence_interval_optimal_plans(final_group_id)

class AutoOptimizationStrategy(OptimizationStrategy):
    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan]:
        raise NotImplementedError("Auto optimization strategy not implemented")


class OptimizerStrategyRegistry:
    """Registry to map strategy types to their implementations"""

    _strategies: dict[str, type[OptimizationStrategy]] = {
        OptimizationStrategyType.GREEDY.value: GreedyStrategy,
        OptimizationStrategyType.CONFIDENCE_INTERVAL.value: ConfidenceIntervalStrategy,
        OptimizationStrategyType.PARETO.value: ParetoStrategy,
        OptimizationStrategyType.SENTINEL.value: SentinelStrategy,
        OptimizationStrategyType.NONE.value: NoOptimizationStrategy,
        OptimizationStrategyType.AUTO.value: AutoOptimizationStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_type: str) -> OptimizationStrategy:
        """Get strategy instance by type"""
        strategy_class = cls._strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown optimization strategy: {strategy_type}")
        return strategy_class()

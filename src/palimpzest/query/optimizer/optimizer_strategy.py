from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from palimpzest.policy import Policy
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.query.optimizer.primitives import Group

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    @abstractmethod
    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan] | list[SentinelPlan]:
        """Strategy decides how to search through the groups for optimal plan(s)"""
        pass


class GreedyStrategy(OptimizationStrategy):
    def _get_greedy_physical_plan(self, groups: dict, group_id: int) -> PhysicalPlan:
        """
        Return the best plan with respect to the user provided policy.
        """
        # get the best physical expression for this group
        best_phys_expr = groups[group_id].best_physical_expression

        # if this expression has no inputs (i.e. it is a BaseScan), create and return the physical plan
        best_plan = None
        if len(best_phys_expr.input_group_ids) == 0:
            best_plan = PhysicalPlan(best_phys_expr.operator, subplans=None, plan_cost=best_phys_expr.plan_cost)

        # otherwise, if this expression is not a join (i.e. it has one input)
        elif len(best_phys_expr.input_group_ids) == 1:
            # get the best physical plan for this group's input
            input_group_id = best_phys_expr.input_group_ids[0]
            input_best_phys_plan = self._get_greedy_physical_plan(groups, input_group_id)

            # add this operator to best physical plan and return
            best_plan = PhysicalPlan(best_phys_expr.operator, subplans=[input_best_phys_plan], plan_cost=best_phys_expr.plan_cost)

        # otherwise, this expression is a join (i.e. it has two inputs)
        elif len(best_phys_expr.input_group_ids) == 2:
            left_input_group_id, right_input_group_id = best_phys_expr.input_group_ids

            # get the best physical plan for the left input
            left_best_phys_plan = self._get_greedy_physical_plan(groups, left_input_group_id)

            # get the best physical plan for the right input
            right_best_phys_plan = self._get_greedy_physical_plan(groups, right_input_group_id)

            # add this operator to best physical plan and return
            best_plan = PhysicalPlan(best_phys_expr.operator, subplans=[left_best_phys_plan, right_best_phys_plan], plan_cost=best_phys_expr.plan_cost)

        # add this operator to best physical plan and return
        return best_plan

    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[PhysicalPlan]:
        logger.info(f"Getting greedy optimal plans for final group id: {final_group_id}")
        plans = [self._get_greedy_physical_plan(groups, final_group_id)]
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
            # if this expression has no inputs (i.e. it is a BaseScan), create and return the physical plan
            if len(phys_expr.input_group_ids) == 0:
                for plan_cost, _ in phys_expr.pareto_optimal_plan_costs:
                    plan = PhysicalPlan(phys_expr.operator, subplans=None, plan_cost=plan_cost)
                    pareto_optimal_plans.append(plan)

            # otherwise, if this expression is not a join (i.e. it has one input)
            elif len(phys_expr.input_group_ids) == 1:
                # get the pareto optimal physical plan(s) for this group's inputs
                input_group_id = phys_expr.input_group_ids[0]
                pareto_optimal_phys_subplans = self._get_candidate_pareto_physical_plans(groups, input_group_id, policy)

                # iterate over the input subplans and find the one(s) which combine with this physical expression
                # to make a pareto-optimal plan
                for plan_cost, (input_plan_cost, _) in phys_expr.pareto_optimal_plan_costs:
                    for subplan in pareto_optimal_phys_subplans:
                        if subplan.plan_cost == input_plan_cost:
                            plan = PhysicalPlan(phys_expr.operator, subplans=[subplan], plan_cost=plan_cost)
                            pareto_optimal_plans.append(plan)

            # otherwise, this expression is a join (i.e. it has two inputs)
            elif len(phys_expr.input_group_ids) == 2:
                left_input_group_id, right_input_group_id = phys_expr.input_group_ids
                pareto_optimal_left_subplans = self._get_candidate_pareto_physical_plans(groups, left_input_group_id, policy)
                pareto_optimal_right_subplans = self._get_candidate_pareto_physical_plans(groups, right_input_group_id, policy)

                # iterate over the input subplans and find the one(s) which combine with this physical expression
                # to make a pareto-optimal plan
                for plan_cost, (left_input_plan_cost, right_input_plan_cost) in phys_expr.pareto_optimal_plan_costs:
                    for left_subplan in pareto_optimal_left_subplans:
                        if left_subplan.plan_cost == left_input_plan_cost:
                            for right_subplan in pareto_optimal_right_subplans:
                                if right_subplan.plan_cost == right_input_plan_cost:
                                    plan = PhysicalPlan(phys_expr.operator, subplans=[left_subplan, right_subplan], plan_cost=plan_cost)
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
        if any([policy.constraint(plan.plan_cost) for plan in plans]):
            plans = [plan for plan in plans if policy.constraint(plan.plan_cost)]

        # select the plan which is best for the given policy
        optimal_plan, plans = plans[0], plans[1:]
        for plan in plans:
            optimal_plan = optimal_plan if policy.choose(optimal_plan.plan_cost, plan.plan_cost) else plan

        plans = [optimal_plan]
        logger.info(f"Done getting pareto optimal plans for final group id: {final_group_id}")
        return plans
    

class SentinelStrategy(OptimizationStrategy):
    def _get_sentinel_plan(self, groups: dict[str, Group], group_id: int) -> SentinelPlan:
        """
        Create and return a SentinelPlan object.

        NOTE: this strategy is only used to construct a SentinelPlan before performing optimization.
              Currently, we do not perform any transformation rules when building the groups which
              are fed into this function. Thus, every physical expression will correspond to the same
              logical operator and share the same logical_op_id. Eventually we will want to consider
              multiple logical re-orderings of operators in our SentinelPlan, but for now it is static.
        """
        # get all the physical expressions for this group as well as their logical_op_id
        phys_exprs = groups[group_id].physical_expressions
        phys_op_set = [expr.operator for expr in phys_exprs]

        # if this expression has no inputs (i.e. it is a scan operator), create and return the sentinel plan
        best_phys_expr = groups[group_id].best_physical_expression
        if len(best_phys_expr.input_group_ids) == 0:
            return SentinelPlan(operator_set=phys_op_set, subplans=None)

        # get the subplans
        subplans = []
        for input_group_id in best_phys_expr.input_group_ids:
            subplan = self._get_sentinel_plan(groups, input_group_id)
            subplans.append(subplan)

        # compose the current physical operator set with its subplans
        return SentinelPlan(operator_set=phys_op_set, subplans=subplans)

    def get_optimal_plans(self, groups: dict, final_group_id: int, policy: Policy, use_final_op_quality: bool) -> list[SentinelPlan]:
        logger.info(f"Getting sentinel optimal plans for final group id: {final_group_id}")
        plans = [self._get_sentinel_plan(groups, final_group_id)]
        logger.info(f"Done getting sentinel optimal plans for final group id: {final_group_id}")
        return plans


class NoOptimizationStrategy(GreedyStrategy):
    """
    NoOptimizationStrategy is used to intentionally construct a PhysicalPlan without applying any
    logical transformations or optimizations. It uses the same get_optimal_plans logic as the
    GreedyOptimizationStrategy.
    """

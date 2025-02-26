from __future__ import annotations

import logging
from typing import Any

from palimpzest.core.data.dataclasses import PlanCost
from palimpzest.policy import Policy
from palimpzest.query.optimizer.cost_model import BaseCostModel
from palimpzest.query.optimizer.optimizer_strategy import OptimizationStrategyType
from palimpzest.query.optimizer.primitives import Expression, Group
from palimpzest.query.optimizer.rules import ImplementationRule, Rule, TransformationRule

logger = logging.getLogger(__name__)

class Task:
    """
    Base class for a task. Each task has a method called perform() which executes the task.
    Examples of tasks include optimizing and exploring groups, optimizing expressions, applying
    rules, and optimizing inputs / costing the full group tree.
    """

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        """
        NOTE: At the moment we do not make use of the context, but in the future
        this can be used to store required physical properties (e.g. sort conditions
        for the query) and bounds (e.g. the operator should not cost more than X).
        """
        raise NotImplementedError("Calling this method from an abstract base class.")


class OptimizeGroup(Task):
    """
    The task to optimize a group.

    This task pushes optimization tasks for the group's current logical and physical
    expressions onto the tasks stack. This will fully expand the space of possible
    logical and physical expressions for the group, because OptimizeLogicalExpression
    and OptimizePhysicalExpression tasks will indirectly schedule new tasks to apply
    rules and to optimize input groups and expressions.
    """

    def __init__(self, group_id: int):
        self.group_id = group_id

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        logger.debug(f"Optimizing group {self.group_id}")
        # get updated instance of the group to be optimized
        if context is None:
            context = {}
        group = groups[self.group_id]

        # if this group has already been optimized, there's nothing more to do
        if group.optimized:
            return []

        # otherwise, optimize all the logical expressions for the group
        new_tasks = []
        for logical_expr in group.logical_expressions:
            task = OptimizeLogicalExpression(logical_expr)
            new_tasks.append(task)

        # and optimize all of the physical expressions in the group
        for physical_expr in group.physical_expressions:
            task = OptimizePhysicalExpression(physical_expr)
            new_tasks.append(task)

        logger.debug(f"Done optimizing group {self.group_id}")
        logger.debug(f"New tasks: {len(new_tasks)}")
        return new_tasks


class ExpandGroup(Task):
    """
    The task to expand a group.

    NOTE: we currently do not use this task, but I'm keeping it around in case we need it
    once we add join operations.
    """

    def __init__(self, group_id: int):
        self.group_id = group_id

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        logger.debug(f"Expanding group {self.group_id}")

        # fetch group
        if context is None:
            context = {}
        group = groups[self.group_id]

        # if the group has been explored before, return []
        if group.explored:
            return []

        # for each logical_expr in the group, add a new OptimizeLogicalExpression() task to the queue
        new_tasks = []
        for logical_expr in group.logical_expressions:
            task = OptimizeLogicalExpression(logical_expr, exploring=True)
            new_tasks.append(task)

        # mark the group as explored and return tasks
        group.set_explored()

        logger.debug(f"Done expanding group {self.group_id}")
        logger.debug(f"New tasks: {len(new_tasks)}")
        return new_tasks


class OptimizeLogicalExpression(Task):
    """
    The task to optimize a (multi-)expression.

    This task filters for the subset of rules which may be applied to the given logical expression
    and schedules ApplyRule tasks for each rule.
    """

    def __init__(self, logical_expression: Expression, exploring: bool = False):
        self.logical_expression = logical_expression
        self.exploring = exploring

    def perform(
        self,
        transformation_rules: list[TransformationRule],
        implementation_rules: list[ImplementationRule],
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        logger.debug(f"Optimizing logical expression {self.logical_expression}")
        # if we're exploring, only apply transformation rules
        if context is None:
            context = {}
        rules = transformation_rules if self.exploring else transformation_rules + implementation_rules

        # filter out rules that have already been applied to logical expression
        rules = list(filter(lambda rule: rule.get_rule_id() not in self.logical_expression.rules_applied, rules))

        # filter for rules that match on this logical expression
        rules = list(filter(lambda rule: rule.matches_pattern(self.logical_expression), rules))

        # TODO compute priority (i.e. "promise") of the rules and sort in order of priority

        # apply rules, exploring the input group(s) of each pattern if necessary
        new_tasks = []
        for rule in rules:
            # TODO: if necessary, expand the input groups of the logical expression to see if they need to be expanded
            apply_rule_task = ApplyRule(rule, self.logical_expression, self.exploring)
            new_tasks.append(apply_rule_task)

        logger.debug(f"Done optimizing logical expression {self.logical_expression}")
        logger.debug(f"New tasks: {len(new_tasks)}")
        return new_tasks


class ApplyRule(Task):
    """
    The task to apply a transformation or implementation rule to a (multi-)expression.

    For TransformationRules, this task will:
    - apply the substitution, receiving new expressions and groups
    - filter the new expressions for ones which may already exist
      - NOTE: we don't filter new groups because this implicitly must be done by the
              transformation rule in order to assign the correct group_id to any
              new expressions.
    - add new expressions to their group's set of logical expressions
    - schedule OptimizeGroup and OptimizeLogicalExpression tasks

    For ImplementationRules, this task will:
    - apply the substitution, receiving new expressions and groups
    - filter the new expressions for ones which may already exist
    - add new expressions to their group's set of physical expressions
    - schedule OptimizePhysicalExpression tasks
    """

    def __init__(self, rule: Rule, logical_expression: Expression, exploring: bool = False):
        self.rule = rule
        self.logical_expression = logical_expression
        self.exploring = exploring

    def perform(
        self,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        context: dict[str, Any] | None = None,
        **physical_op_params,
    ) -> tuple[list[Task], int]:
        logger.debug(f"Applying rule {self.rule} to logical expression {self.logical_expression}")
        
        # check if rule has already been applied to this logical expression; return [] if so
        if context is None:
            context = {}
        if self.rule.get_rule_id() in self.logical_expression.rules_applied:
            return []

        # MAYBE ?TODO?: iterate over bindings for logical expression and rule?
        #               perhaps some rules can be applied more than once to an expression?

        # get the group of the logical expression
        group_id = self.logical_expression.group_id
        group = groups[group_id]

        # process new expressions; update groups and create new tasks as needed
        new_tasks = []
        if issubclass(self.rule, TransformationRule):
            # apply transformation rule
            new_expressions, new_groups = self.rule.substitute(
                self.logical_expression, groups, expressions, **physical_op_params
            )

            # filter out any expressions which are duplicates (i.e. they've been previously computed)
            new_expressions = [expr for expr in new_expressions if expr.get_expr_id() not in expressions]
            expressions.update({expr.get_expr_id(): expr for expr in new_expressions})

            # add all new groups to the groups mapping
            for group in new_groups:
                groups[group.group_id] = group
                task = OptimizeGroup(group.group_id)

            # add new expressions to their respective groups
            for expr in new_expressions:
                group = groups[expr.group_id]
                group.logical_expressions.add(expr)

            # NOTE: we place new tasks for groups on the top of the stack so that they may be
            #       optimized before we optimize expressions which take new groups as inputs
            # create new tasks for optimizing new logical expressions
            for expr in new_expressions:
                task = OptimizeLogicalExpression(expr, self.exploring)
                new_tasks.append(task)

            # create new tasks for optimizing new groups
            for group in new_groups:
                task = OptimizeGroup(group.group_id)
                new_tasks.append(task)

        else:
            # apply implementation rule
            new_expressions = self.rule.substitute(self.logical_expression, **physical_op_params)
            new_expressions = [expr for expr in new_expressions if expr.get_expr_id() not in expressions]
            costed_phys_op_ids = context['costed_phys_op_ids']
            if costed_phys_op_ids is not None:
                new_expressions = [expr for expr in new_expressions if expr.operator.get_op_id() in costed_phys_op_ids]
            expressions.update({expr.get_expr_id(): expr for expr in new_expressions})
            group.physical_expressions.update(new_expressions)

            # create new task
            for expr in new_expressions:
                task = OptimizePhysicalExpression(expr)
                new_tasks.append(task)

        # mark that the rule has been applied to the logical expression
        self.logical_expression.add_applied_rule(self.rule)

        logger.debug(f"Done applying rule {self.rule} to logical expression {self.logical_expression}")
        logger.debug(f"New tasks: {len(new_tasks)}")
        return new_tasks


class OptimizePhysicalExpression(Task):
    """
    The task to optimize a physical expression and derive its cost.

    This task computes the cost of input groups for the given physical expression (scheduling
    OptimizeGroup tasks if needed), computes the cost of the given expression, and then updates
    the expression's group depending on whether this expression is its `best_physical_expression`
    or in its `pareto_optimal_physical_expressions`.
    """

    def __init__(self, physical_expression: Expression, exploring: bool = False):
        self.physical_expression = physical_expression
        self.exploring = exploring

    def update_best_physical_expression(self, group: Group, policy: Policy) -> Group:
        """
        Update the best physical expression for the given group and policy (if necessary).
        """
        # get the PlanCosts for the current best expression and this physical expression
        best_plan_cost = (
            group.best_physical_expression.plan_cost if group.best_physical_expression is not None else None
        )
        expr_plan_cost = self.physical_expression.plan_cost

        # pre-compute whether or not this physical expression satisfies the policy constraint
        expr_satisfies_constraint = policy.constraint(expr_plan_cost)

        # if we do not have a best physical expression for the group, we set this to be the best expression
        if group.best_physical_expression is None:
            group.best_physical_expression = self.physical_expression
            group.satisfies_constraint = expr_satisfies_constraint

        # if the group currently satisfies the constraint, only update the best physical expression
        # if this expression also satisfies the constraint and is more policy optimal
        elif group.satisfies_constraint and expr_satisfies_constraint and policy.choose(expr_plan_cost, best_plan_cost):
            group.best_physical_expression = self.physical_expression

        # finally, if the group does not satisfy the constraint, update the best physical expression if
        # this expression does satisfy the constraint, or if it is more policy optimal
        elif not group.satisfies_constraint and (
            expr_satisfies_constraint or policy.choose(expr_plan_cost, best_plan_cost)
        ):
            group.best_physical_expression = self.physical_expression
            group.satisfies_constraint = expr_satisfies_constraint

        return group

    def _is_dominated(self, plan_cost: PlanCost, other_plan_cost: PlanCost, policy: Policy):
        """
        Return true if plan_cost is dominated by other_plan_cost and False otherwise.

        If plan costs are perfectly tied on dimensions of interest, other dimensions
        will be used as a tiebreaker.
        """
        # get the dictionary representation of this poicy
        policy_dict = policy.get_dict()

        # get the metrics which matter for this policy
        metrics_of_interest = {metric for metric, weight in policy_dict.items() if weight > 0.0}
        remaining_metrics = {metric for metric, weight in policy_dict.items() if weight == 0.0}

        # corner case: if the two plan costs are perfectly tied on all dimensions of interest,
        # use other dimensions as tiebreaker
        if (
            all([getattr(plan_cost, metric) == getattr(other_plan_cost, metric) for metric in metrics_of_interest])
            and plan_cost.op_estimates.cardinality == other_plan_cost.op_estimates.cardinality
        ):
            for metric in remaining_metrics:
                if metric == "cost" and plan_cost.cost < other_plan_cost.cost:  # noqa: SIM114
                    return False
                elif metric == "time" and plan_cost.time < other_plan_cost.time:  # noqa: SIM114
                    return False
                elif metric == "quality" and plan_cost.quality > other_plan_cost.quality:
                    return False

            # if plan_cost is dominated by other_plan_cost on remaining metrics, return True
            return True

        # normal case: identify whether plan_cost is dominated by other_plan_cost
        cost_dominated = True if policy_dict["cost"] == 0.0 else other_plan_cost.cost <= plan_cost.cost
        time_dominated = True if policy_dict["time"] == 0.0 else other_plan_cost.time <= plan_cost.time
        quality_dominated = True if policy_dict["quality"] == 0.0 else other_plan_cost.quality >= plan_cost.quality
        cardinality_dominated = other_plan_cost.op_estimates.cardinality <= plan_cost.op_estimates.cardinality

        return cost_dominated and time_dominated and quality_dominated and cardinality_dominated

    def _is_pareto_optimal(self, expr_plan_cost: PlanCost, pareto_optimal_physical_expressions: list[Expression], policy: Policy) -> bool:
        """
        Return True if expr_plan_cost is pareto optimal and False otherwise.
        """
        pareto_optimal = True
        for pareto_phys_expr in pareto_optimal_physical_expressions:
            for other_expr_plan_cost, _ in pareto_phys_expr.pareto_optimal_plan_costs:
                if self._is_dominated(expr_plan_cost, other_expr_plan_cost, policy):
                    pareto_optimal = False
                    break

        return pareto_optimal

    def update_pareto_optimal_physical_expressions(self, group: Group, policy: Policy) -> Group:
        """
        Update the pareto optimal physical expressions for the given group and policy (if necessary).
        """
        for pareto_expr_plan_cost, _ in self.physical_expression.pareto_optimal_plan_costs:
            # if the pareto optimal physical expressions are empty, set the pareto optimal
            # physical expressions to be this expression
            if group.pareto_optimal_physical_expressions is None:
                group.pareto_optimal_physical_expressions = [self.physical_expression]

            # otherwise, if this expression is pareto optimal, update the pareto frontier
            elif self._is_pareto_optimal(pareto_expr_plan_cost, group.pareto_optimal_physical_expressions, policy):
                all_physical_expressions = [self.physical_expression] + group.pareto_optimal_physical_expressions

                # compute the pareto optimal set of expressions (or plan costs)
                pareto_optimal_physical_expressions = []
                for idx, expr in enumerate(all_physical_expressions):
                    for plan_cost, _ in expr.pareto_optimal_plan_costs:
                        pareto_optimal = True

                        # check if any other_expr has a plan cost which dominates plan_cost
                        for other_idx, other_expr in enumerate(all_physical_expressions):
                            if idx == other_idx:
                                continue

                            # if plan is dominated by other_expr, set pareto_optimal = False and break
                            for other_plan_cost, _ in other_expr.pareto_optimal_plan_costs:
                                if self._is_dominated(plan_cost, other_plan_cost, policy):
                                    pareto_optimal = False
                                    break

                            # break early if plan_cost is already dominated by another expression's plan_cost
                            if not pareto_optimal:
                                break

                        # add expr to pareto frontier if it has at least one plan cost which is not dominated
                        if pareto_optimal:
                            pareto_optimal_physical_expressions.append(expr)

                            # we can break now because we've identified that this expression has a plan on the pareto frontier
                            break

                # set pareto optimal physical expressions for the group
                group.pareto_optimal_physical_expressions = pareto_optimal_physical_expressions

        return group

    def perform(
        self,
        cost_model: BaseCostModel,
        groups: dict[int, Group],
        policy: Policy,
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        logger.debug(f"Optimizing physical expression {self.physical_expression}")

        if context is None:
            context = {}

        # return if we've already computed the cost of this physical expression
        if (  # noqa: SIM114
            context['optimization_strategy_type'] in [OptimizationStrategyType.GREEDY, OptimizationStrategyType.SENTINEL, OptimizationStrategyType.NONE]
            and self.physical_expression.plan_cost is not None
        ):
            return []

        elif (
            context['optimization_strategy_type'] == OptimizationStrategyType.PARETO
            and self.physical_expression.pareto_optimal_plan_costs is not None
        ):
            return []

        # for expressions with an input group, compute the input plan cost(s)
        best_input_plan_cost = PlanCost(cost=0, time=0, quality=1)
        input_plan_costs = [PlanCost(cost=0, time=0, quality=1)]
        if len(self.physical_expression.input_group_ids) > 0:
            # get the input group
            input_group_id = self.physical_expression.input_group_ids[0]  # TODO: need to handle joins
            input_group = groups[input_group_id]

            # compute the input plan cost or list of input plan costs
            new_tasks = []
            if (
                context['optimization_strategy_type'] in [OptimizationStrategyType.GREEDY, OptimizationStrategyType.SENTINEL, OptimizationStrategyType.NONE]
                and input_group.best_physical_expression is not None
            ):
                # TODO: apply policy constraint here
                best_input_plan_cost = input_group.best_physical_expression.plan_cost

            elif (
                context['optimization_strategy_type'] == OptimizationStrategyType.PARETO
                and input_group.pareto_optimal_physical_expressions is not None
            ):
                # TODO: apply policy constraint here
                input_plan_costs = []
                for pareto_physical_expression in input_group.pareto_optimal_physical_expressions:
                    plan_costs = list(map(lambda tup: tup[0], pareto_physical_expression.pareto_optimal_plan_costs))
                    input_plan_costs.extend(plan_costs)

                # NOTE: this list will not necessarily be pareto-optimal, as a plan cost on the pareto frontier of
                # one pareto_optimal_physical_expression might be dominated by the plan cost on another physical
                # expression's pareto frontier; we handle this below by taking the pareto frontier of all_possible_plan_costs
                # de-duplicate equivalent plan costs; we will still reconstruct plans with equivalent cost in optimizer.py
                input_plan_costs = list(set(input_plan_costs))

            else:
                task = OptimizeGroup(input_group_id)
                new_tasks.append(task)

            # if not all input groups have been costed, we need to compute these first and then retry this task
            if len(new_tasks) > 0:
                return [self] + new_tasks

        group = groups[self.physical_expression.group_id]
        if context['optimization_strategy_type'] == OptimizationStrategyType.PARETO:
            # compute all possible plan costs for this physical expression given the pareto optimal input plan costs
            all_possible_plan_costs = []
            for input_plan_cost in input_plan_costs:
                op_plan_cost = cost_model(self.physical_expression.operator, input_plan_cost.op_estimates)

                # compute the total cost for this physical expression by summing its operator's PlanCost
                # with the input groups' total PlanCost; also set the op_estimates for this expression's operator
                full_plan_cost = op_plan_cost + input_plan_cost
                full_plan_cost.op_estimates = op_plan_cost.op_estimates
                all_possible_plan_costs.append((full_plan_cost, input_plan_cost))

            # reduce the set of possible plan costs to the subset which are pareto-optimal
            pareto_optimal_plan_costs = []
            for idx, (plan_cost, input_plan_cost) in enumerate(all_possible_plan_costs):
                pareto_optimal = True

                # check if any other_expr dominates expr
                for other_idx, (other_plan_cost, _) in enumerate(all_possible_plan_costs):
                    if idx == other_idx:
                        continue

                    # if plan is dominated by other_expr, set pareto_optimal = False and break
                    if self._is_dominated(plan_cost, other_plan_cost, policy):
                        pareto_optimal = False
                        break

                # add expr to pareto frontier if it's not dominated
                if pareto_optimal:
                    pareto_optimal_plan_costs.append((plan_cost, input_plan_cost))

            # set the pareto frontier of plan costs which can be obtained by this physical expression
            self.physical_expression.pareto_optimal_plan_costs = pareto_optimal_plan_costs

            # update the group's pareto optimal costs
            group = self.update_pareto_optimal_physical_expressions(group, policy)

        else:
            # otherwise, compute the cost of this operator given the optimal input plan cost
            op_plan_cost = cost_model(self.physical_expression.operator, best_input_plan_cost.op_estimates)

            # compute the total cost for this physical expression by summing its operator's PlanCost
            # with the input groups' total PlanCost; also set the op_estimates for this expression's operator
            full_plan_cost = op_plan_cost + best_input_plan_cost
            full_plan_cost.op_estimates = op_plan_cost.op_estimates
            self.physical_expression.plan_cost = full_plan_cost

            # update the best physical expression for the group
            group = self.update_best_physical_expression(group, policy)

        # set the group's optimized flag to True, store the updated group, and return
        group.optimized = True
        groups[self.physical_expression.group_id] = group

        logger.debug(f"Done optimizing physical expression {self.physical_expression}")
        return []

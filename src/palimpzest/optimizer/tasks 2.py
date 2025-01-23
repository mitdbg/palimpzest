from __future__ import annotations
from palimpzest.constants import OptimizationStrategy
from palimpzest.dataclasses import PlanCost
from palimpzest.cost_model import CostModel
from palimpzest.optimizer.primitives import Expression, Group
from palimpzest.optimizer.rules import TransformationRule, ImplementationRule, Rule
from palimpzest.policy import Policy
from typing import Any, Dict, List, Tuple


class Task:
    """
    Base class for a task. Each task has a method called perform() which executes the task.
    Examples of tasks include optimizing and exploring groups, optimizing expressions, applying
    rules, and optimizing inputs / costing the full group tree.
    """
    def perform(self, groups: Dict[int, Group], context: Dict[str, Any]={}) -> List[Task]:
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

    def perform(self, groups: Dict[int, Group], context: Dict[str, Any]={}) -> List[Task]:
        # get updated instance of the group to be optimized
        group = groups[self.group_id]

        # if this group has already been optimized, there's nothing more to do
        if group.best_physical_expression is not None:
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

        return new_tasks


class ExpandGroup(Task):
    """
    The task to expand a group.

    NOTE: we currently do not use this task, but I'm keeping it around in case we need it
    once we add join operations.
    """
    def __init__(self, group_id: int):
        self.group_id = group_id

    def perform(self, groups: Dict[int, Group], context: Dict[str, Any]={}) -> List[Task]:
        # fetch group
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
        transformation_rules: List[TransformationRule],
        implementation_rules: List[ImplementationRule],
        context: Dict[str, Any]={},
    ) -> List[Task]:
        # if we're exploring, only apply transformation rules
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

    def perform(self, groups: Dict[int, Group], expressions: Dict[int, Expression], context: Dict[str, Any]={}, **physical_op_params) -> Tuple[List[Task], int]:
        # check if rule has already been applied to this logical expression; return [] if so
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
            new_expressions, new_groups = self.rule.substitute(self.logical_expression, groups, expressions, **physical_op_params)

            # add all new groups to the groups mapping and create a task to optimize them
            for group in new_groups:
                groups[group.group_id] = group
                task = OptimizeGroup(group.group_id)
                new_tasks.append(task)

            # filter out any expressions which are duplicates (i.e. they've been previously computed)
            new_expressions = [expr for expr in new_expressions if expr.get_expr_id() not in expressions]
            expressions.update({expr.get_expr_id(): expr for expr in new_expressions})

            for expr in new_expressions:
                group = groups[expr.group_id]
                group.logical_expressions.add(expr)

                # create new task
                task = OptimizeLogicalExpression(expr, self.exploring)
                new_tasks.append(task)
        else:
            # apply implementation rule
            new_expressions = self.rule.substitute(self.logical_expression, **physical_op_params)
            new_expressions = [expr for expr in new_expressions if expr.get_expr_id() not in expressions]
            expressions.update({expr.get_expr_id(): expr for expr in new_expressions})
            group.physical_expressions.update(new_expressions)

            # create new task
            for expr in new_expressions:
                task = OptimizePhysicalExpression(expr)
                new_tasks.append(task)

        # mark that the rule has been applied to the logical expression
        self.logical_expression.add_applied_rule(self.rule)

        return new_tasks


class OptimizePhysicalExpression(Task):
    """
    The task to optimize a physical expression and derive its cost.

    This task computes the cost of input groups for the given physical expression (scheduling
    OptimizeGroup tasks if needed), computes the cost of the given expression, and then updates
    the expression's group depending on whether this expression is its best_physical_expression
    or in its ci_best_physical_expressions.
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
            group.best_physical_expression.plan_cost
            if group.best_physical_expression is not None
            else None
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
        elif (
            group.satisfies_constraint
            and expr_satisfies_constraint
            and policy.choose(expr_plan_cost, best_plan_cost)
        ):
            group.best_physical_expression = self.physical_expression

        # finally, if the group does not satisfy the constraint, update the best physical expression if
        # this expression does satisfy the constraint, or if it is more policy optimal
        elif (
            not group.satisfies_constraint
            and (expr_satisfies_constraint or policy.choose(expr_plan_cost, best_plan_cost))
        ):
            group.best_physical_expression = self.physical_expression
            group.satisfies_constraint = expr_satisfies_constraint

        return group

    def update_ci_best_physical_expressions(self, group: Group, policy: Policy) -> Group:
        """
        Update the CI best physical expressions for the given group and policy (if necessary).
        """
        # get the primary metric for the policy
        policy_metric = policy.get_primary_metric()

        # get the PlanCosts for the current best expression and this physical expression
        expr_plan_cost = self.physical_expression.plan_cost

        # pre-compute whether or not this physical expression satisfies the policy constraint
        expr_satisfies_constraint = policy.constraint(expr_plan_cost)

        # attribute names for lower and upper bounds
        lower_bound = f"{policy_metric}_lower_bound"
        upper_bound = f"{policy_metric}_upper_bound"

        # get the expression and plan's upper and lower bounds on the metric of interest
        expr_lower_bound = getattr(expr_plan_cost, lower_bound)
        expr_upper_bound = getattr(expr_plan_cost, upper_bound)
        group_lower_bound = getattr(group, lower_bound)

        # if the CI best physical expressions is empty, add this expression
        if group.ci_best_physical_expressions == []:
            group.ci_best_physical_expressions = [self.physical_expression]
            group.satisfies_constraint = expr_satisfies_constraint
            setattr(group, lower_bound, expr_lower_bound)
            setattr(group, upper_bound, expr_upper_bound)

        # if the group currently satisfies the constraint, only update the CI best physical expressions
        # if this expression also satisfies the constraint and has an upper bound on the policy metric
        # above the group's lower bound on the policy metric
        elif (
            group.satisfies_constraint
            and expr_satisfies_constraint
            and expr_upper_bound > group_lower_bound
        ):
            # filter out any current best expressions whose upper bound is below the lower bound of this expression
            group.ci_best_physical_expressions = [
                curr_expr for curr_expr in group.ci_best_physical_expressions
                if not getattr(curr_expr, upper_bound) < expr_lower_bound
            ]

            # add this expression to the CI best physical expressions
            group.ci_best_physical_expressions.append(self.physical_expression)

            # compute the upper and lower bounds for the group
            new_group_upper_bound = max(map(lambda expr: getattr(expr, upper_bound), group.ci_best_physical_expressions))
            new_group_lower_bound = max(map(lambda expr: getattr(expr, lower_bound), group.ci_best_physical_expressions))

            # set the new upper and lower bounds for the group
            setattr(group, lower_bound, new_group_lower_bound)
            setattr(group, upper_bound, new_group_upper_bound)

        # if the group does not satisfy the constraint and the expression does satisfy the constraint,
        # set the CI best physical expressions to be this expression
        elif not group.satisfies_constraint and expr_satisfies_constraint:
            group.ci_best_physical_expressions = [self.physical_expression]
            group.satisfies_constraint = expr_satisfies_constraint
            setattr(group, lower_bound, expr_lower_bound)
            setattr(group, upper_bound, expr_upper_bound)

        # finally, update the CI best physical expressions if the group does not satisfy the constraint
        # and the expression does not satisfy the constraint, but the expression has an upper bound on the
        # policy metric above the group's lower bound on the policy metric
        elif (
            not group.satisfies_constraint
            and not expr_satisfies_constraint
            and expr_upper_bound > group_lower_bound
        ):
            # filter out any current best expressions whose upper bound is below the lower bound of this expression
            group.ci_best_physical_expressions = [
                curr_expr for curr_expr in group.ci_best_physical_expressions
                if not getattr(curr_expr, upper_bound) < expr_lower_bound
            ]

            # add this expression to the CI best physical expressions
            group.ci_best_physical_expressions.append(self.physical_expression)

            # compute the upper and lower bounds for the group
            new_group_upper_bound = max(map(lambda expr: getattr(expr, upper_bound), group.ci_best_physical_expressions))
            new_group_lower_bound = max(map(lambda expr: getattr(expr, lower_bound), group.ci_best_physical_expressions))

            # set the new upper and lower bounds for the group
            setattr(group, lower_bound, new_group_lower_bound)
            setattr(group, upper_bound, new_group_upper_bound)
        
        return group


    def perform(self, cost_model: CostModel, groups: Dict[int, Group], policy: Policy, context: Dict[str, Any]={}) -> List[Task]:
        # return if we've already computed the cost of this physical expression
        if self.physical_expression.plan_cost is not None:
            return []

        # compute the cumulative cost of the input groups
        new_tasks = []
        total_input_plan_cost = PlanCost(cost=0, time=0, quality=1)
        source_op_estimates = None
        for input_group_id in self.physical_expression.input_group_ids:
            group = groups[input_group_id]
            if group.best_physical_expression is not None:
                expr_plan_cost = group.best_physical_expression.plan_cost
                # TODO: apply policy constraint here
                # NOTE: assumes sequential execution of input groups
                total_input_plan_cost += expr_plan_cost
                source_op_estimates = expr_plan_cost.op_estimates # TODO: this needs to be handled correctly for joins w/multiple inputs
            else:
                task = OptimizeGroup(input_group_id)
                new_tasks.append(task)

        # if not all input groups have been costed, we need to compute these first and then retry this task
        if len(new_tasks) > 0:
            return [self] + new_tasks

        # otherwise, compute the cost of this operator
        op_plan_cost = cost_model(self.physical_expression.operator, source_op_estimates)

        # compute the total cost for this physical expression by summing its operator's PlanCost
        # with the input groups' total PlanCost; also set the op_estimates for this expression's operator
        full_plan_cost = op_plan_cost + total_input_plan_cost
        full_plan_cost.op_estimates = op_plan_cost.op_estimates
        self.physical_expression.plan_cost = full_plan_cost

        group = groups[self.physical_expression.group_id]
        if context['optimization_strategy'] == OptimizationStrategy.OPTIMAL:
            group = self.update_best_physical_expression(group, policy)
            groups[self.physical_expression.group_id] = group

        elif context['optimization_strategy'] == OptimizationStrategy.CONFIDENCE_INTERVAL:
            group = self.update_best_physical_expression(group, policy)
            group = self.update_ci_best_physical_expressions(group, policy)
            groups[self.physical_expression.group_id] = group

        elif context['optimization_strategy'] == OptimizationStrategy.SENTINEL:
            group = self.update_best_physical_expression(group, policy)
            groups[self.physical_expression.group_id] = group

        return []

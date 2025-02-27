from __future__ import annotations

from palimpzest.core.lib.fields import Field
from palimpzest.query.operators.logical import LogicalOperator
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.optimizer.plan import PlanCost
from palimpzest.utils.hash_helpers import hash_for_id


class Expression:
    """
    An Expression (technically a "multi-expression") consists of either a logical operator
    (if it's a logical expression) or a physical operator (if it's a physical expression)
    and the group ids which are inputs to this expression
    """

    def __init__(
        self,
        operator: LogicalOperator | PhysicalOperator,
        input_group_ids: list[int],
        input_fields: dict[str, Field],
        depends_on_field_names: set[str],
        generated_fields: dict[str, Field],
        group_id: int | None = None,
    ):
        self.operator = operator
        self.input_group_ids = input_group_ids
        self.input_fields = input_fields
        self.depends_on_field_names = depends_on_field_names
        self.generated_fields = generated_fields
        self.group_id = group_id
        self.rules_applied = set()

        # NOTE: this will be the best possible plan cost achieved by this expression for some
        # greedy definition of "best"
        self.plan_cost: PlanCost | None = None

        # NOTE: this will be a list of tuples where each tuple has a (pareto-optimal) plan cost
        # and the input plan cost for which that pareto-optimal plan cost is attainable
        self.pareto_optimal_plan_costs: list[tuple[PlanCost, PlanCost]] | None = None

    def __eq__(self, other):
        return self.operator == other.operator and self.input_group_ids == other.input_group_ids

    def __str__(self):
        op_id = self.operator.get_logical_op_id() if isinstance(self.operator, LogicalOperator) else self.operator.get_op_id()
        return str(tuple(sorted(self.input_group_ids)) + (op_id, str(self.__class__.__name__)))

    def __hash__(self):
        hash_str = self.__str__()
        hash_id = int(hash_for_id(hash_str), 16)
        return hash_id

    def add_applied_rule(self, rule):
        self.rules_applied.add(rule.get_rule_id())

    def set_group_id(self, group_id: int) -> None:
        self.group_id = group_id

    def get_expr_id(self) -> int:
        return self.__hash__()


class LogicalExpression(Expression):
    pass


class PhysicalExpression(Expression):
    pass


class Group:
    """
    A group is a set of logically equivalent expressions (both logical (query trees) and physical (execution plans)).
    Represents the execution of an un-ordered set of logical operators.
    Maintains a set of logical multi-expressions and physical multi-expressions.
    """

    def __init__(self, logical_expressions: list[Expression], fields: dict[str, Field], properties: dict[str, set[str]]):
        self.logical_expressions = set(logical_expressions)
        self.physical_expressions = set()
        self.fields = fields
        self.explored = False
        self.best_physical_expression: PhysicalExpression | None = None
        self.pareto_optimal_physical_expressions: list[PhysicalExpression] | None = None
        self.optimized = False

        # properties of the Group which distinguish it from groups w/identical fields,
        # e.g. which filters, limits have been applied; is the output sorted, etc.
        self.properties = properties

        # compute the group id
        self.group_id = self.compute_group_id()

    def set_explored(self):
        self.explored = True

    def compute_group_id(self) -> int:
        # sort field names
        sorted_fields = sorted(self.fields.keys())

        # sort properties
        sorted_properties = []
        for key in sorted(self.properties.keys()):
            sorted_properties.extend(sorted(self.properties[key]))

        hash_str = str(tuple(sorted_fields + sorted_properties))
        hash_id = int(hash_for_id(hash_str), 16)
        return hash_id

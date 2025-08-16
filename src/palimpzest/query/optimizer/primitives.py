from __future__ import annotations

from pydantic.fields import FieldInfo

from palimpzest.query.operators.logical import LogicalOperator
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.optimizer import rules
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
        input_fields: dict[str, FieldInfo],
        depends_on_field_names: set[str],
        generated_fields: dict[str, FieldInfo],
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
        # and the tuple of input plan cost(s) for which that pareto-optimal plan cost is attainable;
        # the tuple of input plan cost(s) is (input_plan_cost, None) for non-join operators and
        # (left_input_plan_cost, right_input_plan_cost) for join operators
        self.pareto_optimal_plan_costs: list[tuple[PlanCost, tuple[PlanCost, PlanCost]]] | None = None

        # compute the expression id
        self.expr_id = self._compute_expr_id()

    def __eq__(self, other):
        return self.expr_id == other.expr_id

    def __str__(self):
        expr_str = f"{self.__class__.__name__}(group_id={self.group_id}, expr_id={self.expr_id})"
        expr_str += f"\n  - input_group_ids: {self.input_group_ids}"
        expr_str += f"\n  - input_fields: {self.input_fields}"
        expr_str += f"\n  - depends_on_field_names: {self.depends_on_field_names}"
        expr_str += f"\n  - generated_fields: {self.generated_fields}"
        expr_str += f"\n  - operator:\n{str(self.operator)}"
        return expr_str

    def __hash__(self):
        op_id = self.operator.get_logical_op_id() if isinstance(self.operator, LogicalOperator) else self.operator.get_full_op_id()
        hash_str = str(tuple(sorted(self.input_group_ids)) + (op_id, str(self.__class__.__name__)))
        hash_id = int(hash_for_id(hash_str), 16)
        return hash_id

    def _compute_expr_id(self) -> int:
        return self.__hash__()

    def add_applied_rule(self, rule: type[rules.Rule]):
        self.rules_applied.add(rule.get_rule_id())

    def set_group_id(self, group_id: int) -> None:
        self.group_id = group_id


class LogicalExpression(Expression):
    pass


class PhysicalExpression(Expression):
    
    @classmethod
    def from_op_and_logical_expr(cls, op: PhysicalOperator, logical_expression: LogicalExpression) -> PhysicalExpression:
        """Construct a PhysicalExpression given a physical operator and a logical expression."""
        return cls(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )


class Group:
    """
    A group is a set of logically equivalent expressions (both logical (query trees) and physical (execution plans)).
    Represents the execution of an un-ordered set of logical operators.
    Maintains a set of logical multi-expressions and physical multi-expressions.
    """

    def __init__(self, logical_expressions: list[LogicalExpression], fields: dict[str, FieldInfo], properties: dict[str, set[str]]):
        self.logical_expressions: set[LogicalExpression] = set(logical_expressions)
        self.physical_expressions: set[PhysicalExpression] = set()
        self.fields = fields
        self.explored = False
        self.best_physical_expression: PhysicalExpression | None = None
        self.pareto_optimal_physical_expressions: list[PhysicalExpression] | None = None
        self.optimized = False

        # properties of the Group which distinguish it from groups w/identical fields,
        # e.g. which filters, limits have been applied; is the output sorted, etc.
        self.properties = properties

        # compute the group id
        self.group_id = self._compute_group_id()

    def set_explored(self):
        self.explored = True

    def _compute_group_id(self) -> int:
        # sort field names
        sorted_fields = sorted(self.fields.keys())

        # sort properties
        sorted_properties = []
        for key in sorted(self.properties.keys()):
            sorted_properties.extend(sorted(self.properties[key]))

        hash_str = str(tuple(sorted_fields + sorted_properties))
        hash_id = int(hash_for_id(hash_str), 16)
        return hash_id

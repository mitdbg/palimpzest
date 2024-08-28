from __future__ import annotations
from palimpzest.constants import MAX_ID_CHARS
from palimpzest.operators import LogicalOperator, PhysicalOperator
from typing import Dict, List, Optional, Set, Union

import hashlib


class Expression:
    """
    An Expression (technically a "multi-expression") consists of either a logical operator
    (if it's a logical expression) or a physical operator (if it's a physical expression)
    and the group ids which are inputs to this expression
    """
    def __init__(
            self,
            operator: Union[LogicalOperator, PhysicalOperator],
            input_group_ids: List[int],
            input_fields: Set[str],
            generated_fields: Set[str],
            group_id: Optional[int] = None,
        ):
        self.operator = operator
        self.input_group_ids = input_group_ids
        self.input_fields = input_fields
        self.generated_fields = generated_fields
        self.group_id = group_id
        self.rules_applied = set()
        self.plan_cost = None

    def __eq__(self, other: Expression):
        return self.operator == other.operator and self.input_group_ids == other.input_group_ids

    def __hash__(self):
        hash_str = str(tuple(sorted(self.input_group_ids)) + (self.operator.get_op_id(), str(self.__class__.__name__)))
        hash_id = int(hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS], 16)
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
    def __init__(self, logical_expressions: List[Expression], fields: Set[str], properties: Dict[str, Set[str]]):
        self.logical_expressions = set(logical_expressions)
        self.physical_expressions = set()
        self.fields = fields
        self.explored = False
        self.best_physical_expression: PhysicalExpression = None
        self.ci_best_physical_expressions: List[PhysicalExpression] = []
        self.satisfies_constraint = False

        # properties of the Group which distinguish it from groups w/identical fields,
        # e.g. which filters, limits have been applied; is the output sorted, etc.
        self.properties = properties

        # compute the group id
        self.group_id = self.compute_group_id()

    def set_explored(self):
        self.explored = True

    def compute_group_id(self) -> int:
        # sort fields
        sorted_fields = sorted(self.fields)

        # sort properties
        sorted_properties = []
        for key in sorted(self.properties.keys()):
            sorted_properties.extend(sorted(self.properties[key]))

        hash_str = str(tuple(sorted_fields + sorted_properties))
        hash_id = int(hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS], 16)
        return hash_id

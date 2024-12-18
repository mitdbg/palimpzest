from __future__ import annotations

import hashlib

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.dataclasses import PlanCost
from palimpzest.operators.physical import PhysicalOperator


class Plan:
    """A generic Plan is a graph of nodes (#TODO a list for now).
    The main subclasses are a LogicalPlan, which is composed of logical Operators, and a PhysicalPlan,
    which is composed of physical Operators.
    Plans are typically generated by objects of class Planner, and consumed by several objects,
    e.g., Execution, CostModel, Optimizer, etc. etc.
    """

    operators = []

    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.operators)

    def __next__(self):
        return next(iter(self.operators))

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, idx: int):
        return self.operators[idx]

    def __str__(self):
        start = self.operators[0]
        plan_str = f" 0. {type(start).__name__} -> {start.output_schema.__name__} \n\n"

        for idx, operator in enumerate(self.operators[1:]):
            plan_str += f" {idx+1}. {str(operator)}\n"

        return plan_str

    def __repr__(self) -> str:
        return str(self)


class PhysicalPlan(Plan):
    def __init__(self, operators: list[PhysicalOperator], plan_cost: PlanCost | None = None):
        self.operators = operators
        self.plan_cost = plan_cost if plan_cost is not None else PlanCost(cost=0.0, time=0.0, quality=1.0)
        self.plan_id = self.compute_plan_id()

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different PhysicalPlan instances with the identical lists of operators will have equivalent plan_ids.
        """
        hash_str = str(tuple(op.get_op_id() for op in self.operators))
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

    def __eq__(self, other):
        return self.operators == other.operators

    def __hash__(self):
        return int(self.plan_id, 16)

    @staticmethod
    def from_ops_and_sub_plan(
        ops: list[PhysicalOperator], ops_plan_cost: PlanCost, sub_plan: PhysicalPlan
    ) -> PhysicalPlan:
        # create copies of all logical operators
        copy_sub_plan = [op.copy() for op in sub_plan.operators]
        copy_ops = [op.copy() for op in ops]

        # construct full set of operators
        copy_sub_plan.extend(copy_ops)

        # aggregate cost of ops and subplan
        full_plan_cost = sub_plan.plan_cost + ops_plan_cost
        full_plan_cost.op_estimates = ops_plan_cost.op_estimates

        # return the PhysicalPlan
        return PhysicalPlan(operators=copy_sub_plan, plan_cost=full_plan_cost)


class SentinelPlan(Plan):
    def __init__(self, operator_sets: list[set[PhysicalOperator]]):
        self.operator_sets = operator_sets
        self.plan_id = self.compute_plan_id()

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different SentinelPlan instances with the identical operator_sets will have equivalent plan_ids.
        """
        hash_str = str(tuple(op.get_op_id() for op_set in self.operator_sets for op in op_set))
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

    def __eq__(self, other):
        return self.operator_sets == other.operator_sets

    def __hash__(self):
        return int(self.plan_id, 16)

    def __str__(self):
        # making the assumption that first operator_set can only be a scan
        start = list(self.operator_sets[0])[0]
        plan_str = f" 0. {type(start).__name__} -> {start.output_schema.__name__} \n\n"

        for idx, operator_set in enumerate(self.operator_sets[1:]):
            if len(operator_set) == 1:
                operator = list(operator_set)[0]
                plan_str += f" {idx+1}. {str(operator)}\n"

            else:
                for inner_idx, operator in enumerate(operator_set):
                    plan_str += f" {idx+1}.{inner_idx+1}. {str(operator)}\n"

        return plan_str

    @staticmethod
    def from_ops_and_sub_plan(op_sets: list[set[PhysicalOperator]], sub_plan: SentinelPlan) -> SentinelPlan:
        # create copies of all logical operators
        copy_sub_plan = [{op.copy() for op in op_set} for op_set in sub_plan.operator_sets]
        copy_ops = [{op.copy() for op in op_set} for op_set in op_sets]

        # construct full set of operators
        copy_sub_plan.extend(copy_ops)

        # return the SentinelPlan
        return SentinelPlan(operator_sets=copy_sub_plan)

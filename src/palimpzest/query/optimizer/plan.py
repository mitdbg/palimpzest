from __future__ import annotations

from abc import ABC, abstractmethod

from palimpzest.core.data.dataclasses import PlanCost
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.utils.hash_helpers import hash_for_id


class Plan(ABC):
    @abstractmethod
    def compute_plan_id(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, slice) -> tuple:
        pass

    @abstractmethod
    def __iter__(self) -> iter:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

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
        return hash_for_id(hash_str)

    def __eq__(self, other):
        return isinstance(other, PhysicalPlan) and self.plan_id == other.plan_id

    def __hash__(self):
        return int(self.plan_id, 16)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        start = self.operators[0]
        plan_str = f" 0. {type(start).__name__} -> {start.output_schema.__name__} \n\n"

        for idx, operator in enumerate(self.operators[1:]):
            plan_str += f" {idx+1}. {str(operator)}\n"

        return plan_str

    def __getitem__(self, slice):
        return self.operators[slice]

    def __iter__(self):
        return iter(self.operators)

    def __len__(self):
        return len(self.operators)

    @staticmethod
    def from_ops_and_sub_plan(ops: list[PhysicalOperator], sub_plan: PhysicalPlan, plan_cost: PlanCost) -> PhysicalPlan:
        # create copies of all logical operators
        copy_sub_plan = [op.copy() for op in sub_plan.operators]
        copy_ops = [op.copy() for op in ops]

        # construct full set of operators
        copy_sub_plan.extend(copy_ops)

        # return the PhysicalPlan
        return PhysicalPlan(operators=copy_sub_plan, plan_cost=plan_cost)


class SentinelPlan(Plan):
    def __init__(self, operator_sets: list[list[PhysicalOperator]]):
        # enforce that first operator_set is a scan and that every operator_set has at least one operator
        if len(operator_sets) > 0:
            assert isinstance(operator_sets[0][0], ScanPhysicalOp), "first operator set must be a scan"
            assert all(len(op_set) > 0 for op_set in operator_sets), "every operator set must have at least one operator"

        # store operator_sets and logical_op_ids; sort operator_sets internally by op_id
        self.operator_sets = operator_sets
        self.operator_sets = [sorted(op_set, key=lambda op: op.get_op_id()) for op_set in self.operator_sets]
        self.logical_op_ids = [op_set[0].logical_op_id for op_set in self.operator_sets]
        self.plan_id = self.compute_plan_id()

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different SentinelPlan instances with the identical operator_sets will have equivalent plan_ids.
        """
        hash_str = ""
        for logical_op_id, op_set in zip(self.logical_op_ids, self.operator_sets):
            hash_str += f"{logical_op_id} {tuple(op.get_op_id() for op in op_set)} "
        return hash_for_id(hash_str)

    def __eq__(self, other):
        return isinstance(other, SentinelPlan) and self.plan_id == other.plan_id

    def __hash__(self):
        return int(self.plan_id, 16)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        # by assertion, first operator_set is guaranteed to be a scan
        start = self.operator_sets[0][0]
        plan_str = f" 0. {type(start).__name__} -> {start.output_schema.__name__} \n\n"

        # build string one operator set at a time
        for idx, operator_set in enumerate(self.operator_sets[1:]):
            if len(operator_set) == 1:
                operator = operator_set[0]
                plan_str += f" {idx+1}. {str(operator)}\n"

            else:
                for inner_idx, operator in enumerate(operator_set):
                    plan_str += f" {idx+1}.{inner_idx+1}. {str(operator)}\n"

        return plan_str

    def __getitem__(self, slice):
        return self.logical_op_ids[slice], self.operator_sets[slice]

    def __iter__(self):
        yield from zip(self.logical_op_ids, self.operator_sets)

    def __len__(self):
        return len(self.logical_op_ids)

    @staticmethod
    def from_ops_and_sub_plan(op_sets: list[list[PhysicalOperator]], sub_plan: SentinelPlan) -> SentinelPlan:
        # create copies of all logical operators
        copy_sub_plan = [[op.copy() for op in op_set] for op_set in sub_plan.operator_sets]
        copy_ops = [[op.copy() for op in op_set] for op_set in op_sets]

        # construct full set of operators
        copy_sub_plan.extend(copy_ops)

        # return the SentinelPlan
        return SentinelPlan(operator_sets=copy_sub_plan)

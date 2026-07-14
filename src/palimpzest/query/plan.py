from __future__ import annotations

from abc import ABC, abstractmethod

from palimpzest.core.models import PlanCost
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, MarshalAndScanDataOp
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
    def __init__(self, operator: PhysicalOperator, subplans: list[PhysicalPlan] | None, plan_cost: PlanCost | None = None):
        self.operator = operator
        self.subplans = [] if subplans is None else subplans
        self.plan_cost = plan_cost if plan_cost is not None else PlanCost(cost=0.0, time=0.0, quality=1.0)
        self.plan_id = self.compute_plan_id()

        # NOTE: unique full_op_id is constructed as "{topological_index}-{full_op_id}" to
        # differentiate between multiple instances of the same physical operator e.g. in self-joins

        # compute mapping from unique full_op_id to next unique full_op_id in the plan
        self.unique_full_op_id_to_next_unique_full_op_and_id = {}
        current_idx, _ = self._compute_next_unique_full_op_map(self.unique_full_op_id_to_next_unique_full_op_and_id)
        self.unique_full_op_id_to_next_unique_full_op_and_id[f"{current_idx}-{self.operator.get_full_op_id()}"] = (None, None)

        # compute mapping from unique full_op_id to upstream unique full_op_ids
        self.unique_full_op_id_to_upstream_full_op_ids = {}
        self._compute_upstream_unique_full_op_ids_map(self.unique_full_op_id_to_upstream_full_op_ids)

        # compute mapping from unique full_op_id to source unique full_op_ids
        self.unique_full_op_id_to_source_full_op_ids = {}
        self._compute_source_unique_full_op_ids_map(self.unique_full_op_id_to_source_full_op_ids)

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different PhysicalPlan instances with the identical lists of operators will have equivalent plan_ids.
        """
        full_op_id = self.operator.get_full_op_id()
        subplan_ids = [subplan.compute_plan_id() for subplan in self.subplans]
        return hash_for_id(str((full_op_id,) + tuple(subplan_ids)))

    def get_est_total_outputs(self, num_samples: int | None = None, current_idx: int | None = None, source_unique_full_op_ids_map: dict | None = None) -> tuple[dict[str, int], int]:
        """Return the estimated total number of output records to be processed by the given operator in this plan."""
        # get the source map from the root of the entire plan; use this map throughout all recursive calls
        # (if you call self.get_source_unique_full_op_ids() from a subplan, it's topo indexes will be different)
        if source_unique_full_op_ids_map is None:
            source_unique_full_op_ids_map = self.unique_full_op_id_to_source_full_op_ids

        # get the estimated total outputs from all subplans
        # NOTE: this will be an empty dictionary for scans
        all_subplan_total_outputs = {}
        for subplan in self.subplans:
            subplan_total_outputs, current_idx = subplan.get_est_total_outputs(num_samples, current_idx, source_unique_full_op_ids_map)
            current_idx += 1
            all_subplan_total_outputs.update(subplan_total_outputs)

        # if current_idx is None, this is the first call, so we initialize it to 0
        if current_idx is None:
            current_idx = 0

        # get total outputs for this operator
        this_op_total_outputs = {}
        this_unique_full_op_id = f"{current_idx}-{self.operator.get_full_op_id()}"

        # if this operator is a scan, return the length of its datasource
        if isinstance(self.operator, MarshalAndScanDataOp):
            total = min(len(self.operator.datasource), num_samples) if num_samples is not None else len(self.operator.datasource)
            this_op_total_outputs = {this_unique_full_op_id: total}

        # if this operator is a context scan, return 1
        elif isinstance(self.operator, ContextScanOp):  # noqa: SIM114
            this_op_total_outputs = {this_unique_full_op_id: 1}

        # if this operator is an aggregate, return 1
        elif isinstance(self.operator, AggregateOp):
            this_op_total_outputs = {this_unique_full_op_id: 1}

        # if this operator is a limit scan, return its limit
        elif isinstance(self.operator, LimitScanOp):
            this_op_total_outputs = {this_unique_full_op_id: self.operator.limit}

        # if this operator is a join, return the Cartesian product of the estimated outputs of its inputs
        elif isinstance(self.operator, JoinOp):
            # get estimated outputs for immediate left and right inputs
            source_unique_full_op_ids = source_unique_full_op_ids_map[f"{current_idx}-{self.operator.get_full_op_id()}"]
            left_unique_full_op_id, right_unique_full_op_id = source_unique_full_op_ids[0], source_unique_full_op_ids[1]
            left_total_outputs = all_subplan_total_outputs[left_unique_full_op_id]
            right_total_outputs = all_subplan_total_outputs[right_unique_full_op_id]
            this_op_total_outputs = {this_unique_full_op_id: left_total_outputs * right_total_outputs}

        # otherwise, return the number of outputs from the immediate input
        else:
            source_unique_full_op_ids = source_unique_full_op_ids_map[f"{current_idx}-{self.operator.get_full_op_id()}"]
            source_unique_full_op_id = source_unique_full_op_ids[0]
            this_op_total_outputs = {this_unique_full_op_id: all_subplan_total_outputs[source_unique_full_op_id]}

        return {**this_op_total_outputs, **all_subplan_total_outputs}, current_idx

    def _compute_next_unique_full_op_map(self, next_map: dict[str, str | None], current_idx: int | None = None) -> tuple[int, str]:
        """Compute a mapping from each operator's unique full_op_id to the next operator in the plan and its unique full_op_id.

        The unique full_op_id is constructed as "{topological_index}-{full_op_id}" to differentiate between
        multiple instances of the same physical operator in the plan (e.g., in self-joins).

        Args:
            next_map: A dictionary to populate with the mapping from unique full_op_id to next (operator, unique_full_op_id) pair.
            current_idx: The current topological index in the plan. If None, starts at 0.

        Returns:
            A tuple containing:
                - The current topological index after processing this plan.
                - The unique full_op_id of this plan's root operator.
        """
        # If there are subplans, compute their next maps first
        subplan_topo_idx_op_id_pairs = []
        for subplan in self.subplans:
            current_idx, current_full_op_id = subplan._compute_next_unique_full_op_map(next_map, current_idx)
            subplan_topo_idx_op_id_pairs.append((current_idx, current_full_op_id))
            current_idx += 1  # increment after processing each subplan

        # for each subplan's root operator, set its next to this plan's root operator
        for topo_idx, full_op_id in subplan_topo_idx_op_id_pairs:
            unique_op_id = f"{topo_idx}-{full_op_id}"
            this_unique_op_id = f"{current_idx}-{self.operator.get_full_op_id()}"
            next_map[unique_op_id] = (self.operator, this_unique_op_id)

        # if this is the first call, initialize current_idx
        if current_idx is None:
            current_idx = 0

        return current_idx, self.operator.get_full_op_id()

    def get_next_unique_full_op_and_id(self, topo_idx: int, operator: PhysicalOperator) -> tuple[PhysicalOperator | None, str | None]:
        """Return the next operator in the plan after the given operator, or None if it is the last operator."""
        unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
        return self.unique_full_op_id_to_next_unique_full_op_and_id[unique_full_op_id]

    def get_next_unique_full_op_id(self, topo_idx: int, operator: PhysicalOperator) -> str | None:
        """Return the full_op_id of the next operator in the plan after the given operator, or None if it is the last operator."""
        unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
        _, next_unique_full_op_id = self.unique_full_op_id_to_next_unique_full_op_and_id[unique_full_op_id]
        return next_unique_full_op_id

    def _compute_upstream_unique_full_op_ids_map(self, upstream_map: dict[str, list[str]], current_idx: int | None = None) -> tuple[int, str, list[str]]:
        # set the upstream unique full_op_ids for this operator
        subplan_topo_idx_upstream_unique_full_op_id_tuples = []
        for subplan in self.subplans:
            current_idx, full_op_id, subplan_upstream_unique_full_op_ids = subplan._compute_upstream_unique_full_op_ids_map(upstream_map, current_idx)
            subplan_topo_idx_upstream_unique_full_op_id_tuples.append((current_idx, full_op_id, subplan_upstream_unique_full_op_ids))
            current_idx += 1

        # if current_idx is None, this is the first call, so we initialize it to 0
        if current_idx is None:
            current_idx = 0

        # compute this operator's unique full_op_id
        this_unique_full_op_id = f"{current_idx}-{self.operator.get_full_op_id()}"

        # update the upstream_map for this operator
        upstream_map[this_unique_full_op_id] = []
        for topo_idx, full_op_id, upstream_unique_full_op_ids in subplan_topo_idx_upstream_unique_full_op_id_tuples:
            subplan_upstream_unique_full_op_ids = [f"{topo_idx}-{full_op_id}"] + upstream_unique_full_op_ids
            upstream_map[this_unique_full_op_id].extend(subplan_upstream_unique_full_op_ids)

        # return the current index and the upstream unique full_op_ids for this operator
        return current_idx, self.operator.get_full_op_id(), upstream_map[this_unique_full_op_id]

    def get_upstream_unique_full_op_ids(self, unique_full_op_id: str) -> list[str]:
        """Return the list of unique full_op_ids for the upstream operators of the operator specified by `unique_full_op_id`."""
        return self.unique_full_op_id_to_upstream_full_op_ids[unique_full_op_id]

    def _compute_source_unique_full_op_ids_map(self, source_map: dict[str, list[str]], current_idx: int | None = None) -> tuple[int, str]:
        # get the topological index and full_op_id pairs for all subplans' root operators
        subplan_topo_idx_op_id_pairs = []
        for subplan in self.subplans:
            current_idx, current_full_op_id = subplan._compute_source_unique_full_op_ids_map(source_map, current_idx)
            subplan_topo_idx_op_id_pairs.append((current_idx, current_full_op_id))
            current_idx += 1

        # if current_idx is None, this is the first call, so we initialize it to 0
        if current_idx is None:
            current_idx = 0

        # compute this operator's unique full_op_id
        this_unique_full_op_id = f"{current_idx}-{self.operator.get_full_op_id()}"

        # update the source_map for this operator
        source_map[this_unique_full_op_id] = []
        for topo_idx, full_op_id in subplan_topo_idx_op_id_pairs:
            unique_full_op_id = f"{topo_idx}-{full_op_id}"
            source_map[this_unique_full_op_id].append(unique_full_op_id)

        # return the current unique full_op_id for this operator
        return current_idx, self.operator.get_full_op_id()

    def get_source_unique_full_op_ids(self, topo_idx: int, operator: PhysicalOperator) -> list[str]:
        """Return the list of unique full_op_ids for the input(s) to this operator."""
        unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
        return self.unique_full_op_id_to_source_full_op_ids[unique_full_op_id]

    def __eq__(self, other):
        return isinstance(other, PhysicalPlan) and self.plan_id == other.plan_id

    def __hash__(self):
        return int(self.plan_id, 16)

    def __repr__(self) -> str:
        return str(self)

    def _get_str(self, idx: int = 0, indent: int = 0) -> str:
        indent_str = " " * (indent * 2)
        plan_str = f"{indent_str}{idx}. {str(self.operator)}\n"
        for subplan in self.subplans:
            plan_str += subplan._get_str(idx=idx + 1, indent=indent + 1)

        return plan_str

    def __str__(self):
        return self._get_str()

    def __getitem__(self, slice):
        ops = [op for op in self]
        return ops[slice]

    def __iter__(self):
        for subplan in self.subplans:
            yield from subplan
        yield self.operator

    def __len__(self):
        return 1 + sum(len(subplan) for subplan in self.subplans)

    @classmethod
    def _from_ops(cls, ops: list[PhysicalOperator], plan_cost: PlanCost | None = None) -> PhysicalPlan:
        """
        NOTE: Do not use this in production code. This is a convenience method for constructing PhysicalPlans in tests.
        This method assumes a left-deep tree structure (i.e. pipeline), where each operator has at most one subplan.
        The PlanCost is applied to all subplans, thus it is not a true representation of the cost of the plan.
        """
        assert len(ops) > 0, "ops must contain at least one PhysicalOperator"

        # build the PhysicalPlan from the list of operators
        if len(ops) == 1:
            return cls(operator=ops[0], subplans=None, plan_cost=plan_cost)

        # recursively build subplans
        subplan = cls._from_ops(ops[:-1], plan_cost=plan_cost)
        return cls(operator=ops[-1], subplans=[subplan], plan_cost=plan_cost)


# TODO(?): take list[PhysicalOperator] as input, but then store OpFrontier
class SentinelPlan(Plan):
    def __init__(self, operator_set: list[PhysicalOperator], subplans: list[SentinelPlan] | None):
        # store operator_set and logical_op_id; sort operator_set internally by full_op_id
        self.operator_set = sorted(operator_set, key=lambda op: op.get_full_op_id())
        self.logical_op_id = self.operator_set[0].logical_op_id
        self.subplans = [] if subplans is None else subplans
        self.plan_id = self.compute_plan_id()

        # compute mapping from unique logical_op_id to next unique logical_op_id in the plan
        self.unique_logical_op_id_to_next_unique_logical_op_id = {}
        current_idx, _ = self._compute_next_unique_logical_op_id_map(self.unique_logical_op_id_to_next_unique_logical_op_id)
        self.unique_logical_op_id_to_next_unique_logical_op_id[f"{current_idx}-{self.logical_op_id}"] = None

        # compute mapping from unique logical_op_id to root dataset ids
        self.unique_logical_op_id_to_root_dataset_ids = {}
        self._compute_root_dataset_ids_map(self.unique_logical_op_id_to_root_dataset_ids)

        # compute mapping from unique logical_op_id to source unique logical_op_ids
        self.unique_logical_op_id_to_source_logical_op_ids = {}
        self._compute_source_unique_logical_op_ids_map(self.unique_logical_op_id_to_source_logical_op_ids)

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different SentinelPlan instances with the identical operator_sets will have equivalent plan_ids.
        """
        full_id = (self.logical_op_id,) + tuple([op.get_full_op_id() for op in self.operator_set])
        subplan_ids = [subplan.compute_plan_id() for subplan in self.subplans]
        return hash_for_id(str((full_id,) + tuple(subplan_ids)))

    def __eq__(self, other):
        return isinstance(other, SentinelPlan) and self.plan_id == other.plan_id

    def __hash__(self):
        return int(self.plan_id, 16)

    def __repr__(self) -> str:
        return str(self)

    def _get_str(self, idx: int = 0, indent: int = 0) -> str:
        indent_str = " " * (indent * 2)
        operator = self.operator_set[0]
        inner_idx_str = "" if len(self.operator_set) == 1 else f"1 - {len(self.operator_set)}."
        plan_str = f"{indent_str}{idx}.{inner_idx_str} {str(operator)}\n"
        for subplan in self.subplans:
            plan_str += subplan._get_str(idx=idx + 1, indent=indent + 1)

        return plan_str

    def __str__(self):
        return self._get_str()

    def __getitem__(self, slice):
        op_set_tuples = [op_set_tuple for op_set_tuple in self]
        return op_set_tuples[slice]

    def __iter__(self):
        for subplan in self.subplans:
            yield from subplan
        yield self.logical_op_id, self.operator_set

    def __len__(self):
        return 1 + sum(len(subplan) for subplan in self.subplans)
    
    def _compute_next_unique_logical_op_id_map(self, next_map: dict[str, str | None], current_idx: int | None = None) -> tuple[int, str]:
        """Compute a mapping from each operator's unique logical_op_id to the next operator's unique logical_op_id.

        The unique logical_op_id is constructed as "{topological_index}-{logical_op_id}" to differentiate between
        multiple instances of the same physical operator in the plan (e.g., in self-joins).

        Args:
            next_map: A dictionary to populate with the mapping from unique logical_op_id to next logical_op_id.
            current_idx: The current topological index in the plan. If None, starts at 0.

        Returns:
            A tuple containing:
                - The current topological index after processing this plan.
                - The unique logical_op_id of this plan's root logical operator.
        """
        # If there are subplans, compute their next maps first
        subplan_topo_idx_op_id_pairs = []
        for subplan in self.subplans:
            current_idx, current_logical_op_id = subplan._compute_next_unique_logical_op_id_map(next_map, current_idx)
            subplan_topo_idx_op_id_pairs.append((current_idx, current_logical_op_id))
            current_idx += 1  # increment after processing each subplan

        # for each subplan's root operator, set its next to this plan's root operator
        for topo_idx, logical_op_id in subplan_topo_idx_op_id_pairs:
            unique_logical_op_id = f"{topo_idx}-{logical_op_id}"
            this_unique_logical_op_id = f"{current_idx}-{self.logical_op_id}"
            next_map[unique_logical_op_id] = this_unique_logical_op_id

        # if this is the first call, initialize current_idx
        if current_idx is None:
            current_idx = 0

        return current_idx, self.logical_op_id

    def get_next_unique_logical_op_id(self, unique_logical_op_id: str) -> str | None:
        """Return the unique logical_op_id of the next operator in the plan after the given operator, or None if it is the last operator."""
        return self.unique_logical_op_id_to_next_unique_logical_op_id[unique_logical_op_id]

    def _compute_root_dataset_ids_map(self, root_dataset_ids_map: dict[str, list[str]], current_idx: int | None = None) -> tuple[int, list[str]]:
        # set the root dataset ids for this operator
        all_subplan_root_dataset_ids = []
        for subplan in self.subplans:
            current_idx, subplan_root_dataset_ids = subplan._compute_root_dataset_ids_map(root_dataset_ids_map, current_idx)
            all_subplan_root_dataset_ids.extend(subplan_root_dataset_ids)
            current_idx += 1

        # if current_idx is None, this is the first call, so we initialize it to 0
        if current_idx is None:
            current_idx = 0

        # compute this operator's unique logical_op_id
        this_unique_logical_op_id = f"{current_idx}-{self.logical_op_id}"

        # if this operator is a root dataset scan, update root_dataset_ids
        root_dataset_ids = []
        if isinstance(self.operator_set[0], MarshalAndScanDataOp):
            root_dataset_ids.append(self.operator_set[0].datasource.id)
        elif isinstance(self.operator_set[0], ContextScanOp):
            root_dataset_ids.append(self.operator_set[0].context.id)

        # update the root_dataset_ids_map for this operator
        root_dataset_ids_map[this_unique_logical_op_id] = root_dataset_ids + all_subplan_root_dataset_ids

        # return the current index and the upstream unique logical_op_ids for this operator
        return current_idx, root_dataset_ids_map[this_unique_logical_op_id]

    def get_root_dataset_ids(self, unique_logical_op_id: str) -> list[str]:
        """Return the list of root dataset ids which are upstream of this operator."""
        return self.unique_logical_op_id_to_root_dataset_ids[unique_logical_op_id]

    def _compute_source_unique_logical_op_ids_map(self, source_map: dict[str, list[str]], current_idx: int | None = None) -> tuple[int, str]:
        # get the topological index and logical_op_id pairs for all subplans' root operators
        subplan_topo_idx_op_id_pairs = []
        for subplan in self.subplans:
            current_idx, current_logical_op_id = subplan._compute_source_unique_logical_op_ids_map(source_map, current_idx)
            subplan_topo_idx_op_id_pairs.append((current_idx, current_logical_op_id))
            current_idx += 1

        # if current_idx is None, this is the first call, so we initialize it to 0
        if current_idx is None:
            current_idx = 0

        # compute this operator's unique logical_op_id
        this_unique_logical_op_id = f"{current_idx}-{self.logical_op_id}"

        # update the source_map for this operator
        source_map[this_unique_logical_op_id] = []
        for topo_idx, logical_op_id in subplan_topo_idx_op_id_pairs:
            unique_logical_op_id = f"{topo_idx}-{logical_op_id}"
            source_map[this_unique_logical_op_id].append(unique_logical_op_id)

        # return the current unique logical_op_id for this operator
        return current_idx, self.logical_op_id

    def get_source_unique_logical_op_ids(self, unique_logical_op_id: str) -> list[str]:
        """Return the list of unique logical_op_ids for the input(s) to this operator."""
        return self.unique_logical_op_id_to_source_logical_op_ids[unique_logical_op_id]

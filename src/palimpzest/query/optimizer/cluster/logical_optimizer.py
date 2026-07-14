from __future__ import annotations

import logging

from palimpzest.core.data.dataset import Dataset
from palimpzest.query.operators.logical import LogicalOperator
from palimpzest.utils.hash_helpers import hash_for_id

logger = logging.getLogger(__name__)


class LogicalPlan:
    def __init__(
        self,
        operators: dict[str, LogicalOperator],
        edges: dict[str, list[str]],
        root_op_id: str = None,
    ):
        """
        A logical plan is a representation of a query plan that consists of logical operators and their relationships. It is used to represent the structure of a query plan before it is executed.

        Args:

        operators: maps each logical operator id to its `LogicalOperator` object.
        Example: `{op_id: logical_operator}`.

        upstream_op_ids: maps each logical operator id to the ids of operators that produce its inputs.
        Example: for `Convert(BaseScan)`, `{convert_id: [scan_id], scan_id: []}`.

        downstream_op_ids: maps each logical operator id to the ids of operators that consume its output.
        Example: `{scan_id: [convert_id], convert_id: []}`.

        final_op_id: the id of the final/output logical operator for the query. This is the “root” from the user’s perspective: the last operator in the logical pipeline.

        """
        self.operators = operators
        self.edges = edges
        self.root_op_id = root_op_id
        self.plan_id = self.compute_plan_id()

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.

        Two different LogicalPlan instances with identical logical operators and
        subplans will have equivalent plan_ids.
        """
        logical_op_id = self.operators[self.root_op_id].get_logical_op_id()
        return hash_for_id(str((logical_op_id)))

    def __eq__(self, other) -> bool:
        """
        Two LogicalPlan instances are considered equal if they have the same root operator and the same set of operators and edges. 
        """

        if not isinstance(other, LogicalPlan):
            return False

        if self.root_op_id != other.root_op_id:
            return False

        if set(self.operators) != set(other.operators):
            return False

        for op_id, operator in self.operators.items():
            if operator != other.operators[op_id]:
                return False

        self_edges = {
            op_id: sorted(edge_op_ids) for op_id, edge_op_ids in self.edges.items()
        }
        other_edges = {
            op_id: sorted(edge_op_ids) for op_id, edge_op_ids in other.edges.items()
        }
        return self_edges == other_edges

    def __hash__(self) -> int:
        return int(self.plan_id, 16)

    def __str__(self) -> str:
        """
        Return a stable, human-readable representation of the logical plan DAG.

        The output includes the root operator id, the plan id, each logical operator
        keyed by operator id, and the graph edges as an adjacency list.
        """
        lines = [
            f"LogicalPlan(root_op_id={self.root_op_id}, plan_id={self.plan_id})",
            "Operators:",
        ]

        for op_id in sorted(self.operators):
            root_marker = " [ROOT]" if op_id == self.root_op_id else ""
            lines.append(f"  {op_id}{root_marker}: {self.operators[op_id]}")

        lines.append("Edges:")
        for op_id in sorted(self.operators):
            downstream_op_ids = self.edges.get(op_id, [])
            if len(downstream_op_ids) == 0:
                lines.append(f"  {op_id} -> []")
            else:
                lines.append(f"  {op_id} -> {downstream_op_ids}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.operators)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> LogicalPlan:
        """
        Walk a given Dataset structure and convert in a LogicalPlan DAG.
        The `Dataset` already contains the query structure:
        - Each `Dataset` has a `_operator` attribute that represents the logical operator applied to it.
        - Each `Dataset` has a `_sources` attribute that represents the input datasets for the logical operator.
        
        This method recursively traverses the `Dataset` structure, collects the logical operators and their relationships, and constructs a `LogicalPlan` object that represents the logical plan DAG.

        `operators` maps logical operator ids to operators
        `edges` maps each operator id to the downstream operators that consume its output
        """

        operators = {}
        edges = {}
        visited_dataset_ids = set()

        root_op_id = cls._visit_dataset(
            dataset,
            operators,
            edges,
            visited_dataset_ids,
        )
        return cls(operators=operators, edges=edges, root_op_id=root_op_id)

    @classmethod
    def _visit_dataset(
        cls,
        dataset: Dataset,
        operators: dict[str, LogicalOperator],
        edges: dict[str, list[str]],
        visited_dataset_ids: set[int],
    ) -> str:
        """
        Recursively visit a Dataset node while building a LogicalPlan DAG.

        Records the Dataset's logical operator in `operators`, initializes its
        downstream adjacency list in `edges`, skips already visited Dataset
        objects, and adds source-to-current edges for each input Dataset.

        Args:
            dataset: Dataset node to visit.
            operators: Mapping from logical operator id to LogicalOperator.
            edges: Mapping from logical operator id to downstream operator ids.
            visited_dataset_ids: Object ids for Dataset nodes already traversed.

        Returns:
            The logical operator id for `dataset`.
        """
        current_operator = dataset._operator
        current_op_id = current_operator.get_unique_logical_op_id()
        if current_op_id is None:
            current_op_id = current_operator.get_logical_op_id()

        operators[current_op_id] = current_operator
        edges.setdefault(current_op_id, [])

        current_dataset_id = id(dataset)
        if current_dataset_id in visited_dataset_ids:
            return current_op_id

        visited_dataset_ids.add(current_dataset_id)

        for source in dataset._sources:
            source_op_id = cls._visit_dataset(
                source,
                operators,
                edges,
                visited_dataset_ids,
            )
            edges.setdefault(source_op_id, [])
            if current_op_id not in edges[source_op_id]:
                edges[source_op_id].append(current_op_id)

        return current_op_id


class LogicalOptimizer:

    def optimize(self, logical_plan: LogicalPlan) -> list[LogicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical plans in order to cost and produce a (near) optimal logical plan.
        """
        logger.info(f"Optimizing logical plan: {logical_plan}")
        return [logical_plan]

"""Sampling budget allocation for clustered physical optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod

from palimpzest.query.optimizer.cluster.physical_operator_clustering import (
    PhysicalOperatorCluster,
)


class ClusterSamplingBudgetAllocator(ABC):
    """Assign an overall cluster optimizer sampling budget to logical operators."""

    @abstractmethod
    def allocate(
        self,
        topological_order: list[str],
        op_clusters: dict[str, PhysicalOperatorCluster],
        total_budget: int,
    ) -> dict[str, int]:
        """Return a per-logical-operator sampling budget."""
        raise NotImplementedError("Calling this method from an abstract base class.")


class EqualClusterSamplingBudgetAllocator(ClusterSamplingBudgetAllocator):
    """Give traditional operators the full budget and split semantic operator budget."""

    def allocate(
        self,
        topological_order: list[str],
        op_clusters: dict[str, PhysicalOperatorCluster],
        total_budget: int,
    ) -> dict[str, int]:
        budgets = {logical_op_id: 0 for logical_op_id in topological_order}
        if len(topological_order) == 0 or total_budget <= 0:
            return budgets

        semantic_logical_op_ids = [
            logical_op_id
            for logical_op_id in topological_order
            if any(op.is_semantic for op in op_clusters[logical_op_id].physical_ops)
        ]
        for logical_op_id in topological_order:
            if logical_op_id not in semantic_logical_op_ids:
                budgets[logical_op_id] = total_budget

        if len(semantic_logical_op_ids) == 0:
            return budgets

        per_operator_budget = total_budget // len(semantic_logical_op_ids)
        remainder = total_budget % len(semantic_logical_op_ids)
        for idx, logical_op_id in enumerate(semantic_logical_op_ids):
            budgets[logical_op_id] = per_operator_budget + (1 if idx < remainder else 0)

        return budgets

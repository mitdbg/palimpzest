"""Utilities for grouping flat physical operator candidates into search clusters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from palimpzest.constants import NAIVE_BYTES_PER_RECORD
from palimpzest.core.models import OperatorCostEstimates
from palimpzest.query.operators.aggregate import SemanticAggregate
from palimpzest.query.operators.batched import BatchedFilter
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.critique_and_refine import (
    CritiqueAndRefineConvert,
    CritiqueAndRefineFilter,
)
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.image_filter import RescaledImageFilter
from palimpzest.query.operators.join import EmbeddingJoin, JoinOp, NestedLoopsJoin
from palimpzest.query.operators.logical import LogicalOperator
from palimpzest.query.operators.mixture_of_agents import (
    MixtureOfAgentsConvert,
    MixtureOfAgentsFilter,
)
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.rag import RAGConvert, RAGFilter
from palimpzest.query.operators.scan import MarshalAndScanDataOp
from palimpzest.query.operators.split import SplitConvert, SplitFilter
from palimpzest.query.operators.topk import TopKOp

# One grouping layer in a manual cluster spec. The first tuple entry is the
# display name; the second is either a physical operator attribute name or a
# callable that extracts the value to group by.
ClusterDimension: TypeAlias = tuple[str, str | Callable[[PhysicalOperator], object]]

# Physical operator class to ordered grouping dimensions.
PhysicalOperatorClusterSpec: TypeAlias = dict[
    type[PhysicalOperator],
    list[ClusterDimension],
]


@dataclass
class PhysicalOperatorCluster:
    """Node in a hierarchy of physical operator candidates.

    Every node represents the concrete physical operators in ``physical_ops``.
    Leaf nodes also keep the per-physical-operator naive cost estimates in
    ``physical_op_cost_estimates``. Internal nodes keep an averaged
    ``cost_estimates`` value computed from their children.
    """

    name: str
    physical_ops: list[PhysicalOperator]
    children: list[PhysicalOperatorCluster] = field(default_factory=list)
    cost_estimates: OperatorCostEstimates | None = None
    physical_op_cost_estimates: dict[str, OperatorCostEstimates] = field(
        default_factory=dict
    )

    def __str__(self) -> str:
        """Render the cluster tree without listing concrete physical operators."""
        lines = []
        stack = [(self, 0)]
        while len(stack) > 0:
            cluster, depth = stack.pop()
            indent = "  " * depth
            cost_suffix = ""
            if cluster.cost_estimates is not None:
                cost_estimates = cluster.cost_estimates
                cost_suffix = (
                    " "
                    f"(card={cost_estimates.cardinality:.4g}, "
                    f"time/rec={cost_estimates.time_per_record:.4g}, "
                    f"cost/rec={cost_estimates.cost_per_record:.4g}, "
                    f"quality={cost_estimates.quality:.4g})"
                )

            if len(cluster.children) == 0:
                lines.append(f"{indent}{cluster.name}{cost_suffix}")
                lines.append(
                    f"{indent}  -> ... ({len(cluster.physical_ops)} physical operators)"
                )
            else:
                lines.append(f"{indent}{cluster.name}{cost_suffix}")
                stack.extend(
                    (child, depth + 1)
                    for child in reversed(cluster.children)
                )

        return "\n".join(lines)


MANUAL_PHYSICAL_OPERATOR_CLUSTER_SPEC: PhysicalOperatorClusterSpec = {
    # First layer is always the physical operator class. These dimensions define
    # the hand-written layers below each class.
    LLMFilter: [("model", "model")],
    BatchedFilter: [("batch_size", "batch_size"), ("model", "model")],
    RescaledImageFilter: [("rescale_factor", "rescale_factor"), ("model", "model")],
    RAGFilter: [
        ("chunk_size", "chunk_size"),
        ("num_chunks_per_field", "num_chunks_per_field"),
        ("model", "model"),
        ("embedding_model", "embedding_model"),
    ],
    SplitFilter: [
        ("num_chunks", "num_chunks"),
        ("min_size_to_chunk", "min_size_to_chunk"),
        ("model", "model"),
    ],
    MixtureOfAgentsFilter: [
        ("num_proposer_models", lambda op: len(op.proposer_models)),
        ("aggregator_model", "aggregator_model"),
        ("temperatures", lambda op: tuple(op.temperatures)),
        ("proposer_models", lambda op: tuple(op.proposer_models)),
    ],
    CritiqueAndRefineFilter: [
        ("model", "model"),
        ("critic_model", "critic_model"),
        ("refine_model", "refine_model"),
    ],
    LLMConvertBonded: [("model", "model")],
    RAGConvert: [
        ("chunk_size", "chunk_size"),
        ("num_chunks_per_field", "num_chunks_per_field"),
        ("model", "model"),
        ("embedding_model", "embedding_model"),
    ],
    SplitConvert: [
        ("num_chunks", "num_chunks"),
        ("min_size_to_chunk", "min_size_to_chunk"),
        ("model", "model"),
    ],
    MixtureOfAgentsConvert: [
        ("num_proposer_models", lambda op: len(op.proposer_models)),
        ("aggregator_model", "aggregator_model"),
        ("temperatures", lambda op: tuple(op.temperatures)),
        ("proposer_models", lambda op: tuple(op.proposer_models)),
    ],
    CritiqueAndRefineConvert: [
        ("model", "model"),
        ("critic_model", "critic_model"),
        ("refine_model", "refine_model"),
    ],
    NestedLoopsJoin: [("model", "model"), ("join_parallelism", "join_parallelism")],
    EmbeddingJoin: [
        ("num_samples", "num_samples"),
        ("embedding_model", "embedding_model"),
        ("model", "model"),
        ("join_parallelism", "join_parallelism"),
    ],
    SemanticAggregate: [("model", "model")],
    TopKOp: [("k", "k")],
}


class PhysicalOperatorClusteringStrategy:
    """Interface for alternative physical operator clustering methods."""

    def build_cluster(
        self,
        logical_op: LogicalOperator,
        physical_ops: list[PhysicalOperator],
        source_op_cost_estimates: OperatorCostEstimates | None = None,
        right_source_op_cost_estimates: OperatorCostEstimates | None = None,
    ) -> PhysicalOperatorCluster:
        """Build a cluster tree for one logical operator's physical candidates."""
        raise NotImplementedError


class ManualPhysicalOperatorClusteringStrategy(PhysicalOperatorClusteringStrategy):
    """Build a cluster tree from the hand-written physical operator spec.

    The registry owns candidate instantiation; this strategy only organizes an
    already-instantiated flat list. It computes naive costs for each concrete
    operator, stores those estimates at leaf nodes, and stores averaged estimates
    at every internal cluster.
    """

    def __init__(
        self,
        cluster_spec: PhysicalOperatorClusterSpec | None = None,
        input_record_size_in_bytes: int | float = NAIVE_BYTES_PER_RECORD,
    ):
        """Create a manual clustering strategy.

        ``cluster_spec`` controls the ordered dimensions below each physical
        operator class. ``input_record_size_in_bytes`` is only used by scan
        operators whose naive estimates require an input record size.
        """
        self.cluster_spec = (
            MANUAL_PHYSICAL_OPERATOR_CLUSTER_SPEC
            if cluster_spec is None
            else cluster_spec
        )
        self.input_record_size_in_bytes = input_record_size_in_bytes

    def build_cluster(
        self,
        logical_op: LogicalOperator,
        physical_ops: list[PhysicalOperator],
        source_op_cost_estimates: OperatorCostEstimates | None = None,
        right_source_op_cost_estimates: OperatorCostEstimates | None = None,
    ) -> PhysicalOperatorCluster:
        """Cluster concrete physical operators for a single logical operator.

        The root node is named after the logical operator. Its first layer is
        the physical operator class, followed by dimensions from the manual
        spec. Leaf nodes retain per-physical-operator naive estimates, and every
        internal node receives averaged child estimates.
        """
        if source_op_cost_estimates is None:
            source_op_cost_estimates = OperatorCostEstimates(
                cardinality=100,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )
        if right_source_op_cost_estimates is None:
            right_source_op_cost_estimates = source_op_cost_estimates

        op_to_cost_estimates = {}
        for op in physical_ops:
            if isinstance(op, MarshalAndScanDataOp):
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    source_op_cost_estimates,
                    input_record_size_in_bytes=self.input_record_size_in_bytes,
                )
            elif isinstance(op, JoinOp):
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    source_op_cost_estimates,
                    right_source_op_cost_estimates,
                )
            else:
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    source_op_cost_estimates
                )

        class_to_ops = defaultdict(list)
        for op in physical_ops:
            class_to_ops[type(op)].append(op)

        children = []
        for op_class, ops in sorted(
            class_to_ops.items(), key=lambda item: item[0].__name__
        ):
            dimensions = self.cluster_spec.get(op_class, [])
            dimension_children = self._build_dimension_clusters(
                ops,
                dimensions,
                op_to_cost_estimates,
            )
            if len(dimension_children) == 0:
                cluster_cost_estimates = self._average_cost_estimates(
                    [op_to_cost_estimates[op.get_full_op_id()] for op in ops]
                )
                physical_op_cost_estimates = {
                    op.get_full_op_id(): op_to_cost_estimates[op.get_full_op_id()]
                    for op in ops
                }
            else:
                cluster_cost_estimates = self._average_cost_estimates(
                    [
                        child.cost_estimates
                        for child in dimension_children
                        if child.cost_estimates is not None
                    ]
                )
                physical_op_cost_estimates = {}
            children.append(
                PhysicalOperatorCluster(
                    name=op_class.__name__,
                    physical_ops=ops,
                    children=dimension_children,
                    cost_estimates=cluster_cost_estimates,
                    physical_op_cost_estimates=physical_op_cost_estimates,
                )
            )

        return PhysicalOperatorCluster(
            name=logical_op.logical_op_name(),
            physical_ops=physical_ops,
            children=children,
            cost_estimates=self._average_cost_estimates(
                [
                    child.cost_estimates
                    for child in children
                    if child.cost_estimates is not None
                ]
            ),
        )

    def _build_dimension_clusters(
        self,
        physical_ops: list[PhysicalOperator],
        dimensions: list[ClusterDimension],
        op_to_cost_estimates: dict[str, OperatorCostEstimates],
    ) -> list[PhysicalOperatorCluster]:
        """Recursively group operators by the remaining manual dimensions."""
        if len(dimensions) == 0:
            return []

        dimension_name, accessor = dimensions[0]
        value_to_ops = defaultdict(list)
        for op in physical_ops:
            value = getattr(op, accessor) if isinstance(accessor, str) else accessor(op)
            if isinstance(value, list):
                value = tuple(value)
            value_to_ops[value].append(op)

        children = []
        for value, ops in sorted(value_to_ops.items(), key=lambda item: str(item[0])):
            next_children = self._build_dimension_clusters(
                ops,
                dimensions[1:],
                op_to_cost_estimates,
            )
            if len(next_children) == 0:
                cluster_cost_estimates = self._average_cost_estimates(
                    [op_to_cost_estimates[op.get_full_op_id()] for op in ops]
                )
                physical_op_cost_estimates = {
                    op.get_full_op_id(): op_to_cost_estimates[op.get_full_op_id()]
                    for op in ops
                }
            else:
                cluster_cost_estimates = self._average_cost_estimates(
                    [
                        child.cost_estimates
                        for child in next_children
                        if child.cost_estimates is not None
                    ]
                )
                physical_op_cost_estimates = {}
            children.append(
                PhysicalOperatorCluster(
                    name=f"{dimension_name}={value}",
                    physical_ops=ops,
                    children=next_children,
                    cost_estimates=cluster_cost_estimates,
                    physical_op_cost_estimates=physical_op_cost_estimates,
                )
            )

        return children

    def _average_cost_estimates(
        self,
        cost_estimates: list[OperatorCostEstimates],
    ) -> OperatorCostEstimates | None:
        """Average every populated ``OperatorCostEstimates`` field independently."""
        if len(cost_estimates) == 0:
            return None

        averaged_fields = {}
        for field_name in OperatorCostEstimates.model_fields:
            values = [
                getattr(cost_estimate, field_name)
                for cost_estimate in cost_estimates
                if getattr(cost_estimate, field_name) is not None
            ]
            averaged_fields[field_name] = (
                sum(values) / len(values)
                if len(values) > 0
                else None
            )

        return OperatorCostEstimates(**averaged_fields)

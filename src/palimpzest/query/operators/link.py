from __future__ import annotations

from typing import Any

from palimpzest.core.data.graph_store import GraphEdge
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.utils.hash_helpers import hash_for_id

class LinkToChildrenOp(PhysicalOperator):
    """
    Physical operator that creates edges from the input record to its parent records.
    It returns the input record unchanged (passthrough).
    """
    def __init__(
        self,
        graph: Any,
        edge_type: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.edge_type = edge_type

    def __str__(self) -> str:
        return f"LinkToChildren(edge_type={self.edge_type})"

    def get_id_params(self) -> dict:
        return {
            "graph_id": self.graph.graph_id,
            "edge_type": self.edge_type,
        }

    def get_op_params(self) -> dict:
        return {
            "graph": self.graph,
            "edge_type": self.edge_type,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        parent_ids = candidate._parent_ids or []
        src_id = candidate._id
        
        for dst_id in parent_ids:
            # Create edge id
            edge_id = hash_for_id(f"link:{self.edge_type}:{src_id}->{dst_id}")
            
            # Create edge
            edge = GraphEdge(
                id=edge_id,
                src=src_id,
                dst=dst_id,
                type=self.edge_type,
                attrs={}
            )
            
            # Add to graph
            # Note: This assumes self.graph is a GraphDataset or similar with add_edge
            self.graph.add_edge(edge)

        # Passthrough stats
        stat = RecordOpStats(
            record_id=candidate._id,
            record_parent_ids=candidate._parent_ids,
            record_source_indices=candidate._source_indices,
            record_state=candidate.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
            op_details=self.get_id_params(),
        )
        
        return DataRecordSet([candidate], [stat], input=candidate)

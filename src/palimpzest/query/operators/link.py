from __future__ import annotations

from typing import Any

from palimpzest.core.data.graph_store import GraphEdge, GraphNode
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


class UpsertGraphNodesOp(PhysicalOperator):
    """Materialize each input record as a node in a GraphDataset.

    Side-effecting operator; returns the input record unchanged.
    """

    def __init__(
        self,
        graph: Any,
        *,
        text_field: str = "text",
        node_type: str | None = "chunk",
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.text_field = text_field
        self.node_type = node_type
        self.overwrite = overwrite

    def __str__(self) -> str:
        return f"UpsertGraphNodes(text_field={self.text_field}, node_type={self.node_type})"

    def get_id_params(self) -> dict:
        return {
            "graph_id": self.graph.graph_id,
            "text_field": self.text_field,
            "node_type": self.node_type,
            "overwrite": self.overwrite,
        }

    def get_op_params(self) -> dict:
        return {
            "graph": self.graph,
            "text_field": self.text_field,
            "node_type": self.node_type,
            "overwrite": self.overwrite,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        node_id = getattr(candidate, "id", None) or candidate._id

        existing = None
        if self.graph.has_node(str(node_id)):
            existing = self.graph.get_node(str(node_id))

        existing_is_placeholder = (
            existing is not None
            and isinstance(existing.attrs, dict)
            and existing.attrs.get("__pz_placeholder__") is True
        )

        if not self.overwrite and existing is not None and not existing_is_placeholder:
            # Idempotent no-op.
            node = None
        else:
            text_val = getattr(candidate, self.text_field, None)
            text = text_val if isinstance(text_val, str) else None

            attrs = getattr(candidate, "attrs", None)
            base_attrs: dict[str, Any] = dict(attrs) if isinstance(attrs, dict) else {}

            chunk_meta: dict[str, Any] = {}
            for k in ("source_node_id", "chunk_index", "prev_chunk_id"):
                v = getattr(candidate, k, None)
                if v is not None:
                    chunk_meta[k] = v
            if chunk_meta:
                base_attrs = dict(base_attrs)
                base_attrs.setdefault("chunk", {}).update(chunk_meta)

            node = GraphNode(
                id=str(node_id),
                label=getattr(candidate, "label", None),
                type=self.node_type if self.node_type is not None else getattr(candidate, "type", None),
                source=getattr(candidate, "source", None),
                level=getattr(candidate, "level", None),
                created_at=getattr(candidate, "created_at", None),
                updated_at=getattr(candidate, "updated_at", None),
                attrs=base_attrs,
                text=text,
                embedding=getattr(candidate, "embedding", None),
            )

            # GraphDataset.add_node bumps revision; prefer store overwrite control.
            if self.graph.has_node(str(node_id)):
                # Overwrite placeholders even when overwrite=False.
                if self.overwrite or existing_is_placeholder:
                    self.graph.upsert_node(node)
            else:
                self.graph.add_node(node, overwrite=self.overwrite)

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
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return DataRecordSet([candidate], [stat], input=candidate)


class LinkFromParentsOp(PhysicalOperator):
    """Create edges from each parent id to the current record id."""

    def __init__(
        self,
        graph: Any,
        edge_type: str,
        *,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.edge_type = edge_type
        self.overwrite = overwrite

    def __str__(self) -> str:
        return f"LinkFromParents(edge_type={self.edge_type})"

    def get_id_params(self) -> dict:
        return {
            "graph_id": self.graph.graph_id,
            "edge_type": self.edge_type,
            "overwrite": self.overwrite,
        }

    def get_op_params(self) -> dict:
        return {"graph": self.graph, "edge_type": self.edge_type, "overwrite": self.overwrite}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        parent_ids = candidate._parent_ids or []
        dst_id = getattr(candidate, "id", None) or candidate._id

        for src_id in parent_ids:
            if src_id is None or src_id == dst_id:
                continue
            if not self.graph.has_node(str(src_id)) or not self.graph.has_node(str(dst_id)):
                continue
            edge_id = hash_for_id(f"link:{self.edge_type}:{src_id}->{dst_id}")
            edge = GraphEdge(id=edge_id, src=str(src_id), dst=str(dst_id), type=self.edge_type, attrs={})
            if self.graph.has_edge(edge_id) and not self.overwrite:
                continue
            self.graph.add_edge(edge, overwrite=self.overwrite)

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
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return DataRecordSet([candidate], [stat], input=candidate)


class LinkFromFieldOp(PhysicalOperator):
    """Create an edge where src id comes from a record field."""

    def __init__(
        self,
        graph: Any,
        edge_type: str,
        *,
        src_field: str,
        dst_field: str | None = None,
        ensure_src_node: bool = False,
        ensure_dst_node: bool = False,
        placeholder_node_type: str | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.edge_type = edge_type
        self.src_field = src_field
        self.dst_field = dst_field
        self.ensure_src_node = ensure_src_node
        self.ensure_dst_node = ensure_dst_node
        self.placeholder_node_type = placeholder_node_type
        self.overwrite = overwrite

    def __str__(self) -> str:
        return f"LinkFromField(edge_type={self.edge_type}, src_field={self.src_field}, dst_field={self.dst_field})"

    def get_id_params(self) -> dict:
        return {
            "graph_id": self.graph.graph_id,
            "edge_type": self.edge_type,
            "src_field": self.src_field,
            "dst_field": self.dst_field,
            "ensure_src_node": self.ensure_src_node,
            "ensure_dst_node": self.ensure_dst_node,
            "placeholder_node_type": self.placeholder_node_type,
            "overwrite": self.overwrite,
        }

    def get_op_params(self) -> dict:
        return {
            "graph": self.graph,
            "edge_type": self.edge_type,
            "src_field": self.src_field,
            "dst_field": self.dst_field,
            "ensure_src_node": self.ensure_src_node,
            "ensure_dst_node": self.ensure_dst_node,
            "placeholder_node_type": self.placeholder_node_type,
            "overwrite": self.overwrite,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        src_id = getattr(candidate, self.src_field, None)
        if src_id is None:
            # no-op
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
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            return DataRecordSet([candidate], [stat], input=candidate)

        dst_id = (
            getattr(candidate, self.dst_field, None)
            if self.dst_field
            else (getattr(candidate, "id", None) or candidate._id)
        )
        if dst_id is None or src_id == dst_id:
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
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            return DataRecordSet([candidate], [stat], input=candidate)

        if self.ensure_src_node and not self.graph.has_node(str(src_id)):
            self.graph.add_node(
                GraphNode(
                    id=str(src_id),
                    type=self.placeholder_node_type,
                    attrs={"__pz_placeholder__": True},
                ),
                overwrite=False,
            )

        if self.ensure_dst_node and not self.graph.has_node(str(dst_id)):
            self.graph.add_node(
                GraphNode(
                    id=str(dst_id),
                    type=self.placeholder_node_type,
                    attrs={"__pz_placeholder__": True},
                ),
                overwrite=False,
            )

        if self.graph.has_node(str(src_id)) and self.graph.has_node(str(dst_id)):
            edge_id = hash_for_id(f"link:{self.edge_type}:{src_id}->{dst_id}")
            if not self.graph.has_edge(edge_id) or self.overwrite:
                self.graph.add_edge(
                    GraphEdge(id=edge_id, src=str(src_id), dst=str(dst_id), type=self.edge_type, attrs={}),
                    overwrite=self.overwrite,
                )

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
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return DataRecordSet([candidate], [stat], input=candidate)

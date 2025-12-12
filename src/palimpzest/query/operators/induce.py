from __future__ import annotations

from typing import Callable, Protocol

from pydantic import BaseModel, Field

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.utils.hash_helpers import hash_for_id


class InducedEdgeResult(BaseModel):
    src_node_id: str
    dst_node_id: str
    edge_type: str
    edge_id: str | None = None
    created: bool = Field(description="True if an edge was added/overwritten")
    existed: bool = Field(description="True if the target edge already existed")
    score: float | None = Field(default=None, description="Optional predicate score")


def _default_edge_id(*, src: str, dst: str, edge_type: str) -> str:
    # Deterministic, stable id across runs.
    return hash_for_id(f"induce:{edge_type}:{src}->{dst}")


class PairDecider(Protocol):
    def __call__(self, src_id: str, src: GraphNode, dst_id: str, dst: GraphNode, graph: GraphDataset) -> float | bool: ...


class InduceEdgesOp(PhysicalOperator):
    """Physical operator that induces/mutates edges in a `GraphDataset`.

    Contract (MVP): input records provide `src_field` and `dst_field` values.
    For each (src, dst) pair, evaluate `predicate` (bool or float score) and
    add an edge when it passes.
    """

    def __init__(
        self,
        graph: GraphDataset,
        edge_type: str,
        src_field: str = "src_node_id",
        dst_field: str = "dst_node_id",
        include_overlay: bool = True,
        predicate: Callable[[str, GraphNode, str, GraphNode, GraphDataset], bool | float] | None = None,
        predicate_id: str | None = None,
        decider: PairDecider | None = None,
        decider_id: str | None = None,
        threshold: float = 0.5,
        overwrite: bool = False,
        edge_id_fn: Callable[[str, str, str], str] | None = None,
        edge_attrs_fn: Callable[[str, str, str, float | None], dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.src_field = src_field
        self.dst_field = dst_field
        self.edge_type = edge_type
        self.include_overlay = include_overlay
        self.predicate = predicate
        self.predicate_id = predicate_id
        self.decider = decider
        self.decider_id = decider_id
        self.threshold = threshold
        self.overwrite = overwrite
        self.edge_id_fn = edge_id_fn
        self.edge_attrs_fn = edge_attrs_fn

    def __str__(self) -> str:
        op = super().__str__()
        op += f"    Graph: {self.graph.graph_id}@{self.graph.revision}\n"
        op += f"    Src Field: {self.src_field}\n"
        op += f"    Dst Field: {self.dst_field}\n"
        op += f"    Edge Type: {self.edge_type}\n"
        op += f"    Threshold: {self.threshold}\n"
        op += f"    Overwrite: {self.overwrite}\n"
        return op

    def get_id_params(self) -> dict:
        id_params = super().get_id_params()
        return {
            "graph_id": self.graph.graph_id,
            "graph_revision": self.graph.revision,
            "src_field": self.src_field,
            "dst_field": self.dst_field,
            "edge_type": self.edge_type,
            "include_overlay": self.include_overlay,
            "predicate_id": self.predicate_id,
            "decider_id": self.decider_id,
            "threshold": self.threshold,
            "overwrite": self.overwrite,
            **id_params,
        }

    def get_op_params(self) -> dict:
        op_params = super().get_op_params()
        return {
            "graph": self.graph,
            "src_field": self.src_field,
            "dst_field": self.dst_field,
            "edge_type": self.edge_type,
            "include_overlay": self.include_overlay,
            "predicate": self.predicate,
            "predicate_id": self.predicate_id,
            "decider": self.decider,
            "decider_id": self.decider_id,
            "threshold": self.threshold,
            "overwrite": self.overwrite,
            "edge_id_fn": self.edge_id_fn,
            "edge_attrs_fn": self.edge_attrs_fn,
            **op_params,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # Conservative default: <=1 induced edge per input record.
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def _passes_edge_type_filter(self) -> bool:
        if self.edge_type.startswith("overlay:"):
            return self.include_overlay
        return True

    def _score_or_bool(
        self,
        *,
        src_node_id: str,
        src_node: GraphNode,
        dst_node_id: str,
        dst_node: GraphNode,
    ) -> tuple[bool, float | None]:
        if self.predicate is None and self.decider is None:
            return True, None

        if self.predicate is None:
            out = self.decider(src_node_id, src_node, dst_node_id, dst_node, self.graph)
        else:
            out = self.predicate(src_node_id, src_node, dst_node_id, dst_node, self.graph)
        if isinstance(out, bool):
            return out, None
        score = float(out)
        return score >= self.threshold, score

    def _edge_id(self, *, src: str, dst: str) -> str:
        if self.edge_id_fn is None:
            return _default_edge_id(src=src, dst=dst, edge_type=self.edge_type)
        return self.edge_id_fn(src, dst, self.edge_type)

    def _edge_attrs(self, *, src: str, dst: str, score: float | None) -> dict:
        if self.edge_attrs_fn is None:
            attrs: dict = {}
            if score is not None:
                attrs["predicate_score"] = score
            return attrs
        return self.edge_attrs_fn(src, dst, self.edge_type, score)

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        if not self._passes_edge_type_filter():
            # Edge type is overlay:* but overlays excluded: no-op, emit record(s) marked not created.
            return DataRecordSet([], [])

        src_node_id = getattr(candidate, self.src_field, None)
        dst_node_id = getattr(candidate, self.dst_field, None)
        if src_node_id is None or dst_node_id is None:
            raise ValueError(f"InduceEdgesOp missing required fields: {self.src_field}, {self.dst_field}")

        if isinstance(src_node_id, list) or isinstance(dst_node_id, list):
            raise ValueError("InduceEdgesOp MVP expects scalar src/dst node ids")

        if not self.graph.has_node(src_node_id) or not self.graph.has_node(dst_node_id):
            # Missing nodes: emit a record indicating no creation.
            data_item = {
                "src_node_id": str(src_node_id),
                "dst_node_id": str(dst_node_id),
                "edge_type": self.edge_type,
                "edge_id": None,
                "created": False,
                "existed": False,
                "score": None,
            }
            dr = DataRecord.from_parent(
                schema=self.output_schema,
                data_item=data_item,
                parent_record=candidate,
                project_cols=[],
            )
            stat = RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=0.0,
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            return DataRecordSet([dr], [stat], input=candidate)

        src_node = self.graph.get_node(src_node_id)
        dst_node = self.graph.get_node(dst_node_id)

        should_add, score = self._score_or_bool(
            src_node_id=src_node_id,
            src_node=src_node,
            dst_node_id=dst_node_id,
            dst_node=dst_node,
        )

        edge_id = self._edge_id(src=src_node_id, dst=dst_node_id)
        existed = self.graph.has_edge(edge_id)
        created = False

        if should_add:
            if existed and not self.overwrite:
                created = False
            else:
                edge = GraphEdge(
                    id=edge_id,
                    src=src_node_id,
                    dst=dst_node_id,
                    type=self.edge_type,
                    attrs=self._edge_attrs(src=src_node_id, dst=dst_node_id, score=score),
                )
                self.graph.add_edge(edge, overwrite=self.overwrite)
                created = True

        data_item = {
            "src_node_id": src_node_id,
            "dst_node_id": dst_node_id,
            "edge_type": self.edge_type,
            "edge_id": edge_id,
            "created": created,
            "existed": existed,
            "score": score,
        }
        dr = DataRecord.from_parent(
            schema=self.output_schema,
            data_item=data_item,
            parent_record=candidate,
            project_cols=[],
        )
        stat = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return DataRecordSet([dr], [stat], input=candidate)

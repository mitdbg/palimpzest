from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str | None = None
    type: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    embedding: list[float] | None = None


class GraphEdge(BaseModel):
    id: str
    src: str
    dst: str
    type: str
    directed: bool = True
    weight: float | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class GraphSnapshot(BaseModel):
    version: int = 1
    graph_id: str
    revision: int
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class GraphDataset:
    """In-memory mutable graph with optional snapshot save/load."""

    def __init__(self, *, graph_id: str | None = None, name: str | None = None) -> None:
        self.name = name
        self.graph_id = graph_id or str(uuid.uuid4())
        self.revision: int = 0

        self._nodes_by_id: dict[str, GraphNode] = {}
        self._edges_by_id: dict[str, GraphEdge] = {}

        self._out_edges_by_src: dict[str, set[str]] = {}
        self._in_edges_by_dst: dict[str, set[str]] = {}
        self._edge_ids_by_type: dict[str, set[str]] = {}

    # -----------------
    # Mutation helpers
    # -----------------
    def _bump_revision(self) -> None:
        self.revision += 1

    def add_node(self, node: GraphNode, *, overwrite: bool = False) -> None:
        if node.id in self._nodes_by_id and not overwrite:
            raise ValueError(f"Node id already exists: {node.id}")

        self._nodes_by_id[node.id] = node
        self._out_edges_by_src.setdefault(node.id, set())
        self._in_edges_by_dst.setdefault(node.id, set())
        self._bump_revision()

    def upsert_node(self, node: GraphNode) -> None:
        self.add_node(node, overwrite=True)

    def remove_node(self, node_id: str, *, cascade: bool = True) -> None:
        if node_id not in self._nodes_by_id:
            raise ValueError(f"Unknown node id: {node_id}")

        incident_edge_ids = set(self._out_edges_by_src.get(node_id, set())) | set(self._in_edges_by_dst.get(node_id, set()))
        if incident_edge_ids and not cascade:
            raise ValueError(f"Cannot remove node {node_id}: {len(incident_edge_ids)} incident edges")

        for edge_id in list(incident_edge_ids):
            self._remove_edge(edge_id, bump_revision=False)

        self._nodes_by_id.pop(node_id, None)
        self._out_edges_by_src.pop(node_id, None)
        self._in_edges_by_dst.pop(node_id, None)
        self._bump_revision()

    def add_edge(self, edge: GraphEdge, *, overwrite: bool = False) -> None:
        if edge.id in self._edges_by_id and not overwrite:
            raise ValueError(f"Edge id already exists: {edge.id}")
        if edge.src not in self._nodes_by_id:
            raise ValueError(f"Edge src node does not exist: {edge.src}")
        if edge.dst not in self._nodes_by_id:
            raise ValueError(f"Edge dst node does not exist: {edge.dst}")

        # If overwriting, first remove existing edge from indexes
        if edge.id in self._edges_by_id:
            self._remove_edge_from_indexes(self._edges_by_id[edge.id])

        self._edges_by_id[edge.id] = edge
        self._add_edge_to_indexes(edge)
        self._bump_revision()

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.add_edge(edge, overwrite=True)

    def remove_edge(self, edge_id: str) -> None:
        self._remove_edge(edge_id, bump_revision=True)

    def _remove_edge(self, edge_id: str, *, bump_revision: bool) -> None:
        if edge_id not in self._edges_by_id:
            raise ValueError(f"Unknown edge id: {edge_id}")

        edge = self._edges_by_id.pop(edge_id)
        self._remove_edge_from_indexes(edge)
        if bump_revision:
            self._bump_revision()

    def _add_edge_to_indexes(self, edge: GraphEdge) -> None:
        self._out_edges_by_src.setdefault(edge.src, set()).add(edge.id)
        self._in_edges_by_dst.setdefault(edge.dst, set()).add(edge.id)
        self._edge_ids_by_type.setdefault(edge.type, set()).add(edge.id)

        if not edge.directed:
            self._out_edges_by_src.setdefault(edge.dst, set()).add(edge.id)
            self._in_edges_by_dst.setdefault(edge.src, set()).add(edge.id)

    def _remove_edge_from_indexes(self, edge: GraphEdge) -> None:
        self._out_edges_by_src.get(edge.src, set()).discard(edge.id)
        self._in_edges_by_dst.get(edge.dst, set()).discard(edge.id)
        self._edge_ids_by_type.get(edge.type, set()).discard(edge.id)

        if not edge.directed:
            self._out_edges_by_src.get(edge.dst, set()).discard(edge.id)
            self._in_edges_by_dst.get(edge.src, set()).discard(edge.id)

    # -----------------
    # Queries
    # -----------------
    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes_by_id

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges_by_id

    def get_node(self, node_id: str) -> GraphNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as e:
            raise ValueError(f"Unknown node id: {node_id}") from e

    def get_edge(self, edge_id: str) -> GraphEdge:
        try:
            return self._edges_by_id[edge_id]
        except KeyError as e:
            raise ValueError(f"Unknown edge id: {edge_id}") from e

    def iter_out_edges(self, node_id: str, *, edge_type: str | None = None) -> Iterable[GraphEdge]:
        if node_id not in self._nodes_by_id:
            raise ValueError(f"Unknown node id: {node_id}")

        edge_ids = self._out_edges_by_src.get(node_id, set())
        if edge_type is None:
            for edge_id in edge_ids:
                yield self._edges_by_id[edge_id]
            return

        for edge_id in edge_ids:
            edge = self._edges_by_id[edge_id]
            if edge.type == edge_type:
                yield edge

    def iter_in_edges(self, node_id: str, *, edge_type: str | None = None) -> Iterable[GraphEdge]:
        if node_id not in self._nodes_by_id:
            raise ValueError(f"Unknown node id: {node_id}")

        edge_ids = self._in_edges_by_dst.get(node_id, set())
        if edge_type is None:
            for edge_id in edge_ids:
                yield self._edges_by_id[edge_id]
            return

        for edge_id in edge_ids:
            edge = self._edges_by_id[edge_id]
            if edge.type == edge_type:
                yield edge

    def iter_neighbors(self, node_id: str, *, edge_type: str | None = None) -> Iterable[tuple[GraphEdge, GraphNode]]:
        """Iterate adjacent neighbors.

        Includes both outgoing and incoming edges. For undirected edges, results are deduplicated.
        """

        seen: set[str] = set()
        for edge in self.iter_out_edges(node_id, edge_type=edge_type):
            if edge.id in seen:
                continue
            seen.add(edge.id)
            neighbor_id = edge.dst if edge.src == node_id else edge.src
            yield edge, self._nodes_by_id[neighbor_id]

        for edge in self.iter_in_edges(node_id, edge_type=edge_type):
            if edge.id in seen:
                continue
            seen.add(edge.id)
            neighbor_id = edge.dst if edge.src == node_id else edge.src
            yield edge, self._nodes_by_id[neighbor_id]

    # -----------------
    # Persistence
    # -----------------
    def to_snapshot(self, *, include_overlay: bool = True) -> GraphSnapshot:
        edges = list(self._edges_by_id.values())
        if not include_overlay:
            edges = [e for e in edges if not e.type.startswith("overlay:")]

        return GraphSnapshot(
            graph_id=self.graph_id,
            revision=self.revision,
            nodes=list(self._nodes_by_id.values()),
            edges=edges,
        )

    @classmethod
    def from_snapshot(cls, snapshot: GraphSnapshot, *, name: str | None = None) -> GraphDataset:
        graph = cls(graph_id=snapshot.graph_id, name=name)
        graph.revision = snapshot.revision

        graph._nodes_by_id = {n.id: n for n in snapshot.nodes}
        graph._edges_by_id = {e.id: e for e in snapshot.edges}

        graph._out_edges_by_src = {node_id: set() for node_id in graph._nodes_by_id}
        graph._in_edges_by_dst = {node_id: set() for node_id in graph._nodes_by_id}
        graph._edge_ids_by_type = {}

        # validate invariants + rebuild indexes
        for edge in graph._edges_by_id.values():
            if edge.src not in graph._nodes_by_id:
                raise ValueError(f"Snapshot edge src missing node: {edge.src}")
            if edge.dst not in graph._nodes_by_id:
                raise ValueError(f"Snapshot edge dst missing node: {edge.dst}")
            graph._add_edge_to_indexes(edge)

        return graph

    def save(self, path: str | Path, *, include_overlay: bool = True) -> None:
        path = Path(path)
        snapshot = self.to_snapshot(include_overlay=include_overlay)

        # Ensure JSON serializable
        payload = snapshot.model_dump(mode="json")
        json.dumps(payload)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> GraphDataset:
        path = Path(path)
        payload = json.loads(path.read_text())
        snapshot = GraphSnapshot.model_validate(payload)
        return cls.from_snapshot(snapshot, name=name)

    def traverse(
        self,
        *,
        start_node_ids: list[str],
        edge_type: str | None = None,
        include_overlay: bool = True,
        beam_width: int = 32,
        max_steps: int = 128,
        allow_revisit: bool = False,
        ranker: callable | None = None,
        ranker_id: str | None = None,
        visit_filter: callable | None = None,
        visit_filter_id: str | None = None,
        admittance: callable | None = None,
        admittance_id: str | None = None,
        termination: callable | None = None,
        termination_id: str | None = None,
        node_program: callable | None = None,
        node_program_id: str | None = None,
        node_program_output_schema: type[BaseModel] | None = None,
        node_program_config: object | None = None,
        tracer: callable | None = None,
        tracer_id: str | None = None,
    ):
        """Create a PZ `Dataset` that traverses this graph using beam search.

        This is a convenience wrapper that builds a one-record `MemoryDataset` seed and applies
        the `Traverse` logical operator.
        """

        from palimpzest.core.data.dataset import Dataset
        from palimpzest.core.data.iter_dataset import MemoryDataset
        from palimpzest.core.lib.schemas import create_schema_from_fields, union_schemas
        from palimpzest.query.operators.logical import Traverse
        from palimpzest.query.operators.traverse import GraphTraversalResult

        class _TraverseSeed(BaseModel):
            start_node_ids: list[str] = Field(default_factory=list)

        seed = MemoryDataset(
            id=f"graph-traverse-seed-{self.graph_id}",
            vals=[{"start_node_ids": start_node_ids}],
            schema=_TraverseSeed,
        )

        output_schema: type[BaseModel] = GraphTraversalResult
        if node_program is not None:
            if node_program_output_schema is None:
                raise ValueError("node_program_output_schema is required when node_program is provided")

            # Make program output fields optional (default=None) so traversal-only
            # records remain valid even if the subprogram yields 0 records.
            optional_fields: list[dict] = []
            for field_name, field in node_program_output_schema.model_fields.items():
                optional_fields.append(
                    {
                        "name": field_name,
                        "type": field.annotation | None,
                        "description": field.description or f"{field_name} (from node_program)",
                        "default": None,
                    }
                )
            optional_program_schema = create_schema_from_fields(optional_fields)
            output_schema = union_schemas([GraphTraversalResult, optional_program_schema])

        operator = Traverse(
            graph=self,
            input_schema=seed.schema,
            output_schema=output_schema,
            start_field="start_node_ids",
            edge_type=edge_type,
            include_overlay=include_overlay,
            beam_width=beam_width,
            max_steps=max_steps,
            allow_revisit=allow_revisit,
            ranker=ranker,
            ranker_id=ranker_id,
            visit_filter=visit_filter,
            visit_filter_id=visit_filter_id,
            admittance=admittance,
            admittance_id=admittance_id,
            termination=termination,
            termination_id=termination_id,
            node_program=node_program,
            node_program_id=node_program_id,
            node_program_config=node_program_config,
            tracer=tracer,
            tracer_id=tracer_id,
        )

        return Dataset(sources=[seed], operator=operator, schema=output_schema)

    def induce_edges(
        self,
        *,
        candidate_pairs: list[tuple[str, str]],
        edge_type: str,
        include_overlay: bool = True,
        predicate: callable | None = None,
        predicate_id: str | None = None,
        threshold: float = 0.5,
        overwrite: bool = False,
        src_field: str = "src_node_id",
        dst_field: str = "dst_node_id",
        edge_id_fn: callable | None = None,
        edge_attrs_fn: callable | None = None,
    ):
        """Create a PZ `Dataset` that evaluates candidate pairs and adds edges.

        This is a convenience wrapper that seeds a `MemoryDataset` of candidate (src,dst)
        pairs and applies the `InduceEdges` logical operator.
        """

        from palimpzest.core.data.dataset import Dataset
        from palimpzest.core.data.iter_dataset import MemoryDataset
        from palimpzest.query.operators.induce import InducedEdgeResult
        from palimpzest.query.operators.logical import InduceEdges

        class _InduceSeed(BaseModel):
            src_node_id: str
            dst_node_id: str

        vals = [{"src_node_id": s, "dst_node_id": d} for (s, d) in candidate_pairs]
        seed = MemoryDataset(
            id=f"graph-induce-seed-{self.graph_id}",
            vals=vals,
            schema=_InduceSeed,
        )

        op = InduceEdges(
            graph=self,
            input_schema=_InduceSeed,
            output_schema=InducedEdgeResult,
            src_field=src_field,
            dst_field=dst_field,
            edge_type=edge_type,
            include_overlay=include_overlay,
            predicate=predicate,
            predicate_id=predicate_id,
            threshold=threshold,
            overwrite=overwrite,
            edge_id_fn=edge_id_fn,
            edge_attrs_fn=edge_attrs_fn,
        )

        return Dataset(sources=[seed], operator=op, schema=InducedEdgeResult)

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str | None = None
    type: str | None = None
    source: str | None = None
    level: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
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


class GraphStore(ABC):
    """Abstract base class for graph storage backends."""

    @abstractmethod
    def add_node(self, node: GraphNode, *, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def add_nodes(self, nodes: Iterable[GraphNode], *, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def remove_node(self, node_id: str, *, cascade: bool = True) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode:
        pass

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        pass

    @abstractmethod
    def iter_nodes(self) -> Iterable[GraphNode]:
        pass

    @abstractmethod
    def count_nodes(self) -> int:
        pass

    @abstractmethod
    def get_node_ids(self) -> list[str]:
        pass

    @abstractmethod
    def add_edge(self, edge: GraphEdge, *, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def add_edges(self, edges: Iterable[GraphEdge], *, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def remove_edge(self, edge_id: str) -> None:
        pass

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge:
        pass

    @abstractmethod
    def has_edge(self, edge_id: str) -> bool:
        pass

    @abstractmethod
    def iter_edges(self) -> Iterable[GraphEdge]:
        pass

    @abstractmethod
    def iter_out_edges(self, node_id: str, *, edge_type: str | None = None) -> Iterable[GraphEdge]:
        pass

    @abstractmethod
    def iter_in_edges(self, node_id: str, *, edge_type: str | None = None) -> Iterable[GraphEdge]:
        pass


class MemoryGraphStore(GraphStore):
    """In-memory implementation of GraphStore."""

    def __init__(self) -> None:
        self._nodes_by_id: dict[str, GraphNode] = {}
        self._edges_by_id: dict[str, GraphEdge] = {}

        self._out_edges_by_src: dict[str, set[str]] = {}
        self._in_edges_by_dst: dict[str, set[str]] = {}
        self._edge_ids_by_type: dict[str, set[str]] = {}

    def add_node(self, node: GraphNode, *, overwrite: bool = False) -> None:
        if node.id in self._nodes_by_id and not overwrite:
            raise ValueError(f"Node id already exists: {node.id}")

        self._nodes_by_id[node.id] = node
        self._out_edges_by_src.setdefault(node.id, set())
        self._in_edges_by_dst.setdefault(node.id, set())

    def add_nodes(self, nodes: Iterable[GraphNode], *, overwrite: bool = False) -> None:
        for node in nodes:
            self.add_node(node, overwrite=overwrite)

    def remove_node(self, node_id: str, *, cascade: bool = True) -> None:
        if node_id not in self._nodes_by_id:
            raise ValueError(f"Unknown node id: {node_id}")

        incident_edge_ids = set(self._out_edges_by_src.get(node_id, set())) | set(self._in_edges_by_dst.get(node_id, set()))
        if incident_edge_ids and not cascade:
            raise ValueError(f"Cannot remove node {node_id}: {len(incident_edge_ids)} incident edges")

        for edge_id in list(incident_edge_ids):
            self.remove_edge(edge_id)

        self._nodes_by_id.pop(node_id, None)
        self._out_edges_by_src.pop(node_id, None)
        self._in_edges_by_dst.pop(node_id, None)

    def get_node(self, node_id: str) -> GraphNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as e:
            raise ValueError(f"Unknown node id: {node_id}") from e

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes_by_id

    def iter_nodes(self) -> Iterable[GraphNode]:
        return self._nodes_by_id.values()

    def count_nodes(self) -> int:
        return len(self._nodes_by_id)

    def get_node_ids(self) -> list[str]:
        return list(self._nodes_by_id.keys())

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

    def add_edges(self, edges: Iterable[GraphEdge], *, overwrite: bool = False) -> None:
        for edge in edges:
            self.add_edge(edge, overwrite=overwrite)

    def remove_edge(self, edge_id: str) -> None:
        if edge_id not in self._edges_by_id:
            raise ValueError(f"Unknown edge id: {edge_id}")

        edge = self._edges_by_id.pop(edge_id)
        self._remove_edge_from_indexes(edge)

    def get_edge(self, edge_id: str) -> GraphEdge:
        try:
            return self._edges_by_id[edge_id]
        except KeyError as e:
            raise ValueError(f"Unknown edge id: {edge_id}") from e

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges_by_id

    def iter_edges(self) -> Iterable[GraphEdge]:
        return self._edges_by_id.values()

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

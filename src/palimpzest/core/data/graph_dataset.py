from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from palimpzest.core.data.induction import (
    CosineSimilarityDecider,
    InductionLog,
    InductionLogEntry,
    InductionSpec,
    KnnBruteForceCandidateGenerator,
)


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


class GraphSnapshot(BaseModel):
    version: int = 1
    graph_id: str
    revision: int
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    induction_log: InductionLog = Field(default_factory=InductionLog)


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

        self._induction_log: InductionLog = InductionLog()

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
            induction_log=self._induction_log,
        )

    @classmethod
    def from_snapshot(cls, snapshot: GraphSnapshot, *, name: str | None = None) -> GraphDataset:
        graph = cls(graph_id=snapshot.graph_id, name=name)
        graph.revision = snapshot.revision
        graph._induction_log = snapshot.induction_log

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
        trace_full_node_text: bool = False,
        trace_node_text_preview_len: int = 240,
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
            trace_full_node_text=trace_full_node_text,
            trace_node_text_preview_len=trace_node_text_preview_len,
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
        decider: callable | None = None,
        decider_id: str | None = None,
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
            decider=decider,
            decider_id=decider_id,
            threshold=threshold,
            overwrite=overwrite,
            edge_id_fn=edge_id_fn,
            edge_attrs_fn=edge_attrs_fn,
        )

        return Dataset(sources=[seed], operator=op, schema=InducedEdgeResult)

    # -----------------
    # Induction log + kNN induction (MVP)
    # -----------------
    def induction_log(self) -> InductionLog:
        return self._induction_log

    def _node_embedding(self, node_id: str, *, embedding_key: str) -> list[float]:
        node = self.get_node(node_id)
        emb = node.embedding if embedding_key == "embedding" else node.attrs.get(embedding_key)
        if emb is None:
            raise ValueError(f"Node {node_id} missing embedding at key {embedding_key!r}")
        if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
            raise ValueError(f"Node {node_id} embedding at key {embedding_key!r} must be a list[float]")
        return [float(x) for x in emb]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b, strict=True):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na**0.5) * (nb**0.5))

    def _impacted_node_ids(self, *, spec_id: str, mode: str) -> set[str]:
        node_ids = set(self._nodes_by_id.keys())
        if mode == "full":
            return node_ids
        if mode != "incremental":
            raise ValueError("mode must be 'full' or 'incremental'")
        entry = self._induction_log.get(spec_id)
        if entry is None:
            return node_ids
        return node_ids - set(entry.processed_node_ids)

    def run_knn_similarity_induction(
        self,
        *,
        edge_type: str = "sim:knn",
        embedding_key: str = "embedding",
        k: int | None = 10,
        threshold: float | None = None,
        include_overlay: bool = True,
        mode: str = "full",
    ):
        """Run a brute-force kNN similarity induction and persist it in the induction log.

        Decisions baked in (per CURRENT_WORKSTREAM):
        - offline batch, re-runnable incrementally for new nodes
        - symmetric edges (implemented as two directed edges), no self-edges
        - no duplicate edges (dedupe by deterministic edge id)
        - keep-first semantics (overwrite=False)
        """

        if (k is None) == (threshold is None):
            raise ValueError("Provide exactly one of: k, threshold")

        spec = InductionSpec(
            edge_type=edge_type,
            include_overlay=include_overlay,
            symmetric=True,
            allow_self_edges=False,
            overwrite=False,
            generator={
                "kind": "knn_bruteforce",
                "params": {
                    "embedding_key": embedding_key,
                    "k": k,
                    "threshold": threshold,
                },
            },
            decider={"kind": "cosine_similarity", "params": {"embedding_key": embedding_key}},
        )
        spec_id = spec.spec_id()

        impacted = self._impacted_node_ids(spec_id=spec_id, mode=mode)
        if not impacted:
            return []

        node_ids = list(self._nodes_by_id.keys())

        def embedding_for_node_id(node_id: str) -> list[float]:
            return self._node_embedding(node_id, embedding_key=embedding_key)

        decider = CosineSimilarityDecider(embedding_for_node_id=embedding_for_node_id)
        generator = KnnBruteForceCandidateGenerator(
            k=k,
            threshold=threshold,
            symmetric=True,
            allow_self_edges=False,
        )

        candidate_pairs = list(generator.generate_pairs(node_ids=node_ids, impacted_node_ids=impacted, score_pair=decider.score_pair))

        def edge_attrs(src: str, dst: str, edge_type_: str, score: float | None) -> dict:
            attrs: dict[str, Any] = {
                "spec_id": spec_id,
                "method": "knn_similarity",
                "embedding_key": embedding_key,
            }
            if score is not None:
                attrs["score"] = score
            if k is not None:
                attrs["k"] = k
            if threshold is not None:
                attrs["threshold"] = threshold
            return attrs

        def score_decider(src_id: str, src: GraphNode, dst_id: str, dst: GraphNode, graph: GraphDataset) -> float:
            return decider.score_pair(src_id=src_id, dst_id=dst_id)

        ds = self.induce_edges(
            candidate_pairs=candidate_pairs,
            edge_type=edge_type,
            include_overlay=include_overlay,
            decider=score_decider,
            decider_id=f"cosine_similarity:{embedding_key}",
            # threshold only matters for legacy predicate gating; candidate generation already filtered.
            threshold=(threshold if threshold is not None else 0.0),
            overwrite=False,
            edge_attrs_fn=edge_attrs,
        )

        out = ds.run()

        # Update induction log only after a successful run.
        entry = InductionLogEntry(spec=spec, processed_node_ids=sorted(self._nodes_by_id.keys()))
        self._induction_log.upsert(entry)
        return out

    def add_induction(self, spec: InductionSpec) -> str:
        spec_id = spec.spec_id()
        if self._induction_log.get(spec_id) is None:
            self._induction_log.upsert(InductionLogEntry(spec=spec, processed_node_ids=[]))
        return spec_id

    def run_induction(self, spec_id: str, *, mode: str = "full"):
        entry = self._induction_log.get(spec_id)
        if entry is None:
            raise ValueError(f"Unknown induction spec_id: {spec_id}")

        gen_kind = entry.spec.generator.kind
        gen_params = entry.spec.generator.params
        dec_kind = entry.spec.decider.kind
        dec_params = entry.spec.decider.params

        if gen_kind == "knn_bruteforce" and dec_kind == "cosine_similarity":
            embedding_key = str(dec_params.get("embedding_key", gen_params.get("embedding_key", "embedding")))
            return self.run_knn_similarity_induction(
                edge_type=entry.spec.edge_type,
                embedding_key=embedding_key,
                k=gen_params.get("k"),
                threshold=gen_params.get("threshold"),
                include_overlay=entry.spec.include_overlay,
                mode=mode,
            )

        if dec_kind in {"predicate", "predicate_compound"}:
            return self.run_predicate_induction(spec_id=spec_id, mode=mode)

        raise ValueError(f"Unknown induction spec: generator={gen_kind}, decider={dec_kind}")

    # -----------------
    # Predicate induction (general references)
    # -----------------
    @staticmethod
    def _get_string_field(node: GraphNode, field: str) -> str | None:
        if field == "label":
            return node.label
        if field == "text":
            return node.text
        if field == "attrs.name":
            v = (node.attrs or {}).get("name")
            return v if isinstance(v, str) else None
        if field == "attrs.metadata.name":
            md = (node.attrs or {}).get("metadata")
            if isinstance(md, dict):
                v = md.get("name")
                return v if isinstance(v, str) else None
            return None
        raise ValueError(f"Unknown field: {field}")

    @classmethod
    def _extract_strings(cls, node: GraphNode, *, fields: list[str]) -> list[str]:
        out: list[str] = []
        for f in fields:
            v = cls._get_string_field(node, f)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out

    @staticmethod
    def _extract_attr(node: GraphNode, *, path: str) -> object:
        if not path.startswith("attrs."):
            raise ValueError("attr path must start with 'attrs.'")
        cur: object = node.attrs
        for part in path.split(".")[1:]:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]

    @staticmethod
    def _contains_with_boundaries(*, haystack: str, needle: str) -> bool:
        h = haystack.lower()
        n = needle.lower()
        start = 0
        while True:
            idx = h.find(n, start)
            if idx < 0:
                return False
            left_ok = idx == 0 or not h[idx - 1].isalnum()
            right_idx = idx + len(n)
            right_ok = right_idx >= len(h) or not h[right_idx].isalnum()
            if left_ok and right_ok:
                return True
            start = idx + 1

    def _generate_candidate_pairs(self, *, generator_kind: str, generator_params: dict[str, Any], impacted: set[str]) -> list[tuple[str, str]]:
        if generator_kind == "all_pairs":
            allow_self = bool(generator_params.get("allow_self_edges", False))
            node_ids = list(self._nodes_by_id.keys())
            pairs: list[tuple[str, str]] = []
            for src in impacted:
                for dst in node_ids:
                    if not allow_self and src == dst:
                        continue
                    pairs.append((src, dst))
            return pairs

        if generator_kind == "text_anchor":
            source_text_field = str(generator_params.get("source_text_field", "text"))
            target_fields = generator_params.get("target_fields") or ["label", "attrs.name", "attrs.metadata.name"]
            if not isinstance(target_fields, list) or not all(isinstance(f, str) for f in target_fields):
                raise ValueError("text_anchor requires target_fields: list[str]")
            min_anchor_len = int(generator_params.get("min_anchor_len", 5))

            anchor_to_targets: dict[str, list[str]] = {}
            target_strings_by_id: dict[str, list[str]] = {}
            for node_id, node in self._nodes_by_id.items():
                strings = self._extract_strings(node, fields=target_fields)
                if strings:
                    target_strings_by_id[node_id] = strings
                    toks = set()
                    for s in strings:
                        toks.update(self._tokenize(s))
                    if toks:
                        anchor = max(toks, key=len)
                        if len(anchor) >= min_anchor_len:
                            anchor_to_targets.setdefault(anchor, []).append(node_id)

            pairs: set[tuple[str, str]] = set()
            for src in impacted:
                node = self.get_node(src)
                src_text = self._get_string_field(node, source_text_field) or ""
                if not src_text:
                    continue
                src_tokens = set(self._tokenize(src_text))
                for tok in src_tokens:
                    for dst in anchor_to_targets.get(tok, []):
                        if dst == src:
                            continue
                        pairs.add((src, dst))
            return sorted(pairs)

        if generator_kind == "attr_bucket":
            attr_path = str(generator_params.get("attr_path"))
            if not attr_path:
                raise ValueError("attr_bucket requires attr_path")
            max_bucket = int(generator_params.get("max_bucket_size", 500))
            allow_self = bool(generator_params.get("allow_self_edges", False))

            buckets: dict[str, list[str]] = {}
            for node_id, node in self._nodes_by_id.items():
                v = self._extract_attr(node, path=attr_path)
                if v is None:
                    continue
                buckets.setdefault(json.dumps(v, sort_keys=True, default=str), []).append(node_id)

            pairs: set[tuple[str, str]] = set()
            for src in impacted:
                v = self._extract_attr(self.get_node(src), path=attr_path)
                if v is None:
                    continue
                bucket = buckets.get(json.dumps(v, sort_keys=True, default=str), [])
                if len(bucket) > max_bucket:
                    continue
                for dst in bucket:
                    if not allow_self and src == dst:
                        continue
                    pairs.add((src, dst))
            return sorted(pairs)

        raise ValueError(f"Unknown generator kind: {generator_kind}")

    def _eval_predicate(
        self,
        *,
        predicate: dict[str, Any],
        src_id: str,
        dst_id: str,
    ) -> tuple[bool, dict[str, Any] | None]:
        kind = predicate.get("kind")
        params = predicate.get("params") or {}
        if kind == "text_contains":
            source_field = str(params.get("source_field", "text"))
            target_fields = params.get("target_fields") or ["label", "attrs.name", "attrs.metadata.name"]
            if not isinstance(target_fields, list) or not all(isinstance(f, str) for f in target_fields):
                raise ValueError("text_contains requires target_fields: list[str]")
            boundaries = bool(params.get("boundaries", True))

            src = self.get_node(src_id)
            dst = self.get_node(dst_id)
            src_text = self._get_string_field(src, source_field) or ""
            if not src_text:
                return False, None
            dst_strings = self._extract_strings(dst, fields=target_fields)
            for s in dst_strings:
                if boundaries:
                    ok = self._contains_with_boundaries(haystack=src_text, needle=s)
                else:
                    ok = s.lower() in src_text.lower()
                if ok:
                    return True, {"matched": s, "source_field": source_field}
            return False, None

        if kind == "regex_match":
            source_field = str(params.get("source_field", "text"))
            patterns = params.get("patterns") or []
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
                raise ValueError("regex_match requires patterns: list[str] | str")
            target_fields = params.get("target_fields") or ["label", "attrs.name", "attrs.metadata.name"]
            if not isinstance(target_fields, list) or not all(isinstance(f, str) for f in target_fields):
                raise ValueError("regex_match requires target_fields: list[str]")

            src = self.get_node(src_id)
            dst = self.get_node(dst_id)
            src_text = self._get_string_field(src, source_field) or ""
            if not src_text:
                return False, None
            dst_tokens = {s.upper() for s in self._extract_strings(dst, fields=target_fields)}
            if not dst_tokens:
                return False, None
            for pat in patterns:
                rx = re.compile(pat)
                for m in rx.finditer(src_text):
                    tok = m.group(0)
                    if isinstance(tok, str) and tok.upper() in dst_tokens:
                        return True, {"matched": tok, "pattern": pat, "source_field": source_field}
            return False, None

        if kind == "attr_equals":
            src_attr = str(params.get("src_attr"))
            dst_attr = str(params.get("dst_attr", src_attr))
            if not src_attr:
                raise ValueError("attr_equals requires src_attr")
            src = self.get_node(src_id)
            dst = self.get_node(dst_id)
            sv = self._extract_attr(src, path=src_attr)
            dv = self._extract_attr(dst, path=dst_attr)
            if sv is None or dv is None:
                return False, None
            if sv == dv:
                return True, {"src_attr": src_attr, "dst_attr": dst_attr, "value": sv}
            return False, None

        raise ValueError(f"Unknown predicate kind: {kind}")

    def run_predicate_induction(self, *, spec_id: str, mode: str = "full"):
        entry = self._induction_log.get(spec_id)
        if entry is None:
            raise ValueError(f"Unknown induction spec_id: {spec_id}")

        generator_kind = entry.spec.generator.kind
        generator_params = entry.spec.generator.params
        dec_kind = entry.spec.decider.kind
        dec_params = entry.spec.decider.params

        if dec_kind == "predicate":
            predicates = [dec_params]
            mode_kind = "all"
        else:
            mode_kind = str(dec_params.get("mode", "all"))
            predicates = dec_params.get("predicates") or []
        if mode_kind not in {"all", "any"}:
            raise ValueError("predicate_compound requires mode: 'all'|'any'")
        if not isinstance(predicates, list) or not all(isinstance(p, dict) for p in predicates):
            raise ValueError("predicate induction requires predicates: list[dict]")

        impacted = self._impacted_node_ids(spec_id=spec_id, mode=mode)
        if not impacted:
            return []

        candidate_pairs = self._generate_candidate_pairs(generator_kind=generator_kind, generator_params=generator_params, impacted=impacted)
        if not candidate_pairs:
            return []

        accepted: list[tuple[str, str]] = []
        evidence_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

        for src_id, dst_id in candidate_pairs:
            if not entry.spec.allow_self_edges and src_id == dst_id:
                continue
            decisions: list[tuple[bool, dict[str, Any] | None, dict[str, Any]]] = []
            for pred in predicates:
                ok, ev = self._eval_predicate(predicate=pred, src_id=src_id, dst_id=dst_id)
                decisions.append((ok, ev, pred))

            if mode_kind == "all":
                passed = all(ok for ok, _ev, _p in decisions)
            else:
                passed = any(ok for ok, _ev, _p in decisions)

            if not passed:
                continue

            pair = (src_id, dst_id)
            accepted.append(pair)
            evidence_by_pair[pair] = {
                "mode": mode_kind,
                "predicates": [{"kind": p.get("kind"), "params": p.get("params") or {}} for _ok, _ev, p in decisions],
                "evidence": {p.get("kind"): ev for ok, ev, p in decisions if ok and ev is not None},
            }

        if not accepted:
            return []

        # Apply symmetry if requested.
        if entry.spec.symmetric:
            extra = []
            for a, b in accepted:
                if a == b:
                    continue
                extra.append((b, a))
                if (b, a) not in evidence_by_pair:
                    evidence_by_pair[(b, a)] = evidence_by_pair[(a, b)]
            accepted = sorted(set(accepted + extra))

        def edge_attrs(src: str, dst: str, edge_type_: str, score: float | None) -> dict:
            _ = edge_type_
            _ = score
            return {
                "spec_id": spec_id,
                "method": "predicate",
                **(evidence_by_pair.get((src, dst)) or {}),
            }

        def accept(_src_id: str, _src: GraphNode, _dst_id: str, _dst: GraphNode, _graph: GraphDataset) -> bool:
            return True

        out = self.induce_edges(
            candidate_pairs=accepted,
            edge_type=entry.spec.edge_type,
            include_overlay=entry.spec.include_overlay,
            decider=accept,
            decider_id="always_accept",
            threshold=0.0,
            overwrite=False,
            edge_attrs_fn=edge_attrs,
        ).run()

        self._induction_log.upsert(InductionLogEntry(spec=entry.spec, processed_node_ids=sorted(self._nodes_by_id.keys())))
        return out

    def add_predicate_induction(
        self,
        *,
        edge_type: str,
        generator_kind: str,
        generator_params: dict[str, Any] | None = None,
        predicates: list[dict[str, Any]] | None = None,
        predicate_mode: str = "all",
        include_overlay: bool = True,
        symmetric: bool = False,
    ) -> str:
        generator_params = {} if generator_params is None else generator_params
        predicates = [] if predicates is None else predicates
        if predicate_mode not in {"all", "any"}:
            raise ValueError("predicate_mode must be 'all' or 'any'")
        spec = InductionSpec(
            edge_type=edge_type,
            include_overlay=include_overlay,
            symmetric=bool(symmetric),
            allow_self_edges=False,
            overwrite=False,
            generator={"kind": generator_kind, "params": generator_params},
            decider={"kind": "predicate_compound", "params": {"mode": predicate_mode, "predicates": predicates}},
        )
        return self.add_induction(spec)

    def add_knn_similarity_topk(
        self,
        *,
        edge_type: str = "sim:knn",
        embedding_key: str = "embedding",
        k: int = 10,
        include_overlay: bool = True,
    ) -> str:
        spec = InductionSpec(
            edge_type=edge_type,
            include_overlay=include_overlay,
            symmetric=True,
            allow_self_edges=False,
            overwrite=False,
            generator={"kind": "knn_bruteforce", "params": {"embedding_key": embedding_key, "k": k, "threshold": None}},
            decider={"kind": "cosine_similarity", "params": {"embedding_key": embedding_key}},
        )
        return self.add_induction(spec)

    def add_knn_similarity_threshold(
        self,
        *,
        edge_type: str = "sim:knn",
        embedding_key: str = "embedding",
        threshold: float,
        include_overlay: bool = True,
    ) -> str:
        spec = InductionSpec(
            edge_type=edge_type,
            include_overlay=include_overlay,
            symmetric=True,
            allow_self_edges=False,
            overwrite=False,
            generator={"kind": "knn_bruteforce", "params": {"embedding_key": embedding_key, "k": None, "threshold": threshold}},
            decider={"kind": "cosine_similarity", "params": {"embedding_key": embedding_key}},
        )
        return self.add_induction(spec)

    def reapply_inductions_incremental(self):
        """Re-run all recorded inductions in incremental mode (new nodes only)."""

        return [self.run_induction(entry.spec.spec_id(), mode="incremental") for entry in list(self._induction_log.entries)]

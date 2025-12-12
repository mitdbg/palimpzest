from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Callable

from pydantic import BaseModel, Field

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator


class GraphTraverseSeed(BaseModel):
    start_node_ids: list[str] = Field(description="Start nodes for traversal")


class GraphTraversalResult(BaseModel):
    node_id: str
    depth: int
    score: float
    path_node_ids: list[str]
    path_edge_ids: list[str]


@dataclass(order=True)
class _PQItem:
    neg_score: float
    node_id: str
    depth: int
    path_node_ids: list[str]
    path_edge_ids: list[str]
    decision_data: dict[str, Any] | None = dataclass_field(default=None, compare=False)


class TraverseOp(PhysicalOperator):
    """Physical beam-search traversal over a GraphDataset."""

    def __init__(
        self,
        graph: GraphDataset,
        start_field: str = "start_node_ids",
        edge_type: str | None = None,
        include_overlay: bool = True,
        max_steps: int = 128,
        allow_revisit: bool = False,
        ranker: Callable[[str, GraphNode, GraphEdge | None, str | None, list[str], list[str]], float] | None = None,
        ranker_id: str | None = None,
        visit_filter: Callable[[str, GraphNode, int, float, list[str], list[str]], bool] | None = None,
        visit_filter_id: str | None = None,
        admittance: Callable[[str, GraphNode, int, float, list[str], list[str]], object] | None = None,
        admittance_id: str | None = None,
        termination: Callable[[dict], object] | None = None,
        termination_id: str | None = None,
        node_program: Callable[..., object] | None = None,
        node_program_id: str | None = None,
        node_program_config: object | None = None,
        decision_program: Callable[..., object] | None = None,
        decision_program_id: str | None = None,
        decision_program_config: object | None = None,
        tracer: Callable[[dict[str, Any]], None] | None = None,
        tracer_id: str | None = None,
        trace_run_id: str | None = None,
        trace_full_node_text: bool = False,
        trace_node_text_preview_len: int = 240,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph = graph
        self.start_field = start_field
        self.edge_type = edge_type
        self.include_overlay = include_overlay
        self.max_steps = max_steps
        self.allow_revisit = allow_revisit
        self.ranker = ranker
        self.ranker_id = ranker_id
        self.visit_filter = visit_filter
        self.visit_filter_id = visit_filter_id
        self.admittance = admittance
        self.admittance_id = admittance_id
        self.termination = termination
        self.termination_id = termination_id
        self.node_program = node_program
        self.node_program_id = node_program_id
        self.node_program_config = node_program_config
        self.decision_program = decision_program
        self.decision_program_id = decision_program_id
        self.decision_program_config = decision_program_config
        self.tracer = tracer
        self.tracer_id = tracer_id
        self.trace_run_id = trace_run_id
        self.trace_full_node_text = trace_full_node_text
        self.trace_node_text_preview_len = trace_node_text_preview_len

        self._trace_seq: int = 0

    def __str__(self) -> str:
        op = super().__str__()
        op += f"    Graph: {self.graph.graph_id}@{self.graph.revision}\n"
        op += f"    Start Field: {self.start_field}\n"
        op += f"    Edge Type: {self.edge_type}\n"
        op += f"    Max Steps: {self.max_steps}\n"
        return op

    def get_id_params(self) -> dict:
        id_params = super().get_id_params()
        return {
            "graph_id": self.graph.graph_id,
            "graph_revision": self.graph.revision,
            "start_field": self.start_field,
            "edge_type": self.edge_type,
            "include_overlay": self.include_overlay,
            "max_steps": self.max_steps,
            "allow_revisit": self.allow_revisit,
            "ranker_id": self.ranker_id,
            "visit_filter_id": self.visit_filter_id,
            "admittance_id": self.admittance_id,
            "termination_id": self.termination_id,
            "node_program_id": self.node_program_id,
            "decision_program_id": self.decision_program_id,
            "tracer_id": self.tracer_id,
            **id_params,
        }

    def get_op_params(self) -> dict:
        op_params = super().get_op_params()
        return {
            "graph": self.graph,
            "start_field": self.start_field,
            "edge_type": self.edge_type,
            "include_overlay": self.include_overlay,
            "max_steps": self.max_steps,
            "allow_revisit": self.allow_revisit,
            "ranker": self.ranker,
            "ranker_id": self.ranker_id,
            "visit_filter": self.visit_filter,
            "visit_filter_id": self.visit_filter_id,
            "admittance": self.admittance,
            "admittance_id": self.admittance_id,
            "termination": self.termination,
            "termination_id": self.termination_id,
            "node_program": self.node_program,
            "node_program_id": self.node_program_id,
            "node_program_config": self.node_program_config,
            "decision_program": self.decision_program,
            "decision_program_id": self.decision_program_id,
            "decision_program_config": self.decision_program_config,
            "tracer": self.tracer,
            "tracer_id": self.tracer_id,
            "trace_run_id": self.trace_run_id,
            **op_params,
        }

    def _trace(self, event_type: str, **data: Any) -> None:
        if self.tracer is None:
            return
        self._trace_seq += 1
        data.setdefault("ts_ms", int(time.time() * 1000))
        data.setdefault("seq", int(self._trace_seq))
        if self.trace_run_id:
            data.setdefault("run_id", str(self.trace_run_id))
        payload: dict[str, Any] = {"event_type": event_type, **data}
        try:
            self.tracer(payload)
        except Exception:
            # Tracing must never break traversal.
            return

    def _node_for_trace(self, node: GraphNode) -> dict[str, Any]:
        if self.trace_full_node_text:
            return {
                **node.model_dump(mode="json", exclude={"embedding"}),
                "embedding_len": len(node.embedding) if isinstance(node.embedding, list) else None,
                "text_len": len(node.text) if isinstance(node.text, str) else None,
            }

        txt = node.text if isinstance(node.text, str) else None
        preview = (txt[: self.trace_node_text_preview_len] if isinstance(txt, str) else None) if self.trace_node_text_preview_len > 0 else None
        return {
            **node.model_dump(mode="json", exclude={"embedding", "text"}),
            "text_preview": preview,
            "text_len": len(txt) if isinstance(txt, str) else None,
            "embedding_len": len(node.embedding) if isinstance(node.embedding, list) else None,
        }

    def _run_node_program(
        self,
        *,
        node_id: str,
        node: GraphNode,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> object | None:
        if self.node_program is None:
            return None

        program = self.node_program
        program_result = program(
            node_id=node_id,
            node=node,
            graph=self.graph,
            depth=depth,
            score=score,
            path_node_ids=path_node_ids,
            path_edge_ids=path_edge_ids,
        )

        # Allow returning a Dataset (preferred) or a DataRecordCollection directly.
        from palimpzest.core.data.dataset import Dataset
        from palimpzest.core.elements.records import DataRecordCollection

        if isinstance(program_result, Dataset):
            if self.node_program_config is None:
                return program_result.run()
            cfg = self.node_program_config
            # QueryProcessorFactory normalization may mutate config objects; avoid cross-call reuse.
            if hasattr(cfg, "model_copy"):
                try:
                    cfg = cfg.model_copy(deep=True)
                except Exception:
                    cfg = self.node_program_config
            return program_result.run(config=cfg)  # type: ignore[arg-type]

        if isinstance(program_result, DataRecordCollection):
            return program_result

        raise TypeError(
            "node_program must return a palimpzest Dataset (preferred) or a DataRecordCollection; "
            f"got {type(program_result)}"
        )

    def _run_decision_program(
        self,
        *,
        node_id: str,
        node: GraphNode,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> object | None:
        if self.decision_program is None:
            return None

        program = self.decision_program
        program_result = program(
            node_id=node_id,
            node=node,
            graph=self.graph,
            depth=depth,
            score=score,
            path_node_ids=path_node_ids,
            path_edge_ids=path_edge_ids,
        )

        from palimpzest.core.data.dataset import Dataset
        from palimpzest.core.elements.records import DataRecordCollection

        if isinstance(program_result, Dataset):
            if self.decision_program_config is None:
                return program_result.run()
            cfg = self.decision_program_config
            # QueryProcessorFactory normalization may mutate config objects; avoid cross-call reuse.
            if hasattr(cfg, "model_copy"):
                try:
                    cfg = cfg.model_copy(deep=True)
                except Exception:
                    cfg = self.decision_program_config
            return program_result.run(config=cfg)  # type: ignore[arg-type]

        if isinstance(program_result, DataRecordCollection):
            return program_result

        raise TypeError(
            "decision_program must return a palimpzest Dataset (preferred) or a DataRecordCollection; "
            f"got {type(program_result)}"
        )

    def _decision_for_enqueue(
        self,
        *,
        node_id: str,
        node: GraphNode,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Compute queue score + optional gating via decision_program."""

        if self.decision_program is None:
            return True, float(score), {}

        decision_result = self._run_decision_program(
            node_id=node_id,
            node=node,
            depth=depth,
            score=score,
            path_node_ids=list(path_node_ids),
            path_edge_ids=list(path_edge_ids),
        )

        decision_data: dict[str, Any] = {}
        try:
            first = next(iter(decision_result))  # type: ignore[call-overload]
        except StopIteration:
            # Decision program filtered everything out.
            return False, float(score), {}
        except Exception:
            # Decision program failure must not break traversal; treat as neutral.
            return True, float(score), {}

        try:
            decision_data = first.to_dict(include_bytes=False) if hasattr(first, "to_dict") else {}
        except Exception:
            decision_data = {}

        if "admit" in decision_data and not bool(decision_data.get("admit")):
            return False, float(score), decision_data

        if isinstance(decision_data.get("rank_score"), (int, float)):
            score = float(decision_data["rank_score"])

        return True, float(score), decision_data

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # Conservative default: one output per step, O(max_steps) work.
        return OperatorCostEstimates(
            cardinality=min(source_op_cost_estimates.cardinality * self.max_steps, source_op_cost_estimates.cardinality * 1000),
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def _iter_out_edges(self, node_id: str):
        for edge in self.graph.iter_out_edges(node_id):
            if not self.include_overlay and edge.type.startswith("overlay:"):
                continue
            if self.edge_type is not None and edge.type != self.edge_type:
                continue
            yield edge

    def _iter_adjacent_edges(self, node_id: str):
        """Yield edges adjacent to node_id (outgoing + incoming), deduped by edge id."""
        seen: set[str] = set()
        for edge in self.graph.iter_out_edges(node_id):
            if edge.id in seen:
                continue
            seen.add(edge.id)
            if not self.include_overlay and edge.type.startswith("overlay:"):
                continue
            if self.edge_type is not None and edge.type != self.edge_type:
                continue
            yield edge

        for edge in self.graph.iter_in_edges(node_id):
            if edge.id in seen:
                continue
            seen.add(edge.id)
            if not self.include_overlay and edge.type.startswith("overlay:"):
                continue
            if self.edge_type is not None and edge.type != self.edge_type:
                continue
            yield edge

    def _score(
        self,
        node_id: str,
        node: GraphNode,
        edge: GraphEdge | None,
        from_node_id: str | None,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> float:
        if self.ranker is None:
            return 0.0
        return float(self.ranker(node_id, node, edge, from_node_id, path_node_ids, path_edge_ids))

    def _passes_filter(
        self,
        node_id: str,
        node: GraphNode,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> bool:
        if self.visit_filter is None:
            return True
        return bool(self.visit_filter(node_id, node, depth, score, path_node_ids, path_edge_ids))

    def _admittance_decision(
        self,
        node_id: str,
        node: GraphNode,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.admittance is None:
            return True, None
        out = self.admittance(node_id, node, depth, score, path_node_ids, path_edge_ids)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool) and isinstance(out[1], dict):
            return bool(out[0]), out[1]
        return bool(out), None

    def _termination_decision(self, state: dict) -> tuple[bool, dict[str, Any] | None]:
        if self.termination is None:
            return False, None
        out = self.termination(state)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool) and isinstance(out[1], dict):
            return bool(out[0]), out[1]
        return bool(out), None

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_node_ids = getattr(candidate, self.start_field, None)
        if start_node_ids is None:
            raise ValueError(f"TraverseOp missing start field: {self.start_field}")
        if isinstance(start_node_ids, str):
            start_node_ids = [start_node_ids]

        self._trace(
            "traverse_init",
            graph_id=self.graph.graph_id,
            graph_revision=self.graph.revision,
            start_node_ids=list(start_node_ids),
            edge_type=self.edge_type,
            include_overlay=self.include_overlay,
            max_steps=self.max_steps,
            allow_revisit=self.allow_revisit,
            ranker_id=self.ranker_id,
            visit_filter_id=self.visit_filter_id,
            admittance_id=self.admittance_id,
            termination_id=self.termination_id,
            node_program_id=self.node_program_id,
            decision_program_id=self.decision_program_id,
            tracer_id=self.tracer_id,
        )

        # Note: we keep a separate "seen" set to prevent re-enqueue when allow_revisit=False,
        # and a "visited" set to report unique popped node count (for tracing/termination state).
        seen: set[str] = set()
        visited: set[str] = set()
        pq: list[_PQItem] = []

        # Seed frontier
        for start_id in start_node_ids:
            if not self.graph.has_node(start_id):
                self._trace("seed_skip_missing_node", node_id=start_id)
                continue
            node = self.graph.get_node(start_id)
            base_score = self._score(start_id, node, None, None, [start_id], [])
            accept, score, decision_data = self._decision_for_enqueue(
                node_id=start_id,
                node=node,
                depth=0,
                score=base_score,
                path_node_ids=[start_id],
                path_edge_ids=[],
            )
            if not accept:
                self._trace(
                    "seed_skip_program",
                    node_id=start_id,
                    depth=0,
                    score=float(base_score),
                    decision_program_id=self.decision_program_id,
                )
                continue
            self._trace(
                "seed_node",
                node_id=start_id,
                depth=0,
                score=float(score),
                node=self._node_for_trace(node),
                rank_meta=decision_data.get("rank_meta") if isinstance(decision_data, dict) else None,
            )
            heapq.heappush(
                pq,
                _PQItem(
                    neg_score=-score,
                    node_id=start_id,
                    depth=0,
                    path_node_ids=[start_id],
                    path_edge_ids=[],
                    decision_data=decision_data,
                ),
            )
            if not self.allow_revisit:
                seen.add(start_id)

        results: list[DataRecord] = []
        stats: list[RecordOpStats] = []

        admitted_nodes = 0
        skip_counts: Counter[str] = Counter()
        node_visits: Counter[str] = Counter()
        edge_traversals: Counter[str] = Counter()
        expansion_counts: Counter[str] = Counter()

        steps = 0
        while pq and steps < self.max_steps:
            steps += 1
            item = heapq.heappop(pq)
            node_id = item.node_id

            self._trace(
                "step_begin",
                step=steps,
                node_id=node_id,
                popped={
                    "node_id": node_id,
                    "depth": item.depth,
                    "score": float(-item.neg_score),
                    "path_node_ids": list(item.path_node_ids),
                    "path_edge_ids": list(item.path_edge_ids),
                },
                frontier_size_before=len(pq),
                visited_count=len(visited),
                admitted_nodes=admitted_nodes,
            )

            if not self.graph.has_node(node_id):
                self._trace("step_skip_missing_node", step=steps, node_id=node_id)
                skip_counts["missing_node"] += 1
                continue

            visited.add(node_id)

            node = self.graph.get_node(node_id)
            score = -item.neg_score

            self._trace(
                "step_node_loaded",
                step=steps,
                node_id=node_id,
                node=self._node_for_trace(node),
            )

            # Decision program decisions are computed at enqueue time (for PQ ordering and gating).
            decision_data: dict[str, Any] = dict(item.decision_data or {})
            if self.decision_program is not None:
                decision_payload = decision_data.get("decision") if isinstance(decision_data.get("decision"), dict) else None
                why = decision_payload.get("reason") if isinstance(decision_payload, dict) else None
                self._trace(
                    "step_gate_program",
                    step=steps,
                    node_id=node_id,
                    admitted=bool(decision_data.get("admit", True)),
                    decision_program_id=self.decision_program_id,
                    rank_meta=decision_data.get("rank_meta"),
                    why=why,
                    decision={
                        "passed": bool(decision_data.get("admit", True)),
                        "kind": "decision_program",
                        "id": self.decision_program_id,
                        "why": why,
                        "payload": decision_payload,
                    },
                )

            # Unified gate: node must pass both visit_filter and admittance (if provided)
            # to be processed (emit + expand).
            passes_filter = self._passes_filter(node_id, node, item.depth, score, item.path_node_ids, item.path_edge_ids)
            self._trace(
                "step_gate_visit_filter",
                step=steps,
                node_id=node_id,
                passed=bool(passes_filter),
                visit_filter_id=self.visit_filter_id,
                depth=item.depth,
                score=float(score),
                decision={
                    "passed": bool(passes_filter),
                    "kind": "visit_filter",
                    "id": self.visit_filter_id,
                    "why": None,
                },
            )
            if not passes_filter:
                self._trace(
                    "step_summary",
                    step=steps,
                    node_id=node_id,
                    admitted=False,
                    skipped=True,
                    skip_reason="visit_filter_rejected",
                    frontier_size_after=len(pq),
                    visited_count=len(visited),
                    admitted_nodes=admitted_nodes,
                    emitted_records=len(results),
                )
                self._trace(
                    "step_end",
                    step=steps,
                    node_id=node_id,
                    skipped=True,
                    skip_reason="visit_filter_rejected",
                    frontier_size_after=len(pq),
                    visited_count=len(visited),
                    admitted_nodes=admitted_nodes,
                    emitted_records=len(results),
                )
                skip_counts["visit_filter_rejected"] += 1
                continue

            is_admitted, adm_meta = self._admittance_decision(node_id, node, item.depth, score, item.path_node_ids, item.path_edge_ids)
            why = (adm_meta or {}).get("reason")
            self._trace(
                "step_gate_admittance",
                step=steps,
                node_id=node_id,
                admitted=bool(is_admitted),
                admittance_id=self.admittance_id,
                depth=item.depth,
                score=float(score),
                prompt=(adm_meta or {}).get("prompt"),
                raw_output=(adm_meta or {}).get("raw_output"),
                reason=(adm_meta or {}).get("reason"),
                why=why,
                model=(adm_meta or {}).get("model"),
                latency_s=(adm_meta or {}).get("latency_s"),
                cache_hit=(adm_meta or {}).get("cache_hit"),
                input_tokens=(adm_meta or {}).get("input_tokens"),
                output_tokens=(adm_meta or {}).get("output_tokens"),
                cached_tokens=(adm_meta or {}).get("cached_tokens"),
                cost_usd=(adm_meta or {}).get("cost_usd"),
                cached_cost_usd=(adm_meta or {}).get("cached_cost_usd"),
                cached_input_tokens=(adm_meta or {}).get("cached_input_tokens"),
                cached_output_tokens=(adm_meta or {}).get("cached_output_tokens"),
                cached_cached_tokens=(adm_meta or {}).get("cached_cached_tokens"),
                decision={
                    "passed": bool(is_admitted),
                    "kind": "admittance",
                    "id": self.admittance_id,
                    "model": (adm_meta or {}).get("model"),
                    "reason": (adm_meta or {}).get("reason"),
                    "why": why,
                    "cache_hit": (adm_meta or {}).get("cache_hit"),
                    "latency_s": (adm_meta or {}).get("latency_s"),
                    "tokens": {
                        "input": (adm_meta or {}).get("input_tokens"),
                        "output": (adm_meta or {}).get("output_tokens"),
                        "cached": (adm_meta or {}).get("cached_tokens"),
                    },
                    "cost_usd": (adm_meta or {}).get("cost_usd"),
                },
            )
            if not is_admitted:
                self._trace(
                    "step_summary",
                    step=steps,
                    node_id=node_id,
                    admitted=False,
                    skipped=True,
                    skip_reason="admittance_rejected",
                    frontier_size_after=len(pq),
                    visited_count=len(visited),
                    admitted_nodes=admitted_nodes,
                    emitted_records=len(results),
                )
                self._trace(
                    "step_end",
                    step=steps,
                    node_id=node_id,
                    skipped=True,
                    skip_reason="admittance_rejected",
                    frontier_size_after=len(pq),
                    visited_count=len(visited),
                    admitted_nodes=admitted_nodes,
                    emitted_records=len(results),
                )
                skip_counts["admittance_rejected"] += 1
                continue

            admitted_nodes += 1
            node_visits[str(node_id)] += 1
            for edge_id in item.path_edge_ids:
                edge_traversals[str(edge_id)] += 1

            traversal_data_item = {
                "node_id": node_id,
                "depth": item.depth,
                "score": score,
                "path_node_ids": list(item.path_node_ids),
                "path_edge_ids": list(item.path_edge_ids),
            }
            if decision_data:
                traversal_data_item.update(decision_data)

            # Always materialize a traversal record to use as lineage parent.
            traversal_dr = DataRecord.from_parent(
                schema=self.output_schema,
                data_item=dict(traversal_data_item),
                parent_record=candidate,
                cardinality_idx=len(results),
            )

            program_result = self._run_node_program(
                node_id=node_id,
                node=node,
                depth=item.depth,
                score=score,
                path_node_ids=list(item.path_node_ids),
                path_edge_ids=list(item.path_edge_ids),
            )

            self._trace(
                "step_node_program_done",
                step=steps,
                node_id=node_id,
                ran=self.node_program is not None,
                produced=program_result is not None,
            )

            emitted_any = False
            if program_result is not None:
                # program_result is a DataRecordCollection
                emitted_program = 0
                for program_dr in program_result:  # type: ignore[union-attr]
                    program_data = program_dr.to_dict(include_bytes=False)
                    combined = {**traversal_data_item, **program_data}
                    dr = DataRecord.from_parent(
                        schema=self.output_schema,
                        data_item=combined,
                        parent_record=traversal_dr,
                        project_cols=[],
                        cardinality_idx=len(results),
                    )
                    results.append(dr)
                    emitted_program += 1
                    stats.append(
                        RecordOpStats(
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
                    )
                    emitted_any = True

                self._trace(
                    "step_emit_program_records",
                    step=steps,
                    node_id=node_id,
                    emitted_records=int(emitted_program),
                )

            if not emitted_any:
                # No per-node program, or it produced no records: emit traversal record.
                results.append(traversal_dr)
                stats.append(
                    RecordOpStats(
                        record_id=traversal_dr._id,
                        record_parent_ids=traversal_dr._parent_ids,
                        record_source_indices=traversal_dr._source_indices,
                        record_state=traversal_dr.to_dict(include_bytes=False),
                        full_op_id=self.get_full_op_id(),
                        logical_op_id=self.logical_op_id,
                        op_name=self.op_name(),
                        time_per_record=0.0,
                        cost_per_record=0.0,
                        op_details={k: str(v) for k, v in self.get_id_params().items()},
                    )
                )

                self._trace(
                    "step_emit_traversal_record",
                    step=steps,
                    node_id=node_id,
                )

            # Termination check happens after (optional) admission + node program.
            state = {
                "steps": steps,
                "node_id": node_id,
                "depth": item.depth,
                "score": score,
                "path_node_ids": list(item.path_node_ids),
                "path_edge_ids": list(item.path_edge_ids),
                "frontier_size": len(pq),
                "visited_count": len(visited),
                "admitted_nodes": admitted_nodes,
                "emitted_records": len(results),
            }
            should_term, term_meta = self._termination_decision(state)
            why = (term_meta or {}).get("reason")
            if should_term:
                self._trace(
                    "step_termination",
                    step=steps,
                    node_id=node_id,
                    terminated=True,
                    termination_id=self.termination_id,
                    state=dict(state),
                    prompt=(term_meta or {}).get("prompt"),
                    raw_output=(term_meta or {}).get("raw_output"),
                    reason=(term_meta or {}).get("reason"),
                    why=why,
                    model=(term_meta or {}).get("model"),
                    latency_s=(term_meta or {}).get("latency_s"),
                    cache_hit=(term_meta or {}).get("cache_hit"),
                    input_tokens=(term_meta or {}).get("input_tokens"),
                    output_tokens=(term_meta or {}).get("output_tokens"),
                    cached_tokens=(term_meta or {}).get("cached_tokens"),
                    cost_usd=(term_meta or {}).get("cost_usd"),
                    cached_cost_usd=(term_meta or {}).get("cached_cost_usd"),
                    cached_input_tokens=(term_meta or {}).get("cached_input_tokens"),
                    cached_output_tokens=(term_meta or {}).get("cached_output_tokens"),
                    cached_cached_tokens=(term_meta or {}).get("cached_cached_tokens"),
                    decision={
                        "passed": True,
                        "kind": "termination",
                        "id": self.termination_id,
                        "model": (term_meta or {}).get("model"),
                        "reason": (term_meta or {}).get("reason"),
                        "why": why,
                        "cache_hit": (term_meta or {}).get("cache_hit"),
                        "latency_s": (term_meta or {}).get("latency_s"),
                        "tokens": {
                            "input": (term_meta or {}).get("input_tokens"),
                            "output": (term_meta or {}).get("output_tokens"),
                            "cached": (term_meta or {}).get("cached_tokens"),
                        },
                        "cost_usd": (term_meta or {}).get("cost_usd"),
                    },
                )
                self._trace(
                    "step_summary",
                    step=steps,
                    node_id=node_id,
                    admitted=True,
                    skipped=False,
                    terminated=True,
                    frontier_size_after=len(pq),
                    visited_count=len(visited),
                    admitted_nodes=admitted_nodes,
                    emitted_records=len(results),
                )
                break
            else:
                self._trace(
                    "step_termination",
                    step=steps,
                    node_id=node_id,
                    terminated=False,
                    termination_id=self.termination_id,
                    state=dict(state),
                    prompt=(term_meta or {}).get("prompt"),
                    raw_output=(term_meta or {}).get("raw_output"),
                    reason=(term_meta or {}).get("reason"),
                    why=why,
                    model=(term_meta or {}).get("model"),
                    latency_s=(term_meta or {}).get("latency_s"),
                    cache_hit=(term_meta or {}).get("cache_hit"),
                    input_tokens=(term_meta or {}).get("input_tokens"),
                    output_tokens=(term_meta or {}).get("output_tokens"),
                    cached_tokens=(term_meta or {}).get("cached_tokens"),
                    cost_usd=(term_meta or {}).get("cost_usd"),
                    cached_cost_usd=(term_meta or {}).get("cached_cost_usd"),
                    cached_input_tokens=(term_meta or {}).get("cached_input_tokens"),
                    cached_output_tokens=(term_meta or {}).get("cached_output_tokens"),
                    cached_cached_tokens=(term_meta or {}).get("cached_cached_tokens"),
                    decision={
                        "passed": False,
                        "kind": "termination",
                        "id": self.termination_id,
                        "model": (term_meta or {}).get("model"),
                        "reason": (term_meta or {}).get("reason"),
                        "why": why,
                        "cache_hit": (term_meta or {}).get("cache_hit"),
                        "latency_s": (term_meta or {}).get("latency_s"),
                        "tokens": {
                            "input": (term_meta or {}).get("input_tokens"),
                            "output": (term_meta or {}).get("output_tokens"),
                            "cached": (term_meta or {}).get("cached_tokens"),
                        },
                        "cost_usd": (term_meta or {}).get("cost_usd"),
                    },
                )

            # Expand neighbors (both directions)
            neighbor_traces: list[dict[str, Any]] = []
            enqueued = 0
            for edge in self._iter_adjacent_edges(node_id):
                neighbor_id = edge.dst if edge.src == node_id else edge.src
                direction = "out" if edge.src == node_id else "in"
                if not self.allow_revisit and neighbor_id in seen:
                    expansion_counts["already_visited"] += 1
                    neighbor_traces.append(
                        {
                            "edge_id": edge.id,
                            "edge_type": edge.type,
                            "direction": direction,
                            "neighbor_id": neighbor_id,
                            "enqueued": False,
                            "skip_reason": "already_visited",
                        }
                    )
                    continue
                if not self.graph.has_node(neighbor_id):
                    expansion_counts["missing_node"] += 1
                    neighbor_traces.append(
                        {
                            "edge_id": edge.id,
                            "edge_type": edge.type,
                            "direction": direction,
                            "neighbor_id": neighbor_id,
                            "enqueued": False,
                            "skip_reason": "missing_node",
                        }
                    )
                    continue

                neighbor = self.graph.get_node(neighbor_id)
                new_path_nodes = item.path_node_ids + [neighbor_id]
                new_path_edges = item.path_edge_ids + [edge.id]

                base_neighbor_score = self._score(neighbor_id, neighbor, edge, node_id, new_path_nodes, new_path_edges)
                accept, neighbor_score, neighbor_decision_data = self._decision_for_enqueue(
                    node_id=neighbor_id,
                    node=neighbor,
                    depth=item.depth + 1,
                    score=base_neighbor_score,
                    path_node_ids=new_path_nodes,
                    path_edge_ids=new_path_edges,
                )
                if not accept:
                    expansion_counts["program_rejected"] += 1
                    neighbor_traces.append(
                        {
                            "edge_id": edge.id,
                            "edge_type": edge.type,
                            "direction": direction,
                            "neighbor_id": neighbor_id,
                            "enqueued": False,
                            "skip_reason": "decision_program_rejected",
                            "score": float(base_neighbor_score),
                        }
                    )
                    continue

                heapq.heappush(
                    pq,
                    _PQItem(
                        neg_score=-neighbor_score,
                        node_id=neighbor_id,
                        depth=item.depth + 1,
                        path_node_ids=new_path_nodes,
                        path_edge_ids=new_path_edges,
                        decision_data=neighbor_decision_data,
                    ),
                )
                expansion_counts["enqueued"] += 1
                enqueued += 1
                if not self.allow_revisit:
                    seen.add(neighbor_id)

                neighbor_traces.append(
                    {
                        "edge_id": edge.id,
                        "edge_type": edge.type,
                        "direction": direction,
                        "neighbor_id": neighbor_id,
                        "enqueued": True,
                        "score": float(neighbor_score),
                        "depth": item.depth + 1,
                        "path_node_ids": list(new_path_nodes),
                        "path_edge_ids": list(new_path_edges),
                        "rank_meta": neighbor_decision_data.get("rank_meta") if isinstance(neighbor_decision_data, dict) else None,
                    }
                )

            self._trace(
                "step_expand",
                step=steps,
                node_id=node_id,
                expanded_edges=len(neighbor_traces),
                neighbors=neighbor_traces,
                frontier_size_after=len(pq),
                visited_count=len(visited),
            )

            self._trace(
                "step_summary",
                step=steps,
                node_id=node_id,
                admitted=True,
                skipped=False,
                terminated=False,
                expanded_edges=len(neighbor_traces),
                enqueued=int(enqueued),
                frontier_size_after=len(pq),
                visited_count=len(visited),
                admitted_nodes=admitted_nodes,
                emitted_records=len(results),
            )

            self._trace(
                "step_end",
                step=steps,
                node_id=node_id,
                skipped=False,
                frontier_size_after=len(pq),
                visited_count=len(visited),
                admitted_nodes=admitted_nodes,
                emitted_records=len(results),
            )

        self._trace(
            "traverse_summary",
            graph_id=self.graph.graph_id,
            graph_revision=self.graph.revision,
            steps=int(steps),
            admitted_nodes=int(admitted_nodes),
            visited_count=len(visited),
            emitted_records=len(results),
            frontier_size=len(pq),
            skip_counts=dict(skip_counts),
            expansion_counts=dict(expansion_counts),
            unique_nodes=len(node_visits),
            top_nodes=[{"node_id": k, "count": int(v)} for k, v in node_visits.most_common(50)],
            top_edges=[{"edge_id": k, "count": int(v)} for k, v in edge_traversals.most_common(50)],
        )

        self._trace(
            "traverse_done",
            steps=int(steps),
            admitted_nodes=int(admitted_nodes),
            emitted_records=len(results),
            frontier_size=len(pq),
            visited_count=len(visited),
        )

        return DataRecordSet(results, stats, input=candidate)

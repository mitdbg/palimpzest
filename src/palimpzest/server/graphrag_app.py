from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from palimpzest.core.data.graph_dataset import GraphDataset, GraphSnapshot
from palimpzest.graphrag.deciders import (
    LLMBooleanDecider,
    LLMBooleanDeciderConfig,
    bootstrap_admittance_criteria,
    bootstrap_termination_criteria,
    build_admittance_instruction,
    build_termination_instruction,
    render_admittance_decision_prompt,
    render_termination_decision_prompt,
)
from palimpzest.graphrag.retrieval import (
    EmbeddingModel,
    HFReranker,
    HFRerankerConfig,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingConfig,
    SentenceTransformerEmbeddingModel,
    VectorIndex,
    default_node_text,
)

DEFAULT_SNAPSHOT_PATH = Path("CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json")


class RunRequest(BaseModel):
    index: str | None = None
    query: str | None = None
    workload_file: str | None = None

    model: str | None = None
    ranking_model: str | None = None
    admittance_model: str | None = None
    termination_model: str | None = None

    entry_points: int = 5
    max_steps: int = 200
    # If set to "all"/"*"/"", traverse all edge types.
    edge_type: str = "all"

    # If true, include detailed per-step traversal trace events.
    debug_trace: bool = False
    # If true (and debug_trace is enabled), include full node text in traversal trace events.
    debug_trace_full_text: bool = False


@dataclass
class RunMeta:
    run_id: str
    index: str | None
    query: str | None
    created_at: float


class RunStore:
    def __init__(self) -> None:
        self._events_by_run_id: dict[str, list[dict[str, Any]]] = {}
        self._history: list[RunMeta] = []
        self._last_query: str = ""

    def add_run(self, *, meta: RunMeta, events: list[dict[str, Any]]) -> None:
        self._events_by_run_id[meta.run_id] = events
        self._history.append(meta)
        if meta.query:
            self._last_query = meta.query

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._events_by_run_id.get(run_id, [])

    def history_payload(self) -> dict[str, Any]:
        # UI expects {history: [...]}
        return {
            "history": [
                {
                    "run_id": m.run_id,
                    "index": m.index,
                    "query": m.query,
                    "created_at": m.created_at,
                }
                for m in reversed(self._history[-50:])
            ]
        }

    def status_payload(self) -> dict[str, Any]:
        return {"running": False, "last_query": self._last_query}


class GraphService:
    def __init__(self) -> None:
        self._graph: GraphDataset | None = None
        self._label_index: list[tuple[str, str]] = []  # (lower_label, node_id)
        self._root_node_id: str | None = None

        # Lazy-built vector index for entry-point selection.
        self._vector_index: VectorIndex | None = None
        self._vector_index_backend: str | None = None
        self._vector_index_error: str | None = None

    def load(self, snapshot_path: Path) -> None:
        snapshot = GraphSnapshot.model_validate(json.loads(snapshot_path.read_text()))
        self._graph = GraphDataset.from_snapshot(snapshot)

        label_index: list[tuple[str, str]] = []
        root_id: str | None = None

        for node in snapshot.nodes:
            label = node.label
            if not label:
                # Older snapshots sometimes stored a display name in attrs.
                maybe = (node.attrs or {}).get("name")
                if isinstance(maybe, str) and maybe.strip():
                    label = maybe.strip()
            if label:
                label_index.append((label.lower(), node.id))

            # heuristic root: CMS hierarchy root has level==3
            lvl = node.level
            if lvl is None:
                lvl = (node.attrs or {}).get("level")
            if root_id is None and lvl == 3:
                root_id = node.id

        self._label_index = label_index
        self._root_node_id = root_id

    @property
    def graph(self) -> GraphDataset:
        if self._graph is None:
            raise RuntimeError("Graph not loaded")
        return self._graph

    def _make_embedding_model(self, *, backend: str | None) -> EmbeddingModel:
        """Create an embedding backend.

        backend examples:
        - None (auto): prefer sentence-transformers, else OpenAI
        - "st" or "st:<model_name>"
        - "openai" or "openai:<model_name>"
        """

        spec = (backend or "").strip()
        if not spec:
            # auto: prefer sentence-transformers
            try:
                return SentenceTransformerEmbeddingModel(config=SentenceTransformerEmbeddingConfig())
            except Exception:
                return OpenAIEmbeddingModel(config=OpenAIEmbeddingConfig())

        if spec.startswith("st"):
            model_name = None
            if ":" in spec:
                _, model_name = spec.split(":", 1)
                model_name = model_name.strip() or None
            cfg = SentenceTransformerEmbeddingConfig(model_name=model_name or SentenceTransformerEmbeddingConfig().model_name)
            return SentenceTransformerEmbeddingModel(config=cfg)

        if spec.startswith("openai"):
            model_name = None
            if ":" in spec:
                _, model_name = spec.split(":", 1)
                model_name = model_name.strip() or None
            cfg = OpenAIEmbeddingConfig(model_name=model_name or OpenAIEmbeddingConfig().model_name)
            return OpenAIEmbeddingModel(config=cfg)

        raise ValueError(f"Unknown embedding backend spec: {backend}")

    def _ensure_vector_index(self, *, embedding_backend: str | None) -> None:
        if self._vector_index is not None and self._vector_index_backend == (embedding_backend or ""):
            return

        self._vector_index = None
        self._vector_index_backend = embedding_backend or ""
        self._vector_index_error = None

        try:
            emb = self._make_embedding_model(backend=embedding_backend)
            self._vector_index = VectorIndex(graph=self.graph, embedding_model=emb, node_text_fn=default_node_text)
        except Exception as e:
            # Keep server functional with heuristic fallback.
            self._vector_index_error = str(e)
            self._vector_index = None

    def pick_start_nodes_scored(self, *, query: str | None, k: int, embedding_backend: str | None = None) -> list[tuple[str, float]]:
        """Pick entry points for traversal.

        Prefers vector search unless the query is an explicit node-id (or comma-separated ids).
        """

        g = self.graph
        if query is None or not query.strip():
            return [(self._root_node_id, 0.0)] if self._root_node_id else []

        q = query.strip()

        # Comma-separated explicit ids. Only take this branch if at least one segment resolves.
        if "," in q:
            ids = [p.strip() for p in q.split(",") if p.strip()]
            valid = [i for i in ids if g.has_node(i)]
            if valid:
                return [(i, 0.0) for i in valid]

        # Direct id.
        if g.has_node(q):
            return [(q, 0.0)]

        # Vector search.
        self._ensure_vector_index(embedding_backend=embedding_backend)
        if self._vector_index is not None:
            hits = self._vector_index.search(query=q, k=max(1, k))
            if hits:
                return hits

        # Fallback: substring label search.
        ql = q.lower()
        matches: list[str] = []
        for label, node_id in self._label_index:
            if ql in label:
                matches.append(node_id)
                if len(matches) >= max(1, k):
                    break

        if matches:
            return [(m, 0.0) for m in matches]
        return [(self._root_node_id, 0.0)] if self._root_node_id else []

    def pick_start_nodes(self, *, query: str | None, k: int, embedding_backend: str | None = None) -> list[str]:
        return [node_id for (node_id, _score) in self.pick_start_nodes_scored(query=query, k=k, embedding_backend=embedding_backend)]

    def graph_payload(self, *, edge_type: str | None = None) -> dict[str, Any]:
        snap = self.graph.to_snapshot()

        nodes = []
        for n in snap.nodes:
            summary = n.label
            if not summary:
                md = (n.attrs or {}).get("metadata")
                if isinstance(md, dict):
                    name = md.get("name")
                    if isinstance(name, str) and name.strip():
                        summary = name.strip()
            nodes.append(
                {
                    "id": n.id,
                    "summary": (summary or ""),
                    "type": "static",
                }
            )

        links = []
        for e in snap.edges:
            if edge_type is not None and e.type != edge_type:
                continue
            links.append(
                {
                    "source": e.src,
                    "target": e.dst,
                    "type": e.type,
                }
            )

        return {"nodes": nodes, "links": links}


def build_events_for_traverse(
    *,
    run_id: str,
    query_text: str,
    visited: list[dict[str, Any]],
    graph: GraphDataset,
    edge_type: str | None,
    entry_points: list[tuple[str, float]],
    trace_events: list[dict[str, Any]] | None = None,
    max_candidates_per_update: int = 50,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seq = 0

    def _emit(event_type: str, *, data: dict[str, Any], **extra: Any) -> None:
        nonlocal seq
        seq += 1
        payload: dict[str, Any] = {
            "event_type": event_type,
            "ts_ms": int(time.time() * 1000),
            "seq": int(seq),
            # keep query_id for UI compatibility; include run_id for consistency
            "query_id": run_id,
            "run_id": run_id,
            "data": data,
            **extra,
        }
        events.append(payload)

    def _summary(node_id: str) -> str:
        node = graph.get_node(node_id)
        if node.label:
            return node.label
        md = (node.attrs or {}).get("metadata")
        if isinstance(md, dict):
            name = md.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return ""

    _emit("trace_init", data={"run_id": run_id})
    _emit("query_start", data={"query": query_text}, query_text=query_text)

    if trace_events:
        # Best-effort mapping for downstream correlation: first admitted step per node id.
        admitted_step_by_node_id: dict[str, int] = {}
        for te in trace_events:
            if te.get("event_type") == "step_end" and te.get("skipped") is False:
                node_id = te.get("node_id")
                step = te.get("step")
                if isinstance(node_id, str) and isinstance(step, int) and node_id not in admitted_step_by_node_id:
                    admitted_step_by_node_id[node_id] = step

        for te in trace_events:
            _emit(
                "traverse_trace",
                data=te,
                data_event_type=te.get("event_type"),
                step=te.get("step"),
                node_id=te.get("node_id"),
            )
    else:
        admitted_step_by_node_id = {}

    # Provide entry points to match the UI's ROOT -> candidates expectation.
    if entry_points:
        _emit(
            "frontier_update",
            data={
                "parent_id": "",
                "candidates": [{"id": node_id, "score": float(score), "summary": _summary(node_id)} for (node_id, score) in entry_points],
            },
            parent_id="",
        )

    # visited items: {node_id, path_node_ids}
    best_path: list[str] = []

    for idx, item in enumerate(visited, start=1):
        node_id = item.get("node_id")
        if not node_id:
            continue
        _emit(
            "search_step",
            data={"node_id": node_id},
            node_id=node_id,
            step=admitted_step_by_node_id.get(str(node_id)),
            step_idx=int(idx),
        )

        # Emit the outgoing frontier for queue/path reconstruction in the UI.
        try:
            candidates: list[dict[str, Any]] = []
            for _, neighbor in graph.iter_neighbors(node_id, edge_type=edge_type):
                candidates.append({"id": neighbor.id, "score": 0.0, "summary": _summary(neighbor.id)})
                if len(candidates) >= max_candidates_per_update:
                    break
            if candidates:
                _emit(
                    "frontier_update",
                    data={"parent_id": node_id, "candidates": candidates},
                    parent_id=node_id,
                    node_id=node_id,
                    step=admitted_step_by_node_id.get(str(node_id)),
                )
        except Exception:
            # Best-effort; events are for visualization only.
            pass

        path = item.get("path_node_ids")
        if isinstance(path, list) and len(path) > len(best_path):
            best_path = [str(x) for x in path]

    _emit("result", data={"answer": f"Visited {len(visited)} nodes.", "path": best_path})
    _emit("query_end", data={"visited": len(visited)})

    return events


def create_app(*, snapshot_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Palimpzest GraphRAG UI API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runs = RunStore()
    graphs = GraphService()

    load_error: str | None = None

    if snapshot_path is None:
        snapshot_path = Path(os.getenv("PZ_GRAPH_SNAPSHOT_PATH", str(DEFAULT_SNAPSHOT_PATH)))
        if not snapshot_path.is_absolute():
            snapshot_path = Path.cwd() / snapshot_path

    if not snapshot_path.exists():
        load_error = (
            "Graph snapshot not found. Set PZ_GRAPH_SNAPSHOT_PATH or run "
            "scratchpad/scripts/ingest_cms_standard.py to generate CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json"
        )
    else:
        graphs.load(snapshot_path)

    def _require_graph() -> None:
        if load_error is not None:
            raise HTTPException(status_code=503, detail=load_error)

    @app.get("/api/resources")
    def get_resources() -> dict[str, Any]:
        # Minimal shape expected by UI.
        return {"indices": ["cms_standard"], "workloads": []}

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        return runs.status_payload()

    @app.get("/api/runs")
    def get_runs() -> dict[str, Any]:
        return runs.history_payload()

    @app.get("/api/graph")
    def get_graph(index: str | None = None) -> dict[str, Any]:
        # Ignore index for now; we serve the loaded snapshot.
        _ = index
        _require_graph()
        return graphs.graph_payload()

    @app.post("/api/run")
    def run_query(req: RunRequest) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        created_at = time.time()

        query_text = req.query or ""

        _require_graph()

        start_nodes_scored = graphs.pick_start_nodes_scored(query=query_text, k=req.entry_points, embedding_backend=req.model)
        start_nodes = [n for (n, _s) in start_nodes_scored]

        def _default_llm_model_id() -> str | None:
            if os.getenv("OPENAI_API_KEY"):
                return "openai/gpt-4o-mini-2024-07-18"
            if os.getenv("ANTHROPIC_API_KEY"):
                return "anthropic/claude-3-5-sonnet-20241022"
            if os.getenv("GEMINI_API_KEY"):
                return "vertex_ai/gemini-2.0-flash"
            if os.getenv("TOGETHER_API_KEY"):
                return "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
            return None

        # Ranker (default: HF bge reranker). No embedding-based ranker.
        ranking_spec = "hf:BAAI/bge-reranker-base" if req.ranking_model is None else (req.ranking_model or "").strip()

        if ranking_spec.lower() in {"", "none", "off", "disabled"}:
            ranking_spec = ""
        ranker_fn = None
        if ranking_spec.lower() not in {"none", "off"}:
            reranker = None
            if ranking_spec.startswith("hf:"):
                hf_name = ranking_spec.split(":", 1)[1].strip() or "BAAI/bge-reranker-base"
                try:
                    allow_download = os.getenv("PZ_GRAPHRAG_HF_ALLOW_DOWNLOAD", "").strip() == "1"
                    reranker = HFReranker(
                        config=HFRerankerConfig(
                            model_name=hf_name,
                            local_files_only=not allow_download,
                            trust_remote_code=True,
                        )
                    )
                except Exception:
                    reranker = None

            else:
                # Only hf:<model> is supported right now.
                reranker = None

            if reranker is not None:
                score_cache: dict[str, float] = {}

                def _ranker(node_id, node, edge, from_node_id, path_node_ids, path_edge_ids):  # noqa: ANN001
                    if node_id in score_cache:
                        return score_cache[node_id]
                    doc = default_node_text(node)
                    s = float(reranker.score(query=query_text, docs=[doc])[0]) if doc else 0.0
                    score_cache[node_id] = s
                    return s

                ranker_fn = _ranker

        # LLM meta prompts + unified gate.
        # IMPORTANT: to avoid surprising network usage, LLM gating is only enabled when
        # admittance_model is explicitly provided.
        adm_model = (req.admittance_model or "").strip()
        gate_fn = None
        admittance_criteria: str | None = None
        termination_criteria: str | None = None
        if adm_model:
            # Bootstrap query-specific admittance criteria once.
            try:
                admittance_criteria = bootstrap_admittance_criteria(model=adm_model, query=query_text)
            except Exception:
                admittance_criteria = None
            decider = LLMBooleanDecider(config=LLMBooleanDeciderConfig(model=adm_model))
            criteria = admittance_criteria or build_admittance_instruction()
            admit_cache: dict[str, bool] = {}
            admit_meta_cache: dict[str, dict[str, Any]] = {}

            def _gate(node_id, node, depth, score, path_node_ids, path_edge_ids):  # noqa: ANN001
                if node_id in admit_cache:
                    cached = dict(admit_meta_cache.get(node_id, {}))
                    cached_cost = cached.get("cost_usd")
                    cached_in = cached.get("input_tokens")
                    cached_out = cached.get("output_tokens")
                    cached_cached = cached.get("cached_tokens")
                    cached.update(
                        {
                            "cache_hit": True,
                            "latency_s": 0.0,
                            "cost_usd": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cached_tokens": 0,
                            "cached_cost_usd": cached_cost,
                            "cached_input_tokens": cached_in,
                            "cached_output_tokens": cached_out,
                            "cached_cached_tokens": cached_cached,
                        }
                    )
                    return admit_cache[node_id], cached
                txt = default_node_text(node)
                prompt = render_admittance_decision_prompt(
                    query=query_text,
                    admittance_criteria=criteria,
                    node_id=str(node_id),
                    depth=int(depth),
                    score=float(score),
                    path_node_ids=[str(x) for x in (path_node_ids[-10:])],
                    node_text=(txt or "")[:2000],
                )
                decision, reason, raw_output, meta = decider.decide_prompt_with_meta(prompt=prompt)
                admit_cache[node_id] = bool(decision)
                meta = {
                    "model": adm_model,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "reason": reason,
                    "cache_hit": False,
                    "latency_s": float(meta.get("latency_s") or 0.0),
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                    "cached_tokens": meta.get("cached_tokens"),
                    "cost_usd": meta.get("cost_usd"),
                    "cached_cost_usd": 0.0,
                    "cached_input_tokens": 0,
                    "cached_output_tokens": 0,
                    "cached_cached_tokens": 0,
                }
                admit_meta_cache[node_id] = meta
                return admit_cache[node_id], meta

            gate_fn = _gate

        # Termination via LLM.
        # Enabled when termination_model is provided OR (by default) when admittance is enabled.
        # To explicitly disable, set termination_model to "none"/"off".
        term_model = (req.termination_model or "").strip()
        if not term_model and adm_model:
            term_model = adm_model
        if term_model.lower() in {"none", "off", "false", "0"}:
            term_model = ""
        termination_fn = None
        if term_model:
            # Bootstrap query-specific termination criteria once.
            if termination_criteria is None:
                try:
                    termination_criteria = bootstrap_termination_criteria(model=term_model, query=query_text)
                except Exception:
                    termination_criteria = None
            tdecider = LLMBooleanDecider(config=LLMBooleanDeciderConfig(model=term_model))
            tcriteria = termination_criteria or build_termination_instruction()
            # reduce cost/latency: call at most every N steps
            term_interval = 5

            term_cache: dict[str, tuple[bool, dict[str, Any]]] = {}

            def _terminate(state):  # noqa: ANN001
                steps = int(state.get("steps", 0) or 0)
                if steps % term_interval != 0:
                    return False, {
                        "model": term_model,
                        "prompt": None,
                        "raw_output": None,
                        "reason": f"interval_skip:{term_interval}",
                        "cache_hit": True,
                        "latency_s": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_tokens": 0,
                        "cost_usd": 0.0,
                        "cached_cost_usd": 0.0,
                        "cached_input_tokens": 0,
                        "cached_output_tokens": 0,
                        "cached_cached_tokens": 0,
                    }
                node_id = str(state.get("node_id", ""))
                cache_key = f"{node_id}|{steps}"
                if cache_key in term_cache:
                    decision, cached = term_cache[cache_key]
                    cached = dict(cached)
                    cached_cost = cached.get("cost_usd")
                    cached_in = cached.get("input_tokens")
                    cached_out = cached.get("output_tokens")
                    cached_cached = cached.get("cached_tokens")
                    cached.update(
                        {
                            "cache_hit": True,
                            "latency_s": 0.0,
                            "cost_usd": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cached_tokens": 0,
                            "cached_cost_usd": cached_cost,
                            "cached_input_tokens": cached_in,
                            "cached_output_tokens": cached_out,
                            "cached_cached_tokens": cached_cached,
                        }
                    )
                    return bool(decision), cached
                try:
                    node = graphs.graph.get_node(node_id)
                    txt = default_node_text(node)
                except Exception:
                    txt = ""
                prompt = render_termination_decision_prompt(
                    query=query_text,
                    termination_criteria=tcriteria,
                    state=state,
                    node_text=(txt or "")[:2000],
                )
                decision, reason, raw_output, meta = tdecider.decide_prompt_with_meta(prompt=prompt)
                out_meta = {
                    "model": term_model,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "reason": reason,
                    "cache_hit": False,
                    "latency_s": float(meta.get("latency_s") or 0.0),
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                    "cached_tokens": meta.get("cached_tokens"),
                    "cost_usd": meta.get("cost_usd"),
                    "cached_cost_usd": 0.0,
                    "cached_input_tokens": 0,
                    "cached_output_tokens": 0,
                    "cached_cached_tokens": 0,
                }

                term_cache[cache_key] = (bool(decision), out_meta)
                return bool(decision), out_meta

            termination_fn = _terminate

        # Edge type handling: default to all edges.
        edge_type = (req.edge_type or "").strip()
        if edge_type.lower() in {"", "*", "all"}:
            edge_type = None

        if not start_nodes:
            # still emit a run for UI, but it will have no steps.
            visited_items: list[dict[str, Any]] = []
            trace_events: list[dict[str, Any]] | None = None
        else:
            trace_events = [] if bool(req.debug_trace) else None

            def _tracer(ev: dict[str, Any]) -> None:
                if trace_events is not None:
                    trace_events.append(ev)

            ds = graphs.graph.traverse(
                start_node_ids=start_nodes,
                edge_type=edge_type,
                max_steps=req.max_steps,
                ranker=ranker_fn,
                ranker_id=ranking_spec if ranker_fn is not None else None,
                # LLM gate is admittance-only (avoid double invocation).
                visit_filter=None,
                visit_filter_id=None,
                admittance=gate_fn,
                admittance_id=adm_model if gate_fn is not None else None,
                termination=termination_fn,
                termination_id=term_model if termination_fn is not None else None,
                tracer=_tracer if trace_events is not None else None,
                tracer_id="graphrag_app" if trace_events is not None else None,
                trace_full_node_text=bool(req.debug_trace_full_text),
            )
            out = ds.run()
            visited_items = [
                {
                    "node_id": r.node_id,
                    "path_node_ids": r.path_node_ids,
                    "score": getattr(r, "score", 0.0),
                }
                for r in out
            ]

        events = build_events_for_traverse(
            run_id=run_id,
            query_text=query_text,
            visited=visited_items,
            graph=graphs.graph,
            edge_type=edge_type,
            entry_points=start_nodes_scored,
            trace_events=trace_events,
        )
        runs.add_run(meta=RunMeta(run_id=run_id, index=req.index, query=query_text, created_at=created_at), events=events)

        return {"run_id": run_id}

    @app.post("/api/stop")
    def stop_legacy() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/api/stop/{run_id}")
    def stop_run(run_id: str) -> dict[str, Any]:
        _ = run_id
        return {"ok": True}

    @app.websocket("/ws/{run_id}")
    async def ws_run(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        for e in runs.get_events(run_id):
            await websocket.send_text(json.dumps(e))
        await websocket.close()

    return app


app = create_app()

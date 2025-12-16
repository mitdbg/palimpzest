from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from palimpzest.core.data.graph_dataset import GraphDataset, GraphSnapshot
from palimpzest.graphrag.deciders import (
    CMS_COMP_OPS_SYSTEM_PROMPT,
    LLMBooleanDecider,
    LLMBooleanDeciderConfig,
    LLMFilterExtractor,
    LLMTextGenerator,
    LLMTextGeneratorConfig,
    FilterCondition,
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

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("graphrag_app")

DEFAULT_SNAPSHOT_PATH = Path("CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json")


class RunRequest(BaseModel):
    index: str | None = None
    query: str | None = None
    workload_file: str | None = None

    model: str | None = None
    ranking_model: str | None = None
    admittance_model: str | None = None
    termination_model: str | None = None
    synthesis_model: str | None = None  # Model for final answer synthesis (defaults to admittance_model)

    entry_points: int = 5
    max_steps: int = 200
    # If set to "all"/"*"/"", traverse all edge types.
    edge_type: str = "all"

    # If true, include detailed per-step traversal trace events.
    debug_trace: bool = False
    # If true (and debug_trace is enabled), include full node text in traversal trace events.
    debug_trace_full_text: bool = False
    # If true, expand neighbors of nodes that fail the hard pruning (visit filter).
    expand_filtered_nodes: bool = False

    # Explicit filters provided by the user
    filters: list[FilterCondition] | None = None

    # Custom admittance instructions to override the default/bootstrapped ones
    admittance_instructions: str | None = None

    # Maximum number of evidence nodes to include in synthesis context
    synthesis_max_nodes: int | None = 15


@dataclass
class RunMeta:
    run_id: str
    index: str | None
    query: str | None
    created_at: float


@dataclass
class ActiveRun:
    """Tracks an in-progress run with a queue for streaming events."""
    meta: RunMeta
    loop: asyncio.AbstractEventLoop  # Captured at creation time for thread-safe access
    event_queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    events: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    error: str | None = None


def _default_runs_dir() -> Path:
    """Get the default directory for persisting run history."""
    # Check environment variable first
    env_dir = os.getenv("PZ_GRAPHRAG_RUNS_DIR")
    if env_dir:
        return Path(env_dir)
    # Default to ~/.palimpzest/runs/
    return Path.home() / ".palimpzest" / "runs"


class RunStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self._persist_dir = persist_dir or _default_runs_dir()
        self._events_by_run_id: dict[str, list[dict[str, Any]]] = {}
        self._history: list[RunMeta] = []
        self._last_query: str = ""
        self._active_runs: dict[str, ActiveRun] = {}
        self._running_run_id: str | None = None
        
        # Load persisted history on startup
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted run history from disk."""
        if not self._persist_dir.exists():
            logger.info(f"No persisted runs found at {self._persist_dir}")
            return
        
        history_file = self._persist_dir / "history.json"
        if not history_file.exists():
            return
        
        try:
            with open(history_file) as f:
                history_data = json.load(f)
            
            loaded_count = 0
            for entry in history_data:
                run_id = entry.get("run_id")
                if not run_id:
                    continue
                
                meta = RunMeta(
                    run_id=run_id,
                    index=entry.get("index"),
                    query=entry.get("query"),
                    created_at=entry.get("created_at", 0.0),
                )
                self._history.append(meta)
                
                # Load events if file exists
                events_file = self._persist_dir / f"{run_id}.json"
                if events_file.exists():
                    try:
                        with open(events_file) as ef:
                            self._events_by_run_id[run_id] = json.load(ef)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load events for run {run_id}: {e}")
            
            if self._history:
                self._last_query = self._history[-1].query or ""
            
            logger.info(f"Loaded {loaded_count} runs from {self._persist_dir}")
        except Exception as e:
            logger.error(f"Failed to load run history: {e}")

    def _save_to_disk(self, run_id: str) -> None:
        """Persist a completed run to disk."""
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Save history index
            history_file = self._persist_dir / "history.json"
            history_data = [
                {
                    "run_id": m.run_id,
                    "index": m.index,
                    "query": m.query,
                    "created_at": m.created_at,
                }
                for m in self._history
            ]
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)
            
            # Save events for this run
            events = self._events_by_run_id.get(run_id, [])
            if events:
                events_file = self._persist_dir / f"{run_id}.json"
                with open(events_file, "w") as f:
                    json.dump(events, f)
            
            logger.info(f"Persisted run {run_id} to {self._persist_dir}")
        except Exception as e:
            logger.error(f"Failed to persist run {run_id}: {e}")

    def start_run(self, *, meta: RunMeta) -> ActiveRun:
        """Start a new run and return the ActiveRun for event streaming."""
        # Capture the current event loop for thread-safe access from background tasks
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        run = ActiveRun(meta=meta, loop=loop)
        self._active_runs[meta.run_id] = run
        self._running_run_id = meta.run_id
        if meta.query:
            self._last_query = meta.query
        return run

    def push_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Push an event to the active run's queue (thread-safe via loop)."""
        run = self._active_runs.get(run_id)
        logger.debug(f"push_event: run_id={run_id}, run_exists={run is not None}, event_type={event.get('event_type')}")
        if run and not run.completed:
            run.events.append(event)
            logger.debug(f"push_event: appended event #{len(run.events)}, pushing to queue")
            # Use call_soon_threadsafe since traversal runs in a thread
            try:
                run.loop.call_soon_threadsafe(run.event_queue.put_nowait, event)
                logger.debug(f"push_event: successfully queued event")
            except RuntimeError as e:
                logger.warning(f"Failed to push event to queue: {e}")

    def complete_run(self, run_id: str, *, error: str | None = None) -> None:
        """Mark a run as complete and archive events."""
        run = self._active_runs.get(run_id)
        if run:
            run.completed = True
            run.error = error
            # Signal end of stream
            try:
                run.loop.call_soon_threadsafe(run.event_queue.put_nowait, None)
            except RuntimeError as e:
                logger.warning(f"Failed to signal run completion: {e}")
            # Archive to history
            self._events_by_run_id[run_id] = run.events
            self._history.append(run.meta)
            if self._running_run_id == run_id:
                self._running_run_id = None
            
            # Persist to disk
            self._save_to_disk(run_id)

    def cleanup_active_run(self, run_id: str) -> None:
        """Remove an active run from tracking (call after WebSocket closes)."""
        self._active_runs.pop(run_id, None)

    def get_active_run(self, run_id: str) -> ActiveRun | None:
        return self._active_runs.get(run_id)

    def add_run(self, *, meta: RunMeta, events: list[dict[str, Any]]) -> None:
        self._events_by_run_id[meta.run_id] = events
        self._history.append(meta)
        if meta.query:
            self._last_query = meta.query

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._events_by_run_id.get(run_id, [])

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from history and disk."""
        # Remove from memory
        self._events_by_run_id.pop(run_id, None)
        self._history = [m for m in self._history if m.run_id != run_id]
        
        # Remove from disk
        try:
            events_file = self._persist_dir / f"{run_id}.json"
            if events_file.exists():
                events_file.unlink()
            
            # Update history file
            history_file = self._persist_dir / "history.json"
            if history_file.exists():
                history_data = [
                    {
                        "run_id": m.run_id,
                        "index": m.index,
                        "query": m.query,
                        "created_at": m.created_at,
                    }
                    for m in self._history
                ]
                with open(history_file, "w") as f:
                    json.dump(history_data, f, indent=2)
            
            logger.info(f"Deleted run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {e}")
            return False

    def clear_history(self) -> int:
        """Delete all runs from history and disk. Returns count of deleted runs."""
        count = len(self._history)
        run_ids = [m.run_id for m in self._history]
        
        self._events_by_run_id.clear()
        self._history.clear()
        
        # Remove from disk
        try:
            for run_id in run_ids:
                events_file = self._persist_dir / f"{run_id}.json"
                if events_file.exists():
                    events_file.unlink()
            
            history_file = self._persist_dir / "history.json"
            if history_file.exists():
                history_file.unlink()
            
            logger.info(f"Cleared {count} runs from history")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
        
        return count

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
        return {
            "running": self._running_run_id is not None,
            "run_id": self._running_run_id,
            "last_query": self._last_query
        }


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
        logger.info(f"Loading graph snapshot from: {snapshot_path}")
        start_time = time.time()
        try:
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
            
            duration = time.time() - start_time
            logger.info(f"Graph loaded successfully. Nodes: {len(snapshot.nodes)}, Edges: {len(snapshot.edges)}. Took {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load graph snapshot: {e}", exc_info=True)
            raise

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
        # DEBUG: Print first few node types
        for i, n in enumerate(snap.nodes[:5]):
            logger.info(f"DEBUG: Node {n.id} type={n.type} attrs={n.attrs.keys() if n.attrs else 'None'}")

        for n in snap.nodes:
            summary = n.label
            if not summary:
                md = (n.attrs or {}).get("metadata")
                if isinstance(md, dict):
                    name = md.get("name")
                    if isinstance(name, str) and name.strip():
                        summary = name.strip()
            
            # Robust type resolution
            node_type = n.type
            if not node_type and n.attrs:
                node_type = n.attrs.get("type") or n.attrs.get("layer")
            
            nodes.append(
                {
                    "id": n.id,
                    "summary": (summary or ""),
                    "type": node_type or "static",
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
    admit_meta_cache: dict[str, dict[str, Any]] | None = None,
    on_event: callable | None = None,
) -> list[dict[str, Any]]:
    """Build traversal events.
    
    If on_event is provided, each event is passed to it for real-time streaming.
    """
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
        if on_event:
            on_event(payload)

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

        # Collect any decision metadata produced by the decision program / gate.
        decision_meta: dict[str, Any] | None = None
        if isinstance(item.get("decision"), dict):
            decision_meta = item.get("decision")
        elif admit_meta_cache and node_id in admit_meta_cache:
            decision_meta = admit_meta_cache.get(node_id)

        admit = bool(item.get("admit"))
        score = item.get("rank_score")
        if score is None:
            score = item.get("score")
        score_f = float(score) if isinstance(score, (int, float)) else 0.0

        summary = _summary(str(node_id))
        reason = (decision_meta or {}).get("reason")
        model = (decision_meta or {}).get("model")
        cache_hit = (decision_meta or {}).get("cache_hit")
        latency_s = (decision_meta or {}).get("latency_s")
        cost_usd = (decision_meta or {}).get("cost_usd")
        raw_output = (decision_meta or {}).get("raw_output")
        raw_output_s = raw_output if isinstance(raw_output, str) else None
        if raw_output_s and len(raw_output_s) > 500:
            raw_output_s = raw_output_s[:500] + "\n…(truncated)…"

        # 1) Existing event type used for graph playback.
        _emit(
            "search_step",
            data={
                "node_id": str(node_id),
                "summary": summary,
                "admit": admit,
                "decision": "admit" if admit else "reject",
                "reason": reason,
                "score": score_f,
                "model": model,
                "cache_hit": cache_hit,
                "stats": {"cost_usd": cost_usd, "latency_s": latency_s},
                "raw_output": raw_output_s,
            },
            node_id=node_id,
            step=admitted_step_by_node_id.get(str(node_id)),
            step_idx=int(idx),
        )

        # 2) Structured evaluation event (the UI uses this to build Evidence + show reasoning).
        _emit(
            "node_evaluation",
            data={
                "node_id": str(node_id),
                "score": score_f,
                "is_relevant": admit,
                "reasoning": reason,
                "metadata": {
                    "summary": summary,
                    "model": model,
                    "cache_hit": cache_hit,
                    "latency_s": latency_s,
                    "cost_usd": cost_usd,
                },
            },
            node_id=node_id,
            step=admitted_step_by_node_id.get(str(node_id)),
            step_idx=int(idx),
        )

        # 3) Evidence event (optional, but makes the UI's Evidence feed more concrete).
        if admit:
            _emit(
                "evidence_collected",
                data={
                    "node_id": str(node_id),
                    "score": score_f,
                    "content": summary,
                    "reasoning": reason,
                    "model": model,
                },
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

    def _allowed_graph_roots() -> list[Path]:
        # Keep graph selection safe: only allow loading from known roots.
        cwd = Path.cwd()
        return [cwd / "CURRENT_WORKSTREAM/exports", cwd / "data"]

    def _resolve_graph_index(index: str) -> Path:
        if not isinstance(index, str) or not index.strip():
            raise HTTPException(status_code=400, detail="Index filename required")

        # Back-compat: historically the UI passed just a filename from CURRENT_WORKSTREAM/exports.
        candidate = Path(index)
        if candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Invalid index path")

        # Reject traversal or strange paths early.
        if "\x00" in index:
            raise HTTPException(status_code=400, detail="Invalid index path")

        cwd = Path.cwd()
        exports_root = cwd / "CURRENT_WORKSTREAM/exports"

        resolved = (exports_root / candidate).resolve() if len(candidate.parts) == 1 else (cwd / candidate).resolve()

        allowed_roots = [p.resolve() for p in _allowed_graph_roots()]
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=400, detail="Invalid index path")

        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Graph snapshot not found")

        return resolved

    @app.get("/api/resources")
    def get_resources() -> dict[str, Any]:
        cwd = Path.cwd()
        exports_dir = cwd / "CURRENT_WORKSTREAM/exports"
        data_dir = cwd / "data"

        indices: list[str] = []

        # Snapshots in CURRENT_WORKSTREAM/exports
        if exports_dir.exists():
            for f in sorted(exports_dir.glob("*.json")):
                indices.append(str(Path("CURRENT_WORKSTREAM/exports") / f.name))

        # Large graphs commonly live in data/*.json (or nested). Only include obvious graph files.
        if data_dir.exists():
            for f in sorted(data_dir.glob("**/*graph*.json")):
                # Return as workspace-relative paths.
                indices.append(str(f.relative_to(cwd)))

        # Also include the default if it exists elsewhere (workspace-relative).
        if DEFAULT_SNAPSHOT_PATH.exists():
            try:
                rel = str(DEFAULT_SNAPSHOT_PATH.resolve().relative_to(cwd.resolve()))
            except Exception:
                rel = str(DEFAULT_SNAPSHOT_PATH.name)
            if rel not in indices:
                indices.append(rel)

        return {"indices": indices, "workloads": []}

    @app.post("/api/load_graph")
    def load_graph(req: dict[str, str]) -> dict[str, Any]:
        index = req.get("index")
        logger.info("API Request: load_graph index=%s", index)

        path = _resolve_graph_index(index or "")

        try:
            graphs.load(path)
            nonlocal load_error
            load_error = None
            return {"ok": True, "nodes": graphs.graph.store.count_nodes()}
        except Exception as e:
            load_error = str(e)
            logger.error(f"load_graph exception: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load graph: {e}") from e

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        return runs.status_payload()

    @app.get("/api/runs")
    def get_runs() -> dict[str, Any]:
        return runs.history_payload()

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, Any]:
        success = runs.delete_run(run_id)
        return {"ok": success, "run_id": run_id}

    @app.delete("/api/runs")
    def clear_runs() -> dict[str, Any]:
        count = runs.clear_history()
        return {"ok": True, "deleted": count}

    @app.get("/api/graph")
    def get_graph(index: str | None = None) -> dict[str, Any]:
        # Ignore index for now; we serve the loaded snapshot.
        _ = index
        _require_graph()
        return graphs.graph_payload()

    @app.post("/api/run")
    async def run_query(req: RunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        logger.info(f"API Request: run_query query='{req.query}' model={req.model}")
        run_id = uuid.uuid4().hex
        created_at = time.time()

        query_text = req.query or ""

        _require_graph()

        # Start the run (creates async queue for streaming)
        runs.start_run(meta=RunMeta(run_id=run_id, index=req.index, query=query_text, created_at=created_at))

        # Schedule the traversal in a background task
        background_tasks.add_task(
            _run_traversal_background,
            run_id=run_id,
            query_text=query_text,
            req=req,
            graphs=graphs,
            runs=runs,
        )

        return {"run_id": run_id}

    def _run_traversal_background(
        *,
        run_id: str,
        query_text: str,
        req: RunRequest,
        graphs: GraphService,
        runs: RunStore,
    ) -> None:
        """Run traversal in background thread, streaming events via runs.push_event()."""
        logger.info(f"Background traversal started for run {run_id}")
        logger.info(f"Active runs: {list(runs._active_runs.keys())}")

        def _emit_event(event: dict[str, Any]) -> None:
            logger.info(f"_emit_event called: {event.get('event_type')}")
            runs.push_event(run_id, event)

        try:
            _run_traversal_inner(
                run_id=run_id,
                query_text=query_text,
                req=req,
                graphs=graphs,
                runs=runs,
                emit_event=_emit_event,
            )
        except Exception as e:
            logger.error(f"Background traversal error for run {run_id}: {e}", exc_info=True)
            # Emit error event before completing
            error_event = {
                "event_type": "error",
                "ts_ms": int(time.time() * 1000),
                "seq": 999999,
                "run_id": run_id,
                "data": {"error": str(e)}
            }
            _emit_event(error_event)
            runs.complete_run(run_id, error=str(e))
        else:
            runs.complete_run(run_id)
        logger.info(f"Background traversal finished for run {run_id}")

    def _run_traversal_inner(
        *,
        run_id: str,
        query_text: str,
        req: RunRequest,
        graphs: GraphService,
        runs: RunStore,
        emit_event: callable,
    ) -> None:
        """Inner traversal logic that may raise exceptions."""

        def _log_prompt(*, kind: str, node_id: str | None, step: int | None, depth: int | None, prompt: str | None) -> None:
            # Only log prompts when explicitly requested; prompts can be large.
            if not bool(req.debug_trace):
                return
            if not isinstance(prompt, str) or not prompt.strip():
                return
            preview = prompt.strip()
            if len(preview) > 1500:
                preview = preview[:1500] + "\n…(truncated)…"
            logger.info(
                "prompt kind=%s run_id=%s step=%s depth=%s node_id=%s\n%s",
                kind,
                run_id,
                step,
                depth,
                node_id,
                preview,
            )

        try:
            start_nodes_scored = graphs.pick_start_nodes_scored(query=query_text, k=req.entry_points, embedding_backend=req.model)
            start_nodes = [n for (n, _s) in start_nodes_scored]
            logger.info(f"Selected {len(start_nodes)} start nodes for query: {start_nodes}")
        except Exception as e:
            logger.error(f"Error picking start nodes: {e}", exc_info=True)
            start_nodes = []
            start_nodes_scored = []

        def _default_llm_model_id() -> str | None:
            if os.getenv("OPENROUTER_API_KEY"):
                return "openrouter/x-ai/grok-4.1-fast"
            if os.getenv("OPENAI_API_KEY"):
                return "openai/gpt-4o-mini-2024-07-18"
            if os.getenv("ANTHROPIC_API_KEY"):
                return "anthropic/claude-3-5-sonnet-20241022"
            if os.getenv("GEMINI_API_KEY"):
                return "vertex_ai/gemini-2.0-flash"
            if os.getenv("TOGETHER_API_KEY"):
                return "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
            return None

        # Extract filters from query
        filters: list[FilterCondition] = []
        # Use admittance model for extraction if available, else default
        ext_model = (req.admittance_model or "").strip() or (_default_llm_model_id() or "")
        if ext_model and ext_model.lower() not in {"none", "off"}:
             try:
                 extractor = LLMFilterExtractor(config=LLMBooleanDeciderConfig(model=ext_model))
                 filters = extractor.extract(query_text)
                 if filters:
                     logger.info(f"Extracted filters: {filters}")
             except Exception as e:
                 logger.warning(f"Failed to extract filters: {e}")

        # Add user-provided filters
        if req.filters:
            filters.extend(req.filters)
            logger.info(f"Added user filters. Total filters: {filters}")

        visit_filter_fn = None
        if filters:
            def _visit_filter(node_id, node, depth, score, path_node_ids, path_edge_ids):
                attrs = node.attrs or {}
                for f in filters:
                    if f.field not in attrs:
                        continue
                    
                    val = attrs[f.field]
                    target = f.value
                    
                    # Try to convert to comparable types (float/int) if possible
                    try:
                        if isinstance(val, str) and isinstance(target, (int, float)):
                            val = float(val)
                        if isinstance(val, (int, float)) and isinstance(target, str):
                            target = float(target)
                    except ValueError:
                        pass

                    try:
                        if f.operator == "==":
                            if val != target: return False
                        elif f.operator == "!=":
                            if val == target: return False
                        elif f.operator == ">":
                            if not (val > target): return False
                        elif f.operator == "<":
                            if not (val < target): return False
                        elif f.operator == ">=":
                            if not (val >= target): return False
                        elif f.operator == "<=":
                            if not (val <= target): return False
                        elif f.operator == "contains":
                            if str(target).lower() not in str(val).lower(): return False
                    except TypeError:
                        pass
                return True
            visit_filter_fn = _visit_filter

        # Ranker (default: HF bge reranker). No embedding-based ranker.
        ranking_spec = "hf:BAAI/bge-reranker-base" if req.ranking_model is None else (req.ranking_model or "").strip()

        if ranking_spec.lower() in {"", "none", "off", "disabled"}:
            ranking_spec = ""
        ranker_fn = None
        reranker = None
        if ranking_spec.lower() not in {"none", "off"}:
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

        # Metrics collection
        run_metrics = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
        }

        def _update_metrics(meta: dict[str, Any]) -> None:
            run_metrics["cost_usd"] += float(meta.get("cost_usd") or 0.0)
            run_metrics["input_tokens"] += int(meta.get("input_tokens") or 0)
            run_metrics["output_tokens"] += int(meta.get("output_tokens") or 0)
            run_metrics["cached_tokens"] += int(meta.get("cached_tokens") or 0)
            if not meta.get("cache_hit"):
                run_metrics["calls"] += 1

        # LLM meta prompts + unified gate.
        # Default behavior: if no model is provided, use an available provider (preferring OpenRouter)
        # to keep the "just run it" path working out of the box.
        adm_model = (req.admittance_model or "").strip() or (_default_llm_model_id() or "")
        gate_fn = None
        admittance_criteria: str | None = None
        termination_criteria: str | None = None
        # Always define caches; traversal output may include `admit` even when admittance is disabled.
        admit_cache: dict[str, bool] = {}
        admit_meta_cache: dict[str, dict[str, Any]] = {}
        if adm_model:
            # Bootstrap query-specific admittance criteria once.
            if req.admittance_instructions and req.admittance_instructions.strip():
                admittance_criteria = req.admittance_instructions.strip()
            else:
                try:
                    admittance_criteria = bootstrap_admittance_criteria(model=adm_model, query=query_text)
                except Exception:
                    admittance_criteria = None
            decider = LLMBooleanDecider(config=LLMBooleanDeciderConfig(model=adm_model))
            criteria = admittance_criteria or build_admittance_instruction()

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
                _update_metrics(meta)

                # Emit detailed admittance event for UI (Node Inspector + Latency Feedback)
                emit_event({
                    "event_type": "step_gate_admittance",
                    "ts_ms": int(time.time() * 1000),
                    "run_id": run_id,
                    "data": {
                        "node_id": str(node_id),
                        "admit": bool(decision),
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "reason": reason,
                        "cost_usd": meta.get("cost_usd"),
                        "tokens": {
                            "input": meta.get("input_tokens"),
                            "output": meta.get("output_tokens")
                        },
                        "cache_hit": False
                    }
                })

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
                _log_prompt(kind="termination", node_id=node_id, step=int(state.get("steps") or 0), depth=None, prompt=prompt)
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
                _update_metrics(out_meta)
                return bool(decision), out_meta

            termination_fn = _terminate

        # Decision program (Map-first): compute rank_score + admit decision via PZ `.map`.
        # NOTE: this runs inside TraverseOp per visited node (not as a global plan).
        class _DecisionSeed(BaseModel):
            node_id: str
            node_text: str | None = None
            query: str
            depth: int
            score: float
            path_node_ids: list[str]

        class _DecisionOut(BaseModel):
            rank_score: float | None = None
            rank_meta: dict[str, Any] | None = None
            admit: bool | None = None
            decision: dict[str, Any] | None = None

        decision_program = None
        decision_program_id = None
        if reranker is not None:
            score_cache2: dict[str, float] = {}

            def _decision_program(*, node_id, node, graph, depth, score, path_node_ids, path_edge_ids):  # noqa: ANN001
                _ = graph
                _ = path_edge_ids

                from palimpzest.core.data.iter_dataset import MemoryDataset

                doc = default_node_text(node)
                seed = MemoryDataset(
                    id=f"graphrag-decision-seed-{run_id}-{node_id}",
                    vals=[
                        {
                            "node_id": str(node_id),
                            "node_text": doc,
                            "query": query_text,
                            "depth": int(depth),
                            "score": float(score),
                            "path_node_ids": list(path_node_ids),
                        }
                    ],
                    schema=_DecisionSeed,
                )

                def score_udf(row: dict[str, Any]) -> dict[str, Any]:
                    nid = str(row.get("node_id"))
                    if reranker is None:
                        return {"rank_score": float(row.get("score") or 0.0), "rank_meta": None}
                    if nid in score_cache2:
                        return {"rank_score": float(score_cache2[nid]), "rank_meta": {"cache_hit": True}}
                    txt = row.get("node_text") or ""
                    t0 = time.time()
                    s = float(reranker.score(query=query_text, docs=[txt])[0]) if txt else 0.0
                    dt = time.time() - t0
                    score_cache2[nid] = s
                    return {"rank_score": s, "rank_meta": {"model": ranking_spec, "latency_s": dt, "cache_hit": False}}

                ds = seed.map(
                    udf=score_udf,
                    cols=[
                        {"name": "rank_score", "type": float, "description": "Rerank score", "default": None},
                        {"name": "rank_meta", "type": dict[str, Any] | None, "description": "Rerank metadata", "default": None},
                    ],
                    depends_on=["node_id", "node_text"],
                )
                return ds

            decision_program = _decision_program
            decision_program_id = f"graphrag_map_decisions:{ranking_spec}"

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
                logger.info(f"_tracer called: event_type={ev.get('event_type')}, node_id={ev.get('node_id')}")
                if trace_events is not None:
                    trace_events.append(ev)
                
                # Emit live event for UI feedback
                # Wrap the traverse operator event into the standard format expected by the UI
                # The traverse operator emits: {"event_type": "step_expand", "node_id": "x", ...}
                # The UI expects: {"event_type": "step_expand", "data": {"node_id": "x", ...}}
                event_type = ev.get("event_type")
                live_ev = {
                    "event_type": event_type,
                    "ts_ms": ev.get("ts_ms") or int(time.time() * 1000),
                    "run_id": ev.get("run_id") or run_id,
                    "data": {k: v for k, v in ev.items() if k not in ("event_type", "ts_ms", "run_id", "seq")},
                }
                logger.info(f"_tracer emitting: {event_type}")
                emit_event(live_ev)

            logger.info(f"Starting traverse with {len(start_nodes)} start nodes, max_steps={req.max_steps}")
            ds = graphs.graph.traverse(
                start_node_ids=start_nodes,
                edge_type=edge_type,
                max_steps=req.max_steps,
                # Rank + admit handled via decision_program (PZ Map/Filter), not callbacks.
                ranker=None if decision_program is not None else ranker_fn,
                ranker_id=None if decision_program is not None else (ranking_spec if ranker_fn is not None else None),
                visit_filter=visit_filter_fn,
                visit_filter_id="llm_extracted_filters" if visit_filter_fn else None,
                admittance=gate_fn,
                admittance_id=adm_model if gate_fn is not None else None,
                termination=termination_fn,
                termination_id=term_model if termination_fn is not None else None,
                decision_program=decision_program,
                decision_program_id=decision_program_id,
                decision_program_output_schema=_DecisionOut if decision_program is not None else None,
                decision_program_config=None,
                trace_run_id=run_id,
                tracer=_tracer,  # ALWAYS pass tracer for live UI events
                tracer_id="graphrag_app",
                trace_full_node_text=bool(req.debug_trace_full_text),
                expand_filtered_nodes=bool(req.expand_filtered_nodes),
            )
            out = ds.run()
            visited_items = []
            for r in out:
                item = {
                    "node_id": r.node_id,
                    "path_node_ids": r.path_node_ids,
                    "score": getattr(r, "score", 0.0),
                    "rank_score": getattr(r, "rank_score", None),
                    "rank_meta": getattr(r, "rank_meta", None),
                    "admit": getattr(r, "admit", False),
                    "decision": getattr(r, "decision", None),
                }
                visited_items.append(item)
                
                # Re-populate caches from results for downstream usage
                if item["node_id"]:
                    if item["admit"]:
                        admit_cache[item["node_id"]] = True
                    if item["decision"]:
                        admit_meta_cache[item["node_id"]] = item["decision"]

        events = build_events_for_traverse(
            run_id=run_id,
            query_text=query_text,
            visited=visited_items,
            graph=graphs.graph,
            edge_type=edge_type,
            entry_points=start_nodes_scored,
            trace_events=trace_events,
            admit_meta_cache=admit_meta_cache if adm_model else None,
            on_event=emit_event,
        )

        # Generate final answer using LLMTextGenerator with CMS system prompt
        final_answer = f"Visited {len(visited_items)} nodes."
        synthesis_model = (req.synthesis_model or "").strip() or adm_model
        if synthesis_model and visited_items:
            try:
                # Collect evidence from admitted nodes
                evidence_texts = []
                for item in visited_items:
                    nid = item.get("node_id")
                    if nid and admit_cache.get(nid):
                        node = graphs.graph.get_node(nid)
                        txt = default_node_text(node)
                        if txt:
                            # Include node label if available - use full text up to 8k chars
                            label = node.label if node else nid
                            evidence_texts.append(f"[{label}]\n{txt[:8000]}")
                
                if evidence_texts:
                    # Build synthesis prompt
                    max_evidence_nodes = req.synthesis_max_nodes or 15
                    context = "\n\n---\n\n".join(evidence_texts[:max_evidence_nodes])
                    user_prompt = f"""Query: {query_text}

Retrieved Evidence ({len(evidence_texts)} nodes):

{context}

Based on the evidence above, please answer the query. Cite specific evidence nodes when possible."""

                    # Use LLMTextGenerator for proper text generation
                    generator = LLMTextGenerator(
                        config=LLMTextGeneratorConfig(
                            model=synthesis_model,
                            temperature=0.0,
                            max_tokens=1500,
                            timeout_s=120.0,
                        )
                    )
                    
                    answer_text, gen_meta = generator.generate_with_meta(
                        system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT,
                        prompt=user_prompt,
                    )
                    
                    if answer_text:
                        final_answer = answer_text
                        # Add synthesis cost to metrics
                        if gen_meta:
                            run_metrics["cost_usd"] += float(gen_meta.get("cost_usd") or 0.0)
                            run_metrics["input_tokens"] += int(gen_meta.get("input_tokens") or 0)
                            run_metrics["output_tokens"] += int(gen_meta.get("output_tokens") or 0)
                            run_metrics["calls"] += 1
                else:
                    final_answer = "No evidence nodes were admitted during traversal. Try adjusting the query or admittance criteria."
            except Exception as e:
                logger.error(f"Error generating answer: {e}", exc_info=True)
                final_answer += f" (Error generating detailed answer: {e})"

        # Emit metrics event
        metrics_event = {
            "event_type": "run_metrics",
            "ts_ms": int(time.time() * 1000),
            "seq": 999999,
            "query_id": run_id,  # For UI query matching
            "run_id": run_id,
            "data": run_metrics
        }
        emit_event(metrics_event)

        # Update result event with generated answer (emit an answer_update event)
        if final_answer:
            answer_event = {
                "event_type": "answer_update",
                "ts_ms": int(time.time() * 1000),
                "seq": 999998,
                "query_id": run_id,  # For UI query matching
                "run_id": run_id,
                "data": {"answer": final_answer}
            }
            emit_event(answer_event)

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
        logger.info(f"WebSocket connected: {run_id}")
        try:
            # Check if this is an active (streaming) run
            active_run = runs.get_active_run(run_id)
            logger.info(f"WebSocket: active_run={active_run is not None}, active_runs={list(runs._active_runs.keys())}")
            if active_run is not None:
                # Stream events in real-time from the queue
                logger.info(f"Streaming live events for run {run_id}, events so far: {len(active_run.events)}")
                
                # First, send any events that were emitted before we connected (catch-up)
                # This handles the race where events arrive between POST return and WS connect
                catch_up_events = list(active_run.events)  # Snapshot current events
                seen_seqs = set()
                for event in catch_up_events:
                    await websocket.send_text(json.dumps(event))
                    if "seq" in event:
                        seen_seqs.add(event["seq"])
                if catch_up_events:
                    logger.info(f"Sent {len(catch_up_events)} catch-up events for run {run_id}")
                
                # If already completed during catch-up, we're done
                if active_run.completed:
                    logger.info(f"Run {run_id} already completed during catch-up")
                else:
                    # Now stream new events as they arrive
                    while True:
                        event = await active_run.event_queue.get()
                        if event is None:
                            # None signals end of stream
                            break
                        # Skip events we already sent during catch-up (by seq number)
                        if "seq" in event and event["seq"] in seen_seqs:
                            continue
                        await websocket.send_text(json.dumps(event))
            else:
                # Replay historical events (run already completed)
                logger.info(f"Replaying historical events for run {run_id}")
                for e in runs.get_events(run_id):
                    await websocket.send_text(json.dumps(e))
        except Exception as e:
            logger.error(f"WebSocket error for {run_id}: {e}")
        finally:
            logger.info(f"WebSocket closed: {run_id}")
            # Clean up active run tracking if completed
            active_run = runs.get_active_run(run_id)
            if active_run and active_run.completed:
                runs.cleanup_active_run(run_id)
            await websocket.close()

    return app


app = create_app()

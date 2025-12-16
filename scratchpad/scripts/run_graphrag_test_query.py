from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

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
from palimpzest.query.processor.config import QueryProcessorConfig

DEFAULT_SNAPSHOT_PATH = Path("CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json")
DEFAULT_QUERY = "What are the most common operational challenges across all three groups (Data management, Production and Reprocessing, Tier0)?"


def _make_embedding_model(*, backend: str | None) -> EmbeddingModel:
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


def pick_start_nodes_scored(
    *,
    graph: GraphDataset,
    query: str | None,
    k: int,
    embedding_backend: str | None,
) -> list[tuple[str, float]]:
    if query is None or not query.strip():
        return []

    q = query.strip()

    # Comma-separated explicit ids. Only take this branch if at least one segment resolves.
    if "," in q:
        ids = [p.strip() for p in q.split(",") if p.strip()]
        valid = [i for i in ids if graph.has_node(i)]
        if valid:
            return [(i, 0.0) for i in valid]

    # Direct id.
    if graph.has_node(q):
        return [(q, 0.0)]

    # Vector search.
    try:
        emb = _make_embedding_model(backend=embedding_backend)
        idx = VectorIndex(graph=graph, embedding_model=emb, node_text_fn=default_node_text)
        hits = idx.search(query=q, k=max(1, k))
        if hits:
            return hits
    except Exception:
        pass

    # Fallback: substring label search.
    ql = q.lower()
    matches: list[str] = []
    for node in graph.to_snapshot().nodes:
        label = (node.label or "").strip()
        if label and ql in label.lower():
            matches.append(node.id)
            if len(matches) >= max(1, k):
                break
    return [(m, 0.0) for m in matches]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a GraphRAG traversal offline and write a traversal trace JSONL.")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    ap.add_argument("--query", type=str, default=DEFAULT_QUERY)
    ap.add_argument("--entry-points", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--edge-type", type=str, default="all", help='Edge type filter, or "all"')
    ap.add_argument("--include-overlay", action="store_true", default=False)
    ap.add_argument("--embedding-backend", type=str, default="", help='""|st[:model]|openai[:model]')

    ap.add_argument("--ranking-model", type=str, default="hf:BAAI/bge-reranker-base", help='"none" or "hf:<model>"')
    ap.add_argument("--admittance-model", type=str, default="", help="LiteLLM model id; empty disables admittance")
    ap.add_argument("--termination-model", type=str, default="", help="LiteLLM model id; empty disables termination")

    ap.add_argument("--debug-trace-full-text", action="store_true", default=False)
    ap.add_argument("--progress", action="store_true", default=False)
    ap.add_argument("--out-dir", type=Path, default=Path("CURRENT_WORKSTREAM/exports"))
    args = ap.parse_args()

    snapshot_path: Path = args.snapshot
    if not snapshot_path.exists() and snapshot_path == DEFAULT_SNAPSHOT_PATH:
        # Workstreams are archived; try to locate the most recent snapshot.
        candidates = sorted(Path("docs/workstreams").glob("*/CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json"))
        if candidates:
            snapshot_path = candidates[-1]

    if not snapshot_path.exists():
        raise SystemExit(
            f"Snapshot not found: {args.snapshot}\n"
            "Pass --snapshot <path> or run `./.venv/bin/python scratchpad/scripts/ingest_cms_standard.py` "
            "to generate `CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json`."
        )

    run_id = uuid.uuid4().hex
    created_at = time.time()
    query_text = (args.query or "").strip()
    if not query_text:
        raise SystemExit("Missing --query")

    snapshot = GraphSnapshot.model_validate(json.loads(snapshot_path.read_text()))
    graph = GraphDataset.from_snapshot(snapshot)

    edge_type = (args.edge_type or "").strip()
    if edge_type.lower() in {"", "*", "all"}:
        edge_type = None

    start_nodes_scored = pick_start_nodes_scored(
        graph=graph,
        query=query_text,
        k=max(1, int(args.entry_points)),
        embedding_backend=(args.embedding_backend or "").strip() or None,
    )
    start_nodes = [n for (n, _s) in start_nodes_scored]
    if not start_nodes:
        raise SystemExit("No start nodes found; try --embedding-backend or a node id query")

    # Ranker (HF reranker).
    ranking_spec = (args.ranking_model or "").strip()
    if ranking_spec.lower() in {"", "none", "off", "disabled"}:
        ranking_spec = ""
    reranker = None
    if ranking_spec.startswith("hf:"):
        hf_name = ranking_spec.split(":", 1)[1].strip() or "BAAI/bge-reranker-base"
        allow_download = os.getenv("PZ_GRAPHRAG_HF_ALLOW_DOWNLOAD", "").strip() == "1"
        try:
            reranker = HFReranker(
                config=HFRerankerConfig(
                    model_name=hf_name,
                    local_files_only=not allow_download,
                    trust_remote_code=True,
                )
            )
        except Exception:
            reranker = None

    # Admittance (optional; explicitly enabled only when provided).
    adm_model = (args.admittance_model or "").strip()
    admit_cache: dict[str, bool] = {}
    admit_meta_cache: dict[str, dict[str, Any]] = {}
    criteria = None
    decider = None
    if adm_model:
        try:
            admittance_criteria = bootstrap_admittance_criteria(model=adm_model, query=query_text)
        except Exception:
            admittance_criteria = None
        criteria = admittance_criteria or build_admittance_instruction()
        decider = LLMBooleanDecider(config=LLMBooleanDeciderConfig(model=adm_model))

    # Termination (optional).
    term_model = (args.termination_model or "").strip()
    term_cache: dict[str, tuple[bool, dict[str, Any]]] = {}
    tcriteria = None
    tdecider = None
    if term_model:
        try:
            termination_criteria = bootstrap_termination_criteria(model=term_model, query=query_text)
        except Exception:
            termination_criteria = None
        tcriteria = termination_criteria or build_termination_instruction()
        tdecider = LLMBooleanDecider(config=LLMBooleanDeciderConfig(model=term_model))

    class _DecisionOutSchema:
        # This is only used for unioning optional fields in GraphDataset.traverse; it must be a pydantic model.
        pass

    from pydantic import BaseModel

    class DecisionOut(BaseModel):
        rank_score: float | None = None
        rank_meta: dict[str, Any] | None = None
        admit: bool | None = None
        decision: dict[str, Any] | None = None

    score_cache: dict[str, float] = {}

    def decision_program(*, node_id, node, graph, depth, score, path_node_ids, path_edge_ids):  # noqa: ANN001
        _ = graph
        _ = path_edge_ids

        from palimpzest.core.data.iter_dataset import MemoryDataset

        seed = MemoryDataset(
            id=f"graphrag-decision-seed-{run_id}-{node_id}",
            vals=[
                {
                    "node_id": str(node_id),
                    "node_text": default_node_text(node),
                    "query": query_text,
                    "depth": int(depth),
                    "score": float(score),
                    "path_node_ids": list(path_node_ids),
                }
            ],
            schema=[
                {"name": "node_id", "type": str, "description": "Node id"},
                {"name": "node_text", "type": str, "description": "Node text"},
                {"name": "query", "type": str, "description": "Query"},
                {"name": "depth", "type": int, "description": "Depth"},
                {"name": "score", "type": float, "description": "Base score"},
                {"name": "path_node_ids", "type": list[str], "description": "Path node ids"},
            ],
        )

        def score_udf(row: dict[str, Any]) -> dict[str, Any]:
            nid = str(row.get("node_id"))
            if reranker is None:
                return {"rank_score": float(row.get("score") or 0.0), "rank_meta": None}
            if nid in score_cache:
                return {"rank_score": float(score_cache[nid]), "rank_meta": {"cache_hit": True, "model": ranking_spec}}
            txt = row.get("node_text") or ""
            t0 = time.time()
            s = float(reranker.score(query=query_text, docs=[txt])[0]) if txt else 0.0
            dt = time.time() - t0
            score_cache[nid] = s
            return {"rank_score": s, "rank_meta": {"model": ranking_spec, "latency_s": dt, "cache_hit": False}}

        def admit_udf(row: dict[str, Any]) -> dict[str, Any]:
            if not adm_model or decider is None or criteria is None:
                return {"admit": True, "decision": None}
            nid = str(row.get("node_id"))
            if nid in admit_cache:
                return {"admit": bool(admit_cache[nid]), "decision": dict(admit_meta_cache.get(nid, {}))}

            prompt = render_admittance_decision_prompt(
                query=query_text,
                admittance_criteria=criteria,
                node_id=nid,
                depth=int(row.get("depth") or 0),
                score=float(row.get("rank_score") or 0.0),
                path_node_ids=list(row.get("path_node_ids") or []),
                node_text=(row.get("node_text") or "")[:2000],
            )
            decision, reason, raw_output, meta = decider.decide_prompt_with_meta(prompt=prompt)
            out_meta = {
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
            }
            admit_cache[nid] = bool(decision)
            admit_meta_cache[nid] = out_meta
            return {"admit": bool(decision), "decision": out_meta}

        ds = seed.map(
            udf=score_udf,
            cols=[
                {"name": "rank_score", "type": float, "description": "Rerank score", "default": None},
                {"name": "rank_meta", "type": dict[str, Any] | None, "description": "Rerank metadata", "default": None},
            ],
            depends_on=["node_id", "node_text"],
        ).map(
            udf=admit_udf,
            cols=[
                {"name": "admit", "type": bool, "description": "Admit", "default": None},
                {"name": "decision", "type": dict[str, Any] | None, "description": "Decision metadata", "default": None},
            ],
            depends_on=["node_id", "rank_score", "node_text"],
        )
        if adm_model:
            ds = ds.filter(lambda r: bool((r or {}).get("admit", False)), depends_on=["admit"])
        return ds

    def termination_fn(state: dict) -> tuple[bool, dict[str, Any] | None]:
        if not term_model or tdecider is None or tcriteria is None:
            return False, None
        cache_key = json.dumps({"query": query_text, "state": state.get("steps")}, sort_keys=True)
        if cache_key in term_cache:
            decision, out_meta = term_cache[cache_key]
            cached = dict(out_meta)
            cached.update({"cache_hit": True, "latency_s": 0.0, "cost_usd": 0.0})
            return bool(decision), cached
        txt = ""
        try:
            nid = state.get("node_id")
            if isinstance(nid, str) and graph.has_node(nid):
                txt = default_node_text(graph.get_node(nid))
        except Exception:
            txt = ""
        prompt = render_termination_decision_prompt(query=query_text, termination_criteria=tcriteria, state=state, node_text=(txt or "")[:2000])
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

    trace_events: list[dict[str, Any]] = []

    def tracer(ev: dict[str, Any]) -> None:
        trace_events.append(ev)

    ds = graph.traverse(
        start_node_ids=start_nodes,
        edge_type=edge_type,
        include_overlay=bool(args.include_overlay),
        max_steps=int(args.max_steps),
        ranker=None,
        ranker_id=None,
        visit_filter=None,
        visit_filter_id=None,
        admittance=None,
        admittance_id=None,
        termination=termination_fn if term_model else None,
        termination_id=term_model if term_model else None,
        decision_program=decision_program,
        decision_program_id=f"offline_decision_program:{ranking_spec}:{adm_model or 'no_adm'}",
        decision_program_output_schema=DecisionOut,
        decision_program_config=QueryProcessorConfig(progress=bool(args.progress), execution_strategy="sequential", max_workers=1),
        tracer=tracer,
        tracer_id="offline_script",
        trace_run_id=run_id,
        trace_full_node_text=bool(args.debug_trace_full_text),
    )

    out = ds.run(config=QueryProcessorConfig(progress=bool(args.progress)))
    visited = [
        {
            "node_id": r.node_id,
            "depth": int(getattr(r, "depth", 0)),
            "score": float(getattr(r, "score", 0.0) or 0.0),
            "path_node_ids": list(getattr(r, "path_node_ids", []) or []),
            "path_edge_ids": list(getattr(r, "path_edge_ids", []) or []),
        }
        for r in out
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.out_dir / f"traverse_trace_{run_id}.jsonl"
    visited_path = args.out_dir / f"traverse_visited_{run_id}.json"
    summary_path = args.out_dir / f"traverse_summary_{run_id}.json"

    trace_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in trace_events) + "\n")
    visited_path.write_text(json.dumps(visited, indent=2, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": created_at,
                "snapshot": str(snapshot_path),
                "query": query_text,
                "edge_type": edge_type,
                "start_nodes_scored": start_nodes_scored,
                "visited_count": len(visited),
                "trace_events": len(trace_events),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(f"run_id={run_id}")
    print(f"trace={trace_path}")
    print(f"visited={visited_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

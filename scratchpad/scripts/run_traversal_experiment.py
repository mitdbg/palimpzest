from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.graphrag.deciders import (
    LLMBooleanDecider,
    LLMBooleanDeciderConfig,
    LLMTextGenerator,
    LLMTextGeneratorConfig,
    CMS_COMP_OPS_SYSTEM_PROMPT,
    bootstrap_admittance_criteria,
    bootstrap_synthesis_criteria,
    bootstrap_termination_criteria,
    build_admittance_instruction,
    build_termination_instruction,
    render_admittance_decision_prompt,
    render_synthesis_prompt,
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


class DatasetConfig(BaseModel):
    snapshot: Path = Field(description="Path to a GraphSnapshot JSON")


class SeedConfig(BaseModel):
    entry_points: int = Field(default=5, ge=1)
    embedding_backend: str | None = Field(
        default=None,
        description='""|st[:model]|openai[:model] (used for vector search when query is not a node id)',
    )
    start_nodes: list[str] | None = Field(
        default=None,
        description="Optional explicit start node ids (overrides vector search / lookup)",
    )
    seed_query: str | None = Field(
        default=None,
        description="Optional query to use only for picking start nodes (defaults to experiment.query)",
    )


class TraversalConfig(BaseModel):
    edge_type: str | None = Field(default=None, description='Edge type filter; null/"all" means all edges')
    include_overlay: bool = True
    max_steps: int = Field(default=200, ge=1)
    allow_revisit: bool = False


class RankingConfig(BaseModel):
    model: str | None = Field(default="hf:BAAI/bge-reranker-base", description='"none" or "hf:<model>"')


class AdmittanceConfig(BaseModel):
    model: str | None = Field(default=None, description="LiteLLM model id; required (admittance always on)")


class TerminationConfig(BaseModel):
    model: str | None = Field(
        default=None,
        description='LiteLLM model id; null/empty disables; "auto" means reuse admittance model',
    )


class TraceConfig(BaseModel):
    out_dir: Path = Field(default=Path("CURRENT_WORKSTREAM/exports"))
    full_text: bool = False


class SynthesisConfig(BaseModel):
    model: str | None = Field(
        default=None,
        description='LiteLLM model id; null disables; "auto" means reuse admittance model',
    )
    max_nodes: int = Field(default=8, ge=1, le=50)
    max_chars_per_node: int = Field(default=1600, ge=100, le=20000)
    bootstrap: bool = Field(default=True, description="Whether to bootstrap synthesis criteria once per run")


class ExperimentConfig(BaseModel):
    name: str | None = None
    run_id: str | None = None

    dataset: DatasetConfig
    query: str

    seed: SeedConfig = Field(default_factory=SeedConfig)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    admittance: AdmittanceConfig = Field(default_factory=AdmittanceConfig)
    termination: TerminationConfig = Field(default_factory=TerminationConfig)

    trace: TraceConfig = Field(default_factory=TraceConfig)
    progress: bool = False

    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)


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


def _load_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML root (expected mapping): {path}")
    return ExperimentConfig.model_validate(payload)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a GraphRAG traversal offline using a YAML config (query + dataset + traversal config).\n\n"
            "Example:\n"
            "  ./.venv/bin/python scratchpad/scripts/run_traversal_experiment.py \\\n"
            "    --config scratchpad/scripts/traversal_experiment.example.yaml\n"
        )
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="Override trace.out_dir from YAML")
    ap.add_argument("--run-id", type=str, default=None, help="Override run_id from YAML")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    run_id = (args.run_id or cfg.run_id or uuid.uuid4().hex).strip()
    created_at = time.time()

    query_text = (cfg.query or "").strip()
    if not query_text:
        raise SystemExit("Missing config.query")

    snapshot_path = cfg.dataset.snapshot
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot not found: {snapshot_path}")

    graph = GraphDataset.load(snapshot_path)

    # Start nodes.
    if cfg.seed.start_nodes:
        start_nodes = [n for n in cfg.seed.start_nodes if graph.has_node(n)]
        if not start_nodes:
            raise SystemExit("seed.start_nodes provided but none exist in graph")
        start_nodes_scored = [(n, 0.0) for n in start_nodes]
    else:
        seed_query = (cfg.seed.seed_query or query_text).strip()
        start_nodes_scored = pick_start_nodes_scored(
            graph=graph,
            query=seed_query,
            k=int(cfg.seed.entry_points),
            embedding_backend=(cfg.seed.embedding_backend or "").strip() or None,
        )
        start_nodes = [n for (n, _s) in start_nodes_scored]
        if not start_nodes:
            raise SystemExit(
                "No start nodes found. Provide seed.start_nodes, set seed.embedding_backend, "
                "or use a node id as seed.seed_query."
            )

    # Normalize edge_type.
    edge_type = (cfg.traversal.edge_type or "").strip() or None
    if edge_type and edge_type.lower() in {"*", "all"}:
        edge_type = None

    # Reranker.
    ranking_spec = (cfg.ranking.model or "").strip()
    if ranking_spec.lower() in {"", "none", "off", "disabled"}:
        raise SystemExit("ranking.model is required (reranking always on)")
    reranker = None
    if ranking_spec.startswith("hf:"):
        hf_name = ranking_spec.split(":", 1)[1].strip() or "BAAI/bge-reranker-base"
        # Do not surprise-download unless explicitly allowed.
        allow_download = False
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

    if reranker is None:
        raise SystemExit(
            f"Failed to initialize reranker for ranking.model={ranking_spec!r}. "
            "Install/cache the HF model locally or update ranking.model."
        )

    # Synthesis model (optional).
    synth_model = (cfg.synthesis.model or "").strip()
    if synth_model.lower() in {"", "none", "off", "false", "0", "disabled"}:
        synth_model = ""
    if synth_model.lower() == "auto":
        synth_model = (cfg.admittance.model or "").strip()
    if not synth_model:
        # default: reuse admittance model if provided
        synth_model = (cfg.admittance.model or "").strip() or ""

    # Admittance.
    # If omitted, default to an available provider model (preferring OpenRouter).
    adm_model = (cfg.admittance.model or "").strip()
    admit_cache: dict[str, bool] = {}
    admit_meta_cache: dict[str, dict[str, Any]] = {}
    criteria = None
    decider = None
    # Instantiate config first so it can fill in a default model.
    decider_cfg = LLMBooleanDeciderConfig(model=adm_model or None)
    adm_model = decider_cfg.model

    if adm_model:
        try:
            admittance_criteria = bootstrap_admittance_criteria(model=adm_model, query=query_text)
        except Exception:
            admittance_criteria = None
        criteria = admittance_criteria or build_admittance_instruction()
        decider = LLMBooleanDecider(config=decider_cfg)

    # Termination.
    term_model = (cfg.termination.model or "").strip()
    if term_model.lower() == "auto":
        term_model = adm_model
    if term_model.lower() in {"", "none", "off", "disabled"}:
        term_model = ""
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

    # Decision program (batch): run once per frontier batch to avoid per-node subplans.
    class DecisionOut(BaseModel):
        rank_score: float | None = None
        rank_meta: dict[str, Any] | None = None
        # NOTE: Admittance is evaluated on the *visited* node via TraverseOp.admittance.
        # The batch program is used only for neighbor ranking.

    class _DecisionRow(BaseModel):
        node_id: str
        rank_score: float | None = None
        rank_meta: dict[str, Any] | None = None

    score_cache: dict[str, float] = {}

    def decision_program_batch(*, candidates: list[dict[str, Any]], graph):  # noqa: ANN001
        _ = graph

        # IMPORTANT: return a DataRecordCollection directly.
        # If we return a Dataset here, TraverseOp will call Dataset.run() during traversal,
        # which creates nested sub-plans.
        from palimpzest.core.elements.records import DataRecord, DataRecordCollection

        node_ids: list[str] = []
        texts: list[str] = []
        for c in candidates:
            nid = str(c.get("node_id"))
            node = c.get("node")
            node_ids.append(nid)
            texts.append(default_node_text(node) if node is not None else "")

        scores_by_id: dict[str, float] = {}
        rank_meta_by_id: dict[str, dict[str, Any]] = {}

        uncached_ids: list[str] = []
        uncached_texts: list[str] = []
        for nid, txt in zip(node_ids, texts, strict=True):
            if nid in score_cache:
                scores_by_id[nid] = float(score_cache[nid])
                rank_meta_by_id[nid] = {"cache_hit": True, "model": ranking_spec}
            else:
                uncached_ids.append(nid)
                uncached_texts.append(txt)

        if uncached_ids:
            t0 = time.time()
            batch_scores = reranker.score(query=query_text, docs=uncached_texts)
            dt = time.time() - t0
            for nid, s in zip(uncached_ids, batch_scores, strict=True):
                score_cache[nid] = float(s)
                scores_by_id[nid] = float(s)
                rank_meta_by_id[nid] = {"cache_hit": False, "model": ranking_spec, "latency_s": dt}

        data_records: list[DataRecord] = []
        batch_id = uuid.uuid4().hex
        for nid in node_ids:
            data_records.append(
                DataRecord(
                    data_item=_DecisionRow(
                        node_id=nid,
                        rank_score=float(scores_by_id.get(nid, 0.0)),
                        rank_meta=dict(rank_meta_by_id.get(nid, {})),
                    ),
                    source_indices=f"decision-batch-{run_id}-{batch_id}",
                )
            )

        return DataRecordCollection(data_records)

    decision_program_batch_id = f"yaml_experiment_batch:{ranking_spec}"
    decision_program_batch_output_schema = DecisionOut
    decision_program_batch_config = QueryProcessorConfig(
        progress=False,
        execution_strategy="sequential",
        max_workers=1,
    )

    def admittance_fn(
        node_id: str,
        node,
        depth: int,
        score: float,
        path_node_ids: list[str],
        path_edge_ids: list[str],
    ) -> tuple[bool, dict[str, Any]]:
        _ = path_edge_ids

        nid = str(node_id)
        if nid in admit_cache:
            meta = dict(admit_meta_cache.get(nid, {}))
            meta.update({"cache_hit": True, "latency_s": 0.0, "cost_usd": 0.0})
            return bool(admit_cache[nid]), meta

        prompt = render_admittance_decision_prompt(
            query=query_text,
            admittance_criteria=criteria,
            node_id=nid,
            depth=int(depth),
            score=float(score),
            path_node_ids=list(path_node_ids),
            node_text=(default_node_text(node) or "")[:2000],
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
        return bool(decision), out_meta

    def termination_fn(state: dict) -> tuple[bool, dict[str, Any] | None]:
        if not term_model or tdecider is None or tcriteria is None:
            return False, None
        cache_key = json.dumps({"query": query_text, "steps": state.get("steps")}, sort_keys=True)
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
        }
        term_cache[cache_key] = (bool(decision), out_meta)
        return bool(decision), out_meta

    trace_events: list[dict[str, Any]] = []

    def print_trace_event(ev: dict[str, Any]) -> None:
        etype = ev.get("event_type")
        if etype == "step_begin":
            nid = ev.get("node_id")
            step = ev.get("step")
            score = ev.get("popped", {}).get("score", 0.0)
            print(f"\n[Step {step}] Visiting {nid} (Score: {score:.4f})")
        elif etype == "step_gate_admittance":
            passed = ev.get("admitted")
            dec = ev.get("decision", {})
            reason = dec.get("reason") or dec.get("why") or "No reason provided"
            status = "ADMITTED" if passed else "REJECTED"
            print(f"  Admittance: {status} | {reason[:100]}{'...' if len(reason)>100 else ''}")
        elif etype == "step_expand":
            n_count = len(ev.get("neighbors", []))
            enqueued = sum(1 for n in ev.get("neighbors", []) if n.get("enqueued"))
            print(f"  Expansion: {n_count} neighbors found, {enqueued} enqueued")
        elif etype == "step_termination":
            if ev.get("terminated"):
                dec = ev.get("decision", {})
                reason = dec.get("reason") or "No reason"
                print(f"  TERMINATED: {reason}")

    def tracer(ev: dict[str, Any]) -> None:
        trace_events.append(ev)
        print_trace_event(ev)

    ds = graph.traverse(
        start_node_ids=start_nodes,
        edge_type=edge_type,
        include_overlay=bool(cfg.traversal.include_overlay),
        max_steps=int(cfg.traversal.max_steps),
        allow_revisit=bool(cfg.traversal.allow_revisit),
        ranker=None,
        ranker_id=None,
        visit_filter=None,
        visit_filter_id=None,
        admittance=admittance_fn,
        admittance_id=adm_model,
        termination=termination_fn if term_model else None,
        termination_id=term_model if term_model else None,
        decision_program=None,
        decision_program_id=None,
        decision_program_output_schema=None,
        decision_program_config=None,
        decision_program_batch=decision_program_batch,
        decision_program_batch_id=decision_program_batch_id,
        decision_program_batch_output_schema=decision_program_batch_output_schema,
        decision_program_batch_config=decision_program_batch_config,
        tracer=tracer,
        tracer_id="yaml_experiment",
        trace_run_id=run_id,
        trace_full_node_text=bool(cfg.trace.full_text),
    )

    out = ds.run(config=QueryProcessorConfig(progress=bool(cfg.progress)))
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

    out_dir = args.out_dir or cfg.trace.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = out_dir / f"traverse_trace_{run_id}.jsonl"
    visited_path = out_dir / f"traverse_visited_{run_id}.json"
    summary_path = out_dir / f"traverse_summary_{run_id}.json"
    config_copy_path = out_dir / f"traverse_config_{run_id}.yaml"
    answer_path = out_dir / f"traverse_answer_{run_id}.json"

    trace_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in trace_events) + "\n")
    visited_path.write_text(json.dumps(visited, indent=2, ensure_ascii=False) + "\n")

    answer_obj: dict[str, Any] | None = None
    if synth_model:
        # Build context from emitted (admitted) records only.
        node_rows: list[dict[str, Any]] = []
        for r in out:
            try:
                node_id = str(getattr(r, "node_id", "") or "")
            except Exception:
                node_id = ""
            if not node_id or not graph.has_node(node_id):
                continue
            try:
                score = float(getattr(r, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            node = graph.get_node(node_id)
            txt = default_node_text(node) or ""
            node_rows.append(
                {
                    "node_id": node_id,
                    "score": score,
                    "type": getattr(node, "type", None),
                    "attrs": getattr(node, "attrs", None),
                    "text": txt,
                }
            )

        # De-dupe by node_id, keep best score.
        best: dict[str, dict[str, Any]] = {}
        for row in node_rows:
            nid = row["node_id"]
            prev = best.get(nid)
            if prev is None or float(row.get("score") or 0.0) > float(prev.get("score") or 0.0):
                best[nid] = row
        picked = sorted(best.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)[: int(cfg.synthesis.max_nodes)]

        blocks: list[str] = []
        used_nodes: list[dict[str, Any]] = []
        for i, row in enumerate(picked, start=1):
            attrs = row.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            path = attrs.get("path")
            url = attrs.get("url")
            header = f"[doc {i}] node_id={row['node_id']}"
            if path:
                header += f" path={path}"
            if url:
                header += f" url={url}"
            header += f" type={row.get('type') or ''}"

            text = str(row.get("text") or "")
            text = text[: int(cfg.synthesis.max_chars_per_node)]
            blocks.append(header + "\n" + text)
            used_nodes.append(
                {
                    "node_id": row["node_id"],
                    "score": float(row.get("score") or 0.0),
                    "path": path,
                    "url": url,
                    "type": row.get("type"),
                }
            )

        context_block = "\n\n".join(blocks)
        if cfg.synthesis.bootstrap:
            try:
                synth_criteria = bootstrap_synthesis_criteria(
                    model=synth_model,
                    query=query_text,
                    system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT,
                )
            except Exception:
                synth_criteria = "Use only the provided documents. Be clear and concise. If insufficient, say what is missing."
        else:
            synth_criteria = "Use only the provided documents. Be clear and concise. If insufficient, say what is missing."

        synth_prompt = render_synthesis_prompt(
            query=query_text,
            synthesis_criteria=synth_criteria,
            context_block=context_block,
            system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT,
        )

        gen = LLMTextGenerator(config=LLMTextGeneratorConfig(model=synth_model, temperature=0.0, max_tokens=700, timeout_s=90.0))
        answer_text, answer_meta = gen.generate_with_meta(system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT, prompt=synth_prompt)

        answer_obj = {
            "run_id": run_id,
            "query": query_text,
            "model": synth_model,
            "answer": answer_text,
            "used_nodes": used_nodes,
            "meta": answer_meta,
        }
        answer_path.write_text(json.dumps(answer_obj, indent=2, ensure_ascii=False) + "\n")
    # Pull final counters from trace to avoid conflating emitted records with visited nodes.
    trace_summary = next((e for e in reversed(trace_events) if e.get("event_type") == "traverse_summary"), None)
    trace_done = next((e for e in reversed(trace_events) if e.get("event_type") == "traverse_done"), None)
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": created_at,
                "name": cfg.name,
                "snapshot": str(snapshot_path),
                "query": query_text,
                "seed_query": (cfg.seed.seed_query or query_text),
                "edge_type": edge_type,
                "start_nodes_scored": start_nodes_scored,
                "visited_count": len(visited),
                "emitted_count": len(visited),
                "trace_steps": int((trace_done or trace_summary or {}).get("steps") or 0),
                "trace_visited_count": int((trace_done or trace_summary or {}).get("visited_count") or 0),
                "trace_admitted_nodes": int((trace_done or trace_summary or {}).get("admitted_nodes") or 0),
                "trace_frontier_size": int((trace_done or trace_summary or {}).get("frontier_size") or 0),
                "trace_events": len(trace_events),
                "ranking_model": ranking_spec or None,
                "admittance_model": adm_model or None,
                "termination_model": term_model or None,
                "synthesis_model": synth_model or None,
                "answer_path": str(answer_path) if (answer_obj is not None) else None,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    config_copy_path.write_text(Path(args.config).read_text())

    print(f"run_id={run_id}")
    print(f"trace={trace_path}")
    print(f"visited={visited_path}")
    print(f"summary={summary_path}")
    print(f"config_copy={config_copy_path}")
    if answer_obj is not None:
        print(f"answer={answer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.query.processor.config import QueryProcessorConfig


def _count_edges(graph: GraphDataset) -> int:
    store = graph.store
    if hasattr(store, "_edges_by_id"):
        return len(store._edges_by_id)  # type: ignore[attr-defined]
    return sum(1 for _ in store.iter_edges())


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = None
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: whitespace tokenization.
        return len(text.split())


def _percentiles(values: list[float], ps: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    values_sorted = sorted(values)
    out: dict[str, float] = {}
    for p in ps:
        k = (p / 100.0) * (len(values_sorted) - 1)
        lo = int(k)
        hi = min(lo + 1, len(values_sorted) - 1)
        frac = k - lo
        out[f"p{int(p)}"] = values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac
    return out


@dataclass(frozen=True)
class ChunkerRunSpec:
    name: str
    chunker_kind: str
    chunk_size: int
    chunk_overlap: int
    chunker_params: dict[str, Any] | None = None


def _summarize_lengths(texts: list[str]) -> dict[str, Any]:
    char_lens = [len(t) for t in texts]
    tok_lens = [_count_tokens(t) for t in texts]

    def _safe_mean(xs: list[int]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _safe_max(xs: list[int]) -> int:
        return max(xs) if xs else 0

    return {
        "count": len(texts),
        "chars": {
            "mean": _safe_mean(char_lens),
            "max": _safe_max(char_lens),
            **_percentiles([float(x) for x in char_lens], [50, 90, 99]),
        },
        "tokens": {
            "mean": _safe_mean(tok_lens),
            "max": _safe_max(tok_lens),
            **_percentiles([float(x) for x in tok_lens], [50, 90, 99]),
        },
    }


def run_one(*, graph_path: Path, spec: ChunkerRunSpec, progress: bool) -> dict[str, Any]:
    t_load0 = time.perf_counter()
    graph = GraphDataset.load(graph_path)
    load_s = time.perf_counter() - t_load0

    nodes_before = len(graph)
    edges_before = _count_edges(graph)

    config = QueryProcessorConfig(verbose=False, progress=progress)

    tracemalloc.start()
    t0 = time.perf_counter()
    chunked = graph.chunk(
        input_col="text",
        output_col="text",
        chunk_size=spec.chunk_size,
        chunk_overlap=spec.chunk_overlap,
        chunker_kind=spec.chunker_kind,
        chunker_params=spec.chunker_params,
        graph=graph,
        edge_policy="has_and_next",
        has_chunk_edge_type="overlay:has_chunk",
        next_chunk_edge_type="overlay:next_chunk",
        chunk_node_type="chunk",
        overwrite_nodes=False,
        overwrite_edges=False,
    )
    out = chunked.run(config)
    records = list(out)
    t_run_s = time.perf_counter() - t0
    _cur, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    nodes_after = len(graph)
    edges_after = _count_edges(graph)

    overlay_edge_types = Counter()
    for e in graph.store.iter_edges():
        if isinstance(e.type, str) and e.type.startswith("overlay:"):
            overlay_edge_types[e.type] += 1

    chunk_texts: list[str] = []
    chunks_by_source: dict[str, int] = defaultdict(int)
    for r in records:
        t = getattr(r, "text", None)
        if isinstance(t, str):
            chunk_texts.append(t)
        sid = getattr(r, "source_node_id", None)
        if isinstance(sid, str):
            chunks_by_source[sid] += 1

    expected_next_edges = sum(max(0, n - 1) for n in chunks_by_source.values())

    return {
        "spec": asdict(spec),
        "timing_s": {
            "load": load_s,
            "run": t_run_s,
            "total": load_s + t_run_s,
            "chunks_per_sec": (len(records) / t_run_s) if t_run_s > 0 else 0.0,
        },
        "memory": {"tracemalloc_peak_bytes": peak_bytes},
        "graph": {
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "nodes_added": nodes_after - nodes_before,
            "edges_before": edges_before,
            "edges_after": edges_after,
            "edges_added": edges_after - edges_before,
            "overlay_edge_type_counts": dict(overlay_edge_types),
        },
        "chunks": {
            "records": len(records),
            "unique_sources": len(chunks_by_source),
            "expected_next_chunk_edges": expected_next_edges,
            "text_stats": _summarize_lengths(chunk_texts),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=Path("data/cms_knowledge_graph.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    # Approximation: ~4 chars/token for English-ish text.
    approx_chars_per_token = 4
    target_tokens = 1000

    specs = [
        ChunkerRunSpec(
            name="recursive",
            chunker_kind="recursive_character",
            chunk_size=target_tokens * approx_chars_per_token,
            chunk_overlap=200 * approx_chars_per_token,
        ),
        ChunkerRunSpec(
            name="character",
            chunker_kind="character",
            chunk_size=target_tokens * approx_chars_per_token,
            chunk_overlap=200 * approx_chars_per_token,
        ),
        ChunkerRunSpec(
            name="token",
            chunker_kind="token",
            chunk_size=target_tokens,
            chunk_overlap=200,
        ),
        ChunkerRunSpec(
            name="markdown",
            chunker_kind="markdown",
            chunk_size=target_tokens * approx_chars_per_token,
            chunk_overlap=200 * approx_chars_per_token,
        ),
    ]

    results: dict[str, Any] = {
        "graph_path": str(args.graph),
        "runs": [],
        "notes": {
            "target_tokens": target_tokens,
            "approx_chars_per_token": approx_chars_per_token,
            "edge_policy": "has_and_next",
            "edge_types": ["overlay:has_chunk", "overlay:next_chunk"],
        },
    }

    for spec in specs:
        print(f"\n=== Running {spec.name} ({spec.chunker_kind}) ===")
        res = run_one(graph_path=args.graph, spec=spec, progress=args.progress)
        results["runs"].append(res)
        print(
            "chunks={chunks} sources={sources} run_s={run_s:.2f} chunks/s={cps:.1f} "
            "p50_tokens={p50:.0f} p90_tokens={p90:.0f} nodes+={nadd} edges+={eadd}".format(
                chunks=res["chunks"]["records"],
                sources=res["chunks"]["unique_sources"],
                run_s=res["timing_s"]["run"],
                cps=res["timing_s"]["chunks_per_sec"],
                p50=res["chunks"]["text_stats"]["tokens"]["p50"],
                p90=res["chunks"]["text_stats"]["tokens"]["p90"],
                nadd=res["graph"]["nodes_added"],
                eadd=res["graph"]["edges_added"],
            )
        )
        print("overlay edges:", res["graph"]["overlay_edge_type_counts"])

    out_path = args.out
    if out_path is None:
        out_path = Path("CURRENT_WORKSTREAM/exports/chunking_benchmark.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()

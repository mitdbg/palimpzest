"""Check whether GraphRAG retrieval results look sensible.

This script is intentionally heuristic: it inspects the trace to find
- the returned best path from the `result` event
- the highest-scoring popped nodes (from step_begin)
- basic keyword coverage for the query's group terms

Usage:
  .venv/bin/python scratchpad/scripts/check_retrieval_quality.py \
    --snapshot data/cms_refined_graph.json \
    --trace CURRENT_WORKSTREAM/exports/graphrag_trace_<run_id>.jsonl

Optional:
  --top-k 25
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphSnapshot


def _clip(s: str | None, n: int = 280) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=25)
    args = p.parse_args()

    snap = GraphSnapshot.model_validate_json(args.snapshot.read_text())
    nodes = {n.id: n for n in snap.nodes}

    # Trace extraction
    result_path: list[str] = []
    query_text: str | None = None

    popped_by_step: dict[int, dict] = {}
    loaded_by_step: dict[int, dict] = {}

    # quick index of visited node_ids by step order
    visited_node_ids_in_order: list[str] = []

    with args.trace.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            et = ev.get("event_type")

            if et == "query_start":
                query_text = ev.get("query_text") or (ev.get("data") or {}).get("query")

            if et == "result":
                data = ev.get("data") or {}
                path = data.get("path")
                if isinstance(path, list):
                    result_path = [str(x) for x in path]

            if et != "traverse_trace":
                continue
            d = ev.get("data") or {}
            inner = d.get("event_type")

            if inner == "step_begin":
                step = d.get("step")
                popped = d.get("popped")
                if isinstance(step, int) and isinstance(popped, dict):
                    popped_by_step[step] = popped

            if inner == "step_node_loaded":
                step = d.get("step")
                node_id = d.get("node_id")
                if isinstance(step, int) and isinstance(node_id, str):
                    loaded_by_step[step] = {"node_id": node_id, "node": d.get("node") or {}}

            if inner == "step_end" and d.get("skipped") is False:
                node_id = d.get("node_id")
                if isinstance(node_id, str):
                    visited_node_ids_in_order.append(node_id)

    print(f"query={query_text!r}")
    print(f"result_path_len={len(result_path)}")

    def _describe(node_id: str) -> str:
        n = nodes.get(node_id)
        if n is None:
            return f"{node_id} (missing)"
        txt = _clip(_norm_ws(n.text or ""), 260) if n.text else None
        return (
            f"{node_id} type={n.type!r} label={n.label!r} source={getattr(n, 'source', None)!r} level={getattr(n, 'level', None)!r} "
            + (f"text={txt!r}" if txt else "text=None")
        )

    print("\nReturned path nodes:")
    for i, node_id in enumerate(result_path, start=1):
        print(f"  {i}. {_describe(node_id)}")

    # Top scored popped nodes (these scores are frontier/ranker scores, not a final relevance score).
    popped_items = []
    for step, popped in popped_by_step.items():
        score = popped.get("score")
        node_id = popped.get("node_id")
        depth = popped.get("depth")
        if isinstance(node_id, str) and isinstance(score, (int, float)):
            popped_items.append((float(score), int(depth) if isinstance(depth, int) else None, int(step), node_id))

    popped_items.sort(reverse=True)
    print(f"\nTop {args.top_k} popped-by-score nodes:")
    for score, depth, step, node_id in popped_items[: max(0, args.top_k)]:
        n = nodes.get(node_id)
        label = n.label if n else None
        ntype = n.type if n else None
        print(f"  - score={score:.6f} depth={depth} step={step} node_id={node_id} type={ntype!r} label={label!r}")

    # Coverage checks for the three groups.
    group_terms = [
        "data management",
        "production and reprocessing",
        "tier0",
    ]

    def _haystack_for(node_id: str) -> str:
        n = nodes.get(node_id)
        if n is None:
            return ""
        parts = [n.label or "", n.text or ""]
        return ("\n".join(parts)).lower()

    coverage = {t: 0 for t in group_terms}
    for node_id in visited_node_ids_in_order:
        h = _haystack_for(node_id)
        for t in group_terms:
            if t in h:
                coverage[t] += 1

    print("\nVisited-node coverage counts (label/text contains term):")
    for t in group_terms:
        print(f"  - {t}: {coverage[t]}")

    # What types did we actually visit?
    type_counts = Counter()
    for node_id in visited_node_ids_in_order:
        n = nodes.get(node_id)
        if n is not None and isinstance(n.type, str):
            type_counts[n.type] += 1

    print("\nVisited node types:")
    for k, v in type_counts.most_common():
        print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

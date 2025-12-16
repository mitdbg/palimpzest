"""Summarize a GraphRAG trace JSONL.

Focuses on traversal dynamics: visited node types, edge types expanded, and depth/score stats.

Usage:
  .venv/bin/python scratchpad/scripts/summarize_graphrag_trace.py \
    --trace CURRENT_WORKSTREAM/exports/graphrag_trace_<run_id>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, type=Path)
    args = p.parse_args()

    node_type_counts: Counter[str] = Counter()
    edge_type_counts: Counter[str] = Counter()
    depths: list[int] = []
    scores: list[float] = []

    total_lines = 0
    traverse_lines = 0

    with args.trace.open() as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("event_type") != "traverse_trace":
                continue
            traverse_lines += 1
            d = ev.get("data") or {}
            et = d.get("event_type")

            if et == "step_begin":
                popped = d.get("popped") or {}
                depth = popped.get("depth")
                score = popped.get("score")
                if isinstance(depth, int):
                    depths.append(depth)
                if isinstance(score, (int, float)):
                    scores.append(float(score))

            if et == "step_node_loaded":
                node = d.get("node") or {}
                nt = node.get("type")
                if isinstance(nt, str):
                    node_type_counts[nt] += 1

            if et == "step_expand":
                for n in d.get("neighbors") or []:
                    etype = n.get("edge_type")
                    if isinstance(etype, str):
                        edge_type_counts[etype] += 1

    def _pct(p: float) -> str:
        return f"{(100.0 * p):.1f}%"

    print(f"trace={args.trace}")
    print(f"lines_total={total_lines} traverse_trace_lines={traverse_lines}")

    if depths:
        print(f"depth: min={min(depths)} max={max(depths)} avg={sum(depths)/len(depths):.2f}")
    if scores:
        print(f"score: min={min(scores):.6f} max={max(scores):.6f} avg={sum(scores)/len(scores):.6f}")

    total_loaded = sum(node_type_counts.values())
    print("node_types_by_loaded:")
    for k, v in node_type_counts.most_common():
        print(f"  - {k}: {v} ({_pct(v / max(1, total_loaded))})")

    total_edges = sum(edge_type_counts.values())
    print("edge_types_by_expanded_neighbor:")
    for k, v in edge_type_counts.most_common():
        print(f"  - {k}: {v} ({_pct(v / max(1, total_edges))})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

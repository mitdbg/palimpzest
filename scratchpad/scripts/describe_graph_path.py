"""Describe a node-id path in a graph snapshot.

Usage:
  .venv/bin/python scratchpad/scripts/describe_graph_path.py \
    --snapshot data/cms_refined_graph.json \
    --ids ff7c...,3975...

Or:
  PZ_PATH_IDS="ff7c...,3975..." .venv/bin/python scratchpad/scripts/describe_graph_path.py --snapshot data/cms_refined_graph.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphSnapshot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--ids", type=str, default="")
    args = p.parse_args()

    raw = (args.ids or "").strip() or (os.getenv("PZ_PATH_IDS") or "").strip()
    if not raw:
        raise SystemExit("Provide --ids or set PZ_PATH_IDS")

    ids = [s.strip() for s in raw.split(",") if s.strip()]

    snap = GraphSnapshot.model_validate_json(args.snapshot.read_text())
    by_id = {n.id: n for n in snap.nodes}

    for i, node_id in enumerate(ids, start=1):
        n = by_id.get(node_id)
        if n is None:
            print(f"{i}. {node_id} (missing)")
            continue
        print(
            f"{i}. {node_id} type={n.type!r} label={n.label!r} source={getattr(n, 'source', None)!r} level={getattr(n, 'level', None)!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

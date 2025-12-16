"""Summarize degree statistics for a given edge type in a GraphDataset snapshot.

Focus: sanity-check large induced edge overlays like `sim:knn`.

Example:
  /Users/jason/projects/mit/palimpzest/.venv/bin/python \
    scratchpad/scripts/summarize_graph_edge_degrees.py \
    --graph data/cms_combined_graph.with_embeddings.knn10.json \
    --edge-type sim:knn \
    --top 25

Output:
- JSON to stdout with counts, degree distribution percentiles, and top hubs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize per-node degrees for an edge type")
    p.add_argument("--graph", required=True, help="Path to GraphDataset snapshot JSON")
    p.add_argument("--edge-type", default="sim:knn", help="Edge type to analyze")
    p.add_argument("--top", type=int, default=25, help="Number of top hubs to print")
    p.add_argument(
        "--include-overlay",
        action="store_true",
        help="Include overlay edges when loading snapshot (default: true in snapshot; this flag is informational)",
    )
    return p.parse_args()


def _node_display(node) -> str:
    label = getattr(node, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    attrs = getattr(node, "attrs", {}) or {}
    md = attrs.get("metadata")
    if isinstance(md, dict):
        name = md.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = attrs.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _percentiles(values: list[int], ps: list[float]) -> dict[str, float]:
    if not values:
        return {str(p): 0.0 for p in ps}
    xs = sorted(values)

    def pick(p: float) -> float:
        # Nearest-rank percentile (simple, stable)
        if p <= 0:
            return float(xs[0])
        if p >= 100:
            return float(xs[-1])
        k = int((p / 100.0) * (len(xs) - 1))
        return float(xs[k])

    return {str(p): pick(p) for p in ps}


def main() -> int:
    args = _parse_args()
    graph_path = Path(args.graph)
    edge_type = (args.edge_type or "").strip()
    if not edge_type:
        raise ValueError("--edge-type must be non-empty")

    g = GraphDataset.load(graph_path)
    snap = g.to_snapshot()

    nodes_by_id = {n.id: n for n in snap.nodes}

    out_deg: dict[str, int] = defaultdict(int)
    in_deg: dict[str, int] = defaultdict(int)
    undirected_pairs: set[tuple[str, str]] = set()

    total_edges = len(snap.edges)
    et_edges = 0
    for e in snap.edges:
        if e.type != edge_type:
            continue
        et_edges += 1
        out_deg[e.src] += 1
        in_deg[e.dst] += 1
        a, b = (e.src, e.dst)
        if a <= b:
            undirected_pairs.add((a, b))
        else:
            undirected_pairs.add((b, a))

    node_ids = list(nodes_by_id.keys())

    combined = []
    for nid in node_ids:
        combined.append(out_deg.get(nid, 0) + in_deg.get(nid, 0))

    nonzero = [d for d in combined if d > 0]

    top = max(0, int(args.top))
    hubs = sorted(
        (
            (
                nid,
                out_deg.get(nid, 0),
                in_deg.get(nid, 0),
                out_deg.get(nid, 0) + in_deg.get(nid, 0),
            )
            for nid in node_ids
        ),
        key=lambda t: (t[3], t[1], t[2], t[0]),
        reverse=True,
    )[:top]

    hub_rows = []
    for nid, od, idg, cd in hubs:
        node = nodes_by_id.get(nid)
        hub_rows.append(
            {
                "node_id": nid,
                "type": getattr(node, "type", None) if node is not None else None,
                "label": _node_display(node) if node is not None else "",
                "out_degree": od,
                "in_degree": idg,
                "degree": cd,
                "has_embedding_st": bool(
                    (getattr(node, "attrs", {}) or {}).get("embedding_st")
                    if node is not None
                    else False
                ),
            }
        )

    pct = _percentiles(combined, [0, 25, 50, 75, 90, 95, 99, 100])
    pct_nonzero = _percentiles(nonzero, [0, 25, 50, 75, 90, 95, 99, 100])

    result = {
        "graph": str(graph_path),
        "graph_id": g.graph_id,
        "revision": g.revision,
        "edge_type": edge_type,
        "node_count": len(node_ids),
        "total_edges": total_edges,
        "edge_type_edges": et_edges,
        "edge_type_unique_undirected_pairs": len(undirected_pairs),
        "degree_distribution": {
            "combined_degree_percentiles": pct,
            "combined_degree_nonzero_percentiles": pct_nonzero,
            "avg_degree": (sum(combined) / len(combined)) if combined else 0.0,
            "avg_degree_nonzero": (sum(nonzero) / len(nonzero)) if nonzero else 0.0,
            "nodes_with_degree_gt0": len(nonzero),
        },
        "top_hubs": hub_rows,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic migration: refined CMS graph -> v1 snapshot.

This script is the minimal Milestone B implementation described in CURRENT_WORKSTREAM:
- Takes a refined snapshot (optionally already enriched with Skeleton + mentions).
- Stamps layer metadata (Skeleton vs Data).
- Normalizes `mentions` edge evidence into a stable v1-ish format.
- Optionally strips overlay artifacts (e.g. similarity edges) and embedding attrs.
- Writes an output snapshot + a JSON validation report.

Example:
  ./.venv/bin/python scratchpad/scripts/migrate_to_v1_schema.py \
    --in data/cms_refined_graph.with_skeleton_mentions.json \
    --out data/cms_v1_graph.json \
    --report CURRENT_WORKSTREAM/exports/cms_v1_validation.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palimpzest.core.data.graph_dataset import GraphDataset


SKELETON_TYPES: set[str] = {"team", "service", "process", "concept"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate refined CMS snapshot to v1 schema")
    p.add_argument("--in", dest="in_graph", required=True, help="Input graph snapshot JSON")
    p.add_argument("--out", dest="out_graph", required=True, help="Output v1 graph snapshot JSON")
    p.add_argument("--report", required=True, help="Write validation/report JSON")
    p.add_argument(
        "--strip-overlay",
        action="store_true",
        help="Drop known overlay edges (e.g. sim:*) and overlay:* edges.",
    )
    p.add_argument(
        "--strip-embedding-keys",
        default="embedding_st",
        help="Comma-separated node attrs keys to drop (e.g. embedding_st). Use empty string to keep all.",
    )
    return p.parse_args()


def _normalize_mentions_edge_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert current predicate-induction attrs into a stable v1 evidence shape.

    Current shape (from GraphDataset.run_predicate_induction):
      {
        "spec_id": "...",
        "method": "predicate",
        "mode": "any"|"all",
        "predicates": [{"kind":..., "params":...}, ...],
        "evidence": {"text_contains": {"matched":..., "source_field":...}, ...}
      }

    Target shape (per CURRENT_WORKSTREAM/notes/schema_v1.md):
      {
        "evidence": {"kind": ..., "source_field": ..., "matched": ...},
        "confidence": 1.0,
        "method": "predicate",
        "spec_id": "..."
      }
    """

    out: dict[str, Any] = {}

    spec_id = attrs.get("spec_id")
    if isinstance(spec_id, str):
        out["spec_id"] = spec_id

    method = attrs.get("method")
    if isinstance(method, str):
        out["method"] = method

    evidence_by_kind = attrs.get("evidence")
    if isinstance(evidence_by_kind, dict) and evidence_by_kind:
        # Prefer a deterministic kind ordering.
        kind = sorted([k for k in evidence_by_kind.keys() if isinstance(k, str)])[0]
        ev = evidence_by_kind.get(kind)
        ev_obj: dict[str, Any] = {"kind": kind}
        if isinstance(ev, dict):
            # text_contains/regex_match share `matched` + `source_field`.
            if "source_field" in ev:
                ev_obj["source_field"] = ev.get("source_field")
            if "matched" in ev:
                ev_obj["matched"] = ev.get("matched")
            if "pattern" in ev:
                ev_obj["pattern"] = ev.get("pattern")
        out["evidence"] = ev_obj

    # MVP: high precision rules; default confidence=1.0 when evidence exists.
    if "evidence" in out:
        out.setdefault("confidence", 1.0)

    return out


def main() -> int:
    args = _parse_args()
    in_path = Path(args.in_graph)
    out_path = Path(args.out_graph)
    report_path = Path(args.report)

    strip_embedding_keys = [k.strip() for k in str(args.strip_embedding_keys).split(",") if k.strip()]

    g = GraphDataset.load(in_path)

    # --- Node normalization: stamp layers + optionally drop embedding attrs
    nodes_by_type = Counter()
    layers = Counter()
    missing_label_skeleton: list[str] = []
    missing_slug_skeleton: list[str] = []

    for node in g.store.iter_nodes():
        if node.type in SKELETON_TYPES:
            layer = "skeleton"
            if not (isinstance(node.label, str) and node.label.strip()):
                missing_label_skeleton.append(node.id)
            slug = (node.attrs or {}).get("slug")
            if not (isinstance(slug, str) and slug.strip()):
                missing_slug_skeleton.append(node.id)
        else:
            layer = "data"

        node.attrs = dict(node.attrs or {})
        node.attrs.setdefault("layer", layer)
        layers[layer] += 1
        nodes_by_type[str(node.type)] += 1

        if strip_embedding_keys:
            for k in strip_embedding_keys:
                node.attrs.pop(k, None)

    # --- Edge normalization: optionally drop overlay, normalize mentions evidence
    edges_by_type_before = Counter([e.type for e in g.store.iter_edges()])

    keep_edge_ids: set[str] = set()
    dropped_overlay_edges = 0
    mentions_without_evidence = 0

    for edge in list(g.store.iter_edges()):
        # Strip overlay edges if requested.
        if args.strip_overlay:
            if edge.type.startswith("overlay:") or edge.type.startswith("sim:"):
                dropped_overlay_edges += 1
                continue

        if edge.type == "mentions":
            edge.attrs = dict(edge.attrs or {})
            norm = _normalize_mentions_edge_attrs(edge.attrs)
            if "evidence" not in norm:
                mentions_without_evidence += 1
            edge.attrs = norm

        keep_edge_ids.add(edge.id)

    if dropped_overlay_edges:
        for edge_id in [e.id for e in g.store.iter_edges() if e.id not in keep_edge_ids]:
            g.remove_edge(edge_id)

    # --- Validation: dangling refs
    node_ids = set(g.store.get_node_ids())
    dangling_edges: list[dict[str, str]] = []
    for e in g.store.iter_edges():
        if e.src not in node_ids or e.dst not in node_ids:
            dangling_edges.append({"edge_id": e.id, "src": e.src, "dst": e.dst, "type": e.type})

    edges_by_type_after = Counter([e.type for e in g.store.iter_edges()])

    report: dict[str, Any] = {
        "status": "ok" if (not dangling_edges and not missing_label_skeleton and not missing_slug_skeleton) else "needs_attention",
        "in_graph": str(in_path),
        "out_graph": str(out_path),
        "strip_overlay": bool(args.strip_overlay),
        "strip_embedding_keys": strip_embedding_keys,
        "counts": {
            "nodes": len(g.store.get_node_ids()),
            "edges": len(list(g.store.iter_edges())),
            "nodes_by_type": dict(nodes_by_type),
            "nodes_by_layer": dict(layers),
            "edges_by_type_before": dict(edges_by_type_before),
            "edges_by_type_after": dict(edges_by_type_after),
            "dropped_overlay_edges": dropped_overlay_edges,
            "mentions_without_evidence": mentions_without_evidence,
        },
        "violations": {
            "dangling_edges": dangling_edges[:50],
            "missing_label_skeleton": missing_label_skeleton[:50],
            "missing_slug_skeleton": missing_slug_skeleton[:50],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.save(out_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

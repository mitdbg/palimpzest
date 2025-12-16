"""Add Skeleton nodes and deterministic `mentions` edges over the refined CMS graph.

This script:
1) Loads the refined snapshot (GraphDataset JSON).
2) Adds a small Skeleton inventory (teams/services/processes/concepts).
3) Runs a high-precision mentions induction: Data -> Skeleton.
4) Saves an updated snapshot + a summary JSON artifact.

Why this shape?
- Keeps v1 Skeleton additive (Data nodes unchanged).
- Uses predicate induction with a typed bipartite candidate generator to avoid O(N^2).

Example:
  /Users/jason/projects/mit/palimpzest/.venv/bin/python \
    scratchpad/scripts/add_skeleton_and_mentions_refined.py \
    --in-graph data/cms_refined_graph.json \
    --seed CURRENT_WORKSTREAM/notes/skeleton_seed.json \
    --out-graph data/cms_refined_graph.with_skeleton_mentions.json \
    --out-summary CURRENT_WORKSTREAM/exports/refined_skeleton_mentions_summary.json

Notes:
- Uses `edge_type="mentions"`.
- Evidence is stored on each mentions edge in edge attrs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.core.data.graph_store import GraphEdge, GraphNode
from palimpzest.core.data.induction import InductionSpec
from palimpzest.utils.hash_helpers import hash_for_id


SKELETON_TYPES = {"team", "service", "process", "concept"}
DATA_TYPES_DEFAULT = {"jira_tickets", "git_docs", "git_directory"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add skeleton nodes + mentions induction on refined graph")
    p.add_argument("--in-graph", required=True, help="Input refined graph snapshot JSON")
    p.add_argument("--seed", required=True, help="Skeleton seed JSON (teams/services/processes/concepts)")
    p.add_argument("--out-graph", required=True, help="Output graph snapshot JSON")
    p.add_argument("--out-summary", required=True, help="Write summary JSON")
    p.add_argument(
        "--data-types",
        default=",".join(sorted(DATA_TYPES_DEFAULT)),
        help="Comma-separated data node types to treat as sources",
    )
    p.add_argument(
        "--overwrite-skeleton",
        action="store_true",
        help="Overwrite existing skeleton nodes if their IDs already exist",
    )
    return p.parse_args()


def _load_seed(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError("seed must be a JSON object")
    return obj


def _mk_node(*, node_id: str, node_type: str, label: str, slug: str, aliases: list[str] | None = None) -> GraphNode:
    attrs: dict[str, Any] = {"slug": slug}
    # Put aliases in attrs.aliases so predicate induction can match them via target_fields.
    if aliases:
        attrs["aliases"] = sorted({a.strip() for a in aliases if isinstance(a, str) and a.strip()})
    return GraphNode(id=node_id, type=node_type, label=label, source="skeleton", attrs=attrs)


def _default_edge_id(*, src: str, dst: str, edge_type: str) -> str:
    return hash_for_id(f"edge:{edge_type}:{src}->{dst}")


def _add_skeleton_nodes(g: GraphDataset, seed: dict[str, Any], *, overwrite: bool) -> list[str]:
    created: list[str] = []

    def add_group(group_key: str, node_type: str) -> None:
        items = seed.get(group_key) or []
        if not isinstance(items, list):
            raise ValueError(f"seed[{group_key!r}] must be a list")
        for it in items:
            if not isinstance(it, dict):
                raise ValueError(f"seed[{group_key!r}] items must be objects")
            slug = str(it.get("slug") or "").strip()
            label = str(it.get("label") or "").strip()
            aliases = it.get("aliases")
            if aliases is None:
                aliases_list: list[str] = []
            elif isinstance(aliases, list) and all(isinstance(a, str) for a in aliases):
                aliases_list = aliases
            else:
                raise ValueError(f"seed[{group_key!r}].aliases must be list[str]")

            if not slug or not label:
                raise ValueError(f"seed[{group_key!r}] requires slug+label")

            node_id = f"{node_type}:{slug}"
            node = _mk_node(node_id=node_id, node_type=node_type, label=label, slug=slug, aliases=aliases_list)
            if g.has_node(node_id):
                if overwrite:
                    g.add_node(node, overwrite=True)
                continue
            g.add_node(node, overwrite=False)
            created.append(node_id)

    add_group("teams", "team")
    add_group("services", "service")
    add_group("processes", "process")
    add_group("concepts", "concept")

    return created


def main() -> int:
    args = _parse_args()
    in_path = Path(args.in_graph)
    seed_path = Path(args.seed)
    out_path = Path(args.out_graph)
    out_summary = Path(args.out_summary)

    data_types = {t.strip() for t in str(args.data_types).split(",") if t.strip()}
    if not data_types:
        raise ValueError("--data-types must include at least one type")

    g = GraphDataset.load(in_path)
    before = g.to_snapshot()

    seed = _load_seed(seed_path)
    created_skeleton_ids = _add_skeleton_nodes(g, seed, overwrite=bool(args.overwrite_skeleton))

    # Build and run mentions induction.
    spec = InductionSpec(
        edge_type="mentions",
        include_overlay=True,
        symmetric=False,
        allow_self_edges=False,
        overwrite=False,
        incremental_mode="source",
        generator={
            "kind": "typed_all_pairs",
            "params": {
                "src_types": sorted(data_types),
                "dst_types": ["team", "service", "process", "concept"],
                "allow_self_edges": False,
            },
        },
        decider={
            "kind": "predicate_compound",
            "params": {
                "mode": "any",
                "predicates": [
                    {
                        "kind": "text_contains",
                        "params": {
                            "source_field": "text",
                            "target_fields": ["label", "attrs.name", "attrs.metadata.name", "attrs.aliases"],
                            "boundaries": True,
                        },
                    }
                ],
            },
        },
    )
    spec_id = g.add_induction(spec)
    induced = g.run_induction(spec_id, mode="full")

    after = g.to_snapshot()

    # Summarize mentions edges.
    mentions_edges = [e for e in after.edges if e.type == "mentions"]
    dst_type_counts = Counter()
    src_type_counts = Counter()
    for e in mentions_edges:
        try:
            src_t = g.get_node(e.src).type
            dst_t = g.get_node(e.dst).type
        except Exception:
            continue
        src_type_counts[src_t or "null"] += 1
        dst_type_counts[dst_t or "null"] += 1

    summary = {
        "status": "ok",
        "in_graph": str(in_path),
        "out_graph": str(out_path),
        "seed": str(seed_path),
        "graph_id": g.graph_id,
        "revision_before": before.revision,
        "revision_after": after.revision,
        "nodes_before": len(before.nodes),
        "nodes_after": len(after.nodes),
        "edges_before": len(before.edges),
        "edges_after": len(after.edges),
        "skeleton_nodes_created": len(created_skeleton_ids),
        "mentions_edges": len(mentions_edges),
        "mentions_src_type_counts": dict(src_type_counts),
        "mentions_dst_type_counts": dict(dst_type_counts),
        "induction_spec_id": spec_id,
        "induced_records": len(induced) if hasattr(induced, "__len__") else None,
        "data_types": sorted(data_types),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.save(out_path)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

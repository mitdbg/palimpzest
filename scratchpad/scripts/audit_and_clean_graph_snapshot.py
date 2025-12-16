from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import ijson


@dataclass
class AuditResult:
    node_count: int = 0
    edge_count: int = 0

    # Node issues
    nodes_missing_id: int = 0
    nodes_duplicate_id: int = 0

    # Edge issues
    edges_missing_fields: int = 0
    edges_missing_src: int = 0
    edges_missing_dst: int = 0
    edges_missing_type: int = 0
    edges_dangling_src: int = 0
    edges_dangling_dst: int = 0
    edges_self: int = 0
    edges_duplicate_id: int = 0

    removed_edges: int = 0


def _iter_json_array(path: Path, pointer: str) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        for item in ijson.items(f, pointer):
            if isinstance(item, dict):
                yield item


def _json_default(obj: object):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def audit_snapshot(path: Path) -> tuple[AuditResult, set[str], set[str]]:
    """Stream-audit a GraphSnapshot JSON file.

    Returns:
      (audit, node_ids, edge_ids)
    """

    audit = AuditResult()

    node_ids: set[str] = set()
    seen_node_ids: set[str] = set()

    for node in _iter_json_array(path, "nodes.item"):
        audit.node_count += 1
        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            audit.nodes_missing_id += 1
            continue
        if nid in seen_node_ids:
            audit.nodes_duplicate_id += 1
            continue
        seen_node_ids.add(nid)
        node_ids.add(nid)

    edge_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    for edge in _iter_json_array(path, "edges.item"):
        audit.edge_count += 1

        eid = edge.get("id")
        if isinstance(eid, str) and eid:
            if eid in seen_edge_ids:
                audit.edges_duplicate_id += 1
            else:
                seen_edge_ids.add(eid)
                edge_ids.add(eid)

        src = edge.get("src")
        dst = edge.get("dst")
        etype = edge.get("type")

        if not isinstance(src, str) or not src:
            audit.edges_missing_src += 1
        if not isinstance(dst, str) or not dst:
            audit.edges_missing_dst += 1
        if not isinstance(etype, str) or not etype:
            audit.edges_missing_type += 1

        if isinstance(src, str) and src and src not in node_ids:
            audit.edges_dangling_src += 1
        if isinstance(dst, str) and dst and dst not in node_ids:
            audit.edges_dangling_dst += 1
        if isinstance(src, str) and isinstance(dst, str) and src and dst and src == dst:
            audit.edges_self += 1

        if (
            not isinstance(src, str)
            or not src
            or not isinstance(dst, str)
            or not dst
            or not isinstance(etype, str)
            or not etype
        ):
            audit.edges_missing_fields += 1

    return audit, node_ids, edge_ids


def clean_snapshot(
    *,
    in_path: Path,
    out_path: Path,
    drop_dangling_edges: bool,
    drop_self_edges_for_types: set[str],
) -> AuditResult:
    """Write a cleaned snapshot.

    Policy:
      - never deletes nodes
      - optionally drops edges where src/dst are missing from nodes
      - optionally drops self-edges for specific edge types
      - always drops edges that are missing required fields (id/src/dst/type)
    """

    audit, node_ids, _ = audit_snapshot(in_path)

    with in_path.open() as f:
        header = json.load(f)

    # We will rewrite edges; keep everything else as-is.
    # Remove huge edge list from header to avoid double memory.
    header.pop("edges", None)

    tmp_edges_path = out_path.with_suffix(out_path.suffix + ".edges.tmp")
    removed = 0
    kept_edges = 0

    try:
        with tmp_edges_path.open("w") as ef:
            ef.write("[")
            first = True
            for edge in _iter_json_array(in_path, "edges.item"):
                src = edge.get("src")
                dst = edge.get("dst")
                etype = edge.get("type")
                eid = edge.get("id")

                # Required fields
                if not (
                    isinstance(eid, str)
                    and eid
                    and isinstance(src, str)
                    and src
                    and isinstance(dst, str)
                    and dst
                    and isinstance(etype, str)
                    and etype
                ):
                    removed += 1
                    continue

                if drop_dangling_edges and (src not in node_ids or dst not in node_ids):
                    removed += 1
                    continue

                if etype in drop_self_edges_for_types and src == dst:
                    removed += 1
                    continue

                if not first:
                    ef.write(",\n")
                first = False
                ef.write(json.dumps(edge, separators=(",", ":"), default=_json_default))
                kept_edges += 1

            ef.write("]\n")
    except Exception:
        tmp_edges_path.unlink(missing_ok=True)
        raise

    audit.removed_edges = removed

    # Assemble output snapshot
    with out_path.open("w") as out:
        out.write("{\n")
        # Write header keys (except edges) deterministically-ish
        keys = list(header.keys())
        for i, k in enumerate(keys):
            out.write(json.dumps(k))
            out.write(": ")
            out.write(json.dumps(header[k], indent=2, sort_keys=True, default=_json_default))
            out.write(",\n")

        out.write('"edges": ')
        out.write(tmp_edges_path.read_text())
        out.write("}\n")

    tmp_edges_path.unlink(missing_ok=True)
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit and (optionally) clean a Palimpzest GraphSnapshot JSON.")
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.with_chunk_mentions.with_knn10.json"),
        help="Input snapshot path.",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.cleaned.json"),
        help="Output cleaned snapshot path.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file.")

    ap.add_argument(
        "--drop-dangling-edges",
        action="store_true",
        help="Remove edges whose src/dst node ids are missing from nodes.",
    )
    ap.add_argument(
        "--drop-self-edge-type",
        action="append",
        dest="drop_self_edge_types",
        default=["sim:knn", "overlay:next_chunk"],
        help="Edge type(s) for which self-edges should be dropped (repeatable).",
    )
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Only audit; do not write a cleaned snapshot.",
    )

    args = ap.parse_args()

    if args.out_path.exists() and not args.overwrite and not args.audit_only:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_path} (pass --overwrite)")

    audit, _, _ = audit_snapshot(args.in_path)

    print("Audit:")
    print(f"  nodes: {audit.node_count}")
    print(f"  edges: {audit.edge_count}")
    print(f"  nodes_missing_id: {audit.nodes_missing_id}")
    print(f"  nodes_duplicate_id: {audit.nodes_duplicate_id}")
    print(f"  edges_missing_fields: {audit.edges_missing_fields}")
    print(f"  edges_missing_src: {audit.edges_missing_src}")
    print(f"  edges_missing_dst: {audit.edges_missing_dst}")
    print(f"  edges_missing_type: {audit.edges_missing_type}")
    print(f"  edges_duplicate_id: {audit.edges_duplicate_id}")
    print(f"  edges_dangling_src: {audit.edges_dangling_src}")
    print(f"  edges_dangling_dst: {audit.edges_dangling_dst}")
    print(f"  edges_self: {audit.edges_self}")

    if args.audit_only:
        return

    cleaned = clean_snapshot(
        in_path=args.in_path,
        out_path=args.out_path,
        drop_dangling_edges=bool(args.drop_dangling_edges),
        drop_self_edges_for_types=set(args.drop_self_edge_types),
    )

    print("\nClean:")
    print(f"  removed_edges: {cleaned.removed_edges}")
    print(f"  out: {args.out_path}")


if __name__ == "__main__":
    main()

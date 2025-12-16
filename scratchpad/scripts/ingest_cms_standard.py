from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from palimpzest.core.data.graph_dataset import GraphEdge, GraphNode, GraphSnapshot
from palimpzest.utils.hash_helpers import hash_for_id


def _node_label(node: dict[str, Any]) -> str | None:
    md = node.get("metadata") or {}
    name = md.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _node_text(node: dict[str, Any]) -> str | None:
    # Prefer source_text; fall back to summary.
    for key in ("source_text", "summary"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _edge_id(*, edge_type: str, src: str, dst: str) -> str:
    return hash_for_id(f"cms_standard:{edge_type}:{src}->{dst}")


_TITLE_RE = re.compile(r"^Title:\s*(\S+)\s*$", re.MULTILINE)


def _extract_ticket_key(node: dict[str, Any]) -> str | None:
    # Prefer metadata.name (when it is a ticket key); else parse Title: from source_text.
    md = node.get("metadata")
    if isinstance(md, dict):
        name = md.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    text = _node_text(node)
    if isinstance(text, str) and text:
        m = _TITLE_RE.search(text)
        if m:
            return m.group(1).strip()
    return None


def ingest(
    *,
    input_path: Path,
    graph_id: str,
    edge_field: str,
    include_eh_parents: bool,
    domain_edges_path: Path | None,
) -> GraphSnapshot:
    payload = json.loads(input_path.read_text())
    if not isinstance(payload, dict) or "nodes" not in payload:
        raise ValueError("Unexpected cms_standard.json schema: expected dict with 'nodes'")

    nodes_raw = payload["nodes"]
    if not isinstance(nodes_raw, list):
        raise ValueError("Unexpected cms_standard.json schema: 'nodes' must be a list")

    nodes: list[GraphNode] = []
    node_ids: set[str] = set()
    ticket_key_to_node_id: dict[str, str] = {}

    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        node_id = n.get("block_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        node_ids.add(node_id)

        attrs: dict[str, Any] = {}
        for k in ("level", "relevance", "access_count"):
            if k in n:
                attrs[k] = n.get(k)
        md = n.get("metadata")
        if isinstance(md, dict) and md:
            attrs["metadata"] = md

        nodes.append(
            GraphNode(
                id=node_id,
                label=_node_label(n),
                type="cms_block",
                attrs=attrs,
                text=_node_text(n),
            )
        )

        ticket_key = _extract_ticket_key(n)
        if isinstance(ticket_key, str) and ticket_key and ticket_key not in ticket_key_to_node_id:
            ticket_key_to_node_id[ticket_key] = node_id

    edges_by_key: dict[tuple[str, str, str], GraphEdge] = {}  # (type, src, dst)

    def _upsert_hierarchy_edge(*, src: str, dst: str, source_field: str) -> None:
        if src not in node_ids or dst not in node_ids:
            return
        key = ("hierarchy:child", src, dst)
        existing = edges_by_key.get(key)
        if existing is None:
            edges_by_key[key] = GraphEdge(
                id=_edge_id(edge_type="hierarchy:child", src=src, dst=dst),
                src=src,
                dst=dst,
                type="hierarchy:child",
                directed=True,
                attrs={"source_fields": [source_field]},
            )
            return

        fields = existing.attrs.get("source_fields")
        if not isinstance(fields, list):
            fields = []
        if source_field not in fields:
            fields.append(source_field)
        existing.attrs["source_fields"] = fields

    def _add_domain_edge(*, src_key: str, dst_key: str, domain_type: str) -> bool:
        src_node_id = ticket_key_to_node_id.get(src_key)
        dst_node_id = ticket_key_to_node_id.get(dst_key)
        if not src_node_id or not dst_node_id:
            return False
        if src_node_id not in node_ids or dst_node_id not in node_ids:
            return False

        edge_type = f"domain:{domain_type}"
        key = (edge_type, src_node_id, dst_node_id)
        if key in edges_by_key:
            return True

        edges_by_key[key] = GraphEdge(
            id=_edge_id(edge_type=edge_type, src=src_node_id, dst=dst_node_id),
            src=src_node_id,
            dst=dst_node_id,
            type=edge_type,
            directed=True,
            attrs={
                "source_file": "domain_edges.json",
                "domain_source": src_key,
                "domain_target": dst_key,
            },
        )
        return True

    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        src = n.get("block_id")
        if not isinstance(src, str) or not src:
            continue

        # Outgoing hierarchy edges from the requested field.
        children = n.get(edge_field)
        if isinstance(children, list):
            for dst in children:
                if not isinstance(dst, str) or not dst:
                    continue
                _upsert_hierarchy_edge(src=src, dst=dst, source_field=edge_field)

        # Additional hierarchy edges derived from eh_parents (more complete in cms_standard.json).
        if include_eh_parents:
            parents = n.get("eh_parents")
            if isinstance(parents, list):
                for parent in parents:
                    if not isinstance(parent, str) or not parent:
                        continue
                    _upsert_hierarchy_edge(src=parent, dst=src, source_field="eh_parents")

    # The upstream JSON has its own version; keep PZ snapshot version separate.
    # Set revision=1 because this is an imported baseline.
    if domain_edges_path is not None and domain_edges_path.exists():
        domain_payload = json.loads(domain_edges_path.read_text())
        if not isinstance(domain_payload, list):
            raise ValueError("Unexpected domain_edges.json schema: expected a list")

        added = 0
        skipped = 0
        for item in domain_payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            src_key = item.get("source")
            dst_key = item.get("target")
            domain_type = item.get("type") or "references"
            if not isinstance(src_key, str) or not isinstance(dst_key, str):
                skipped += 1
                continue
            if not isinstance(domain_type, str) or not domain_type:
                domain_type = "references"

            if _add_domain_edge(src_key=src_key, dst_key=dst_key, domain_type=domain_type):
                added += 1
            else:
                skipped += 1

        # Keep some useful stats for debugging.
        # (This is stored on the graph; it does not affect operator semantics.)
        # Note: attrs on the snapshot itself isn't modeled, so we store it on the graph_id only via printouts.
        print(f"Domain edges: loaded={len(domain_payload)} added={added} skipped={skipped}")

    return GraphSnapshot(graph_id=graph_id, revision=1, nodes=nodes, edges=list(edges_by_key.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest cms_standard.json into Palimpzest GraphSnapshot JSON")
    parser.add_argument(
        "--input",
        default="/Users/jason/projects/mit/sct/data/cms_hierarchy/cms_standard.json",
        help="Path to cms_standard.json",
    )
    parser.add_argument(
        "--output",
        default="/Users/jason/projects/mit/palimpzest/CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json",
        help="Where to write GraphSnapshot JSON",
    )
    parser.add_argument("--graph-id", default="cms_standard", help="GraphDataset graph_id")
    parser.add_argument(
        "--edge-field",
        default="children",
        help="Which field to interpret as outgoing child links (e.g. 'children' or 'eh_children')",
    )
    parser.add_argument(
        "--no-eh-parents",
        action="store_true",
        help="Disable adding hierarchy edges derived from 'eh_parents' (parent->child).",
    )
    parser.add_argument(
        "--domain-edges",
        default="/Users/jason/projects/mit/sct/data/cms_hierarchy/domain_edges.json",
        help="Optional path to domain_edges.json (ticket-to-ticket edges like references).",
    )
    parser.add_argument(
        "--no-domain-edges",
        action="store_true",
        help="Disable merging domain_edges.json.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    snapshot = ingest(
        input_path=input_path,
        graph_id=args.graph_id,
        edge_field=args.edge_field,
        include_eh_parents=not args.no_eh_parents,
        domain_edges_path=None if args.no_domain_edges else Path(args.domain_edges),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))

    print(f"Wrote GraphSnapshot: {output_path}")
    print(f"graph_id={snapshot.graph_id} revision={snapshot.revision} nodes={len(snapshot.nodes)} edges={len(snapshot.edges)}")


if __name__ == "__main__":
    main()

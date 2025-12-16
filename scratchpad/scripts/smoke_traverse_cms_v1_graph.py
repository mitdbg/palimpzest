"""Basic, no-LLM graph traversal smoke tests for `data/cms_v1_graph.json`.

This is meant to quickly answer: "what does traversal look like on cms_v1_graph?"

Usage:
  ./.venv/bin/python scratchpad/scripts/smoke_traverse_cms_v1_graph.py

Optional:
  PZ_CMS_V1_SNAPSHOT=data/cms_v1_graph.json ./.venv/bin/python scratchpad/scripts/smoke_traverse_cms_v1_graph.py
"""

from __future__ import annotations

import os
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset


DEFAULT_SNAPSHOT = Path("data/cms_v1_graph.json")


def _env_path(name: str, default: Path) -> Path:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (Path.cwd() / p)


def _label(n) -> str:  # noqa: ANN001
    label = (getattr(n, "label", None) or "").strip()
    if label:
        return label
    attrs = getattr(n, "attrs", None) or {}
    name = attrs.get("name") if isinstance(attrs, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _node_preview(graph: GraphDataset, node_id: str) -> str:
    try:
        n = graph.get_node(node_id)
    except Exception:
        return f"{node_id} (missing)"

    label = _label(n)
    ntype = getattr(n, "type", None)
    lvl = getattr(n, "level", None)
    src = getattr(n, "source", None)
    return f"{node_id} label={label!r} type={ntype!r} level={lvl!r} source={src!r}"


def _edge_type_counts_with_total(graph: GraphDataset, *, top_k: int = 12) -> tuple[int, list[tuple[str, int]]]:
    counts: Counter[str] = Counter()
    for e in graph.store.iter_edges():
        counts[str(e.type)] += 1
    return sum(counts.values()), counts.most_common(top_k)


def _find_roots(graph: GraphDataset, *, edge_type: str = "hierarchy:child", max_roots: int = 25) -> list[str]:
    # Prefer explicit CMS hierarchy root heuristic (level==3).
    roots: list[str] = []
    for n in graph.store.iter_nodes():
        lvl = getattr(n, "level", None)
        if lvl is None:
            attrs = getattr(n, "attrs", None) or {}
            lvl = attrs.get("level") if isinstance(attrs, dict) else None
        if lvl == 3:
            roots.append(str(n.id))
            if len(roots) >= max_roots:
                return roots

    # Fallback: nodes with no incoming hierarchy edges.
    has_parent: set[str] = set()
    for e in graph.store.iter_edges():
        if str(e.type) == edge_type:
            has_parent.add(str(e.dst))

    for n in graph.store.iter_nodes():
        nid = str(n.id)
        if nid not in has_parent:
            roots.append(nid)
            if len(roots) >= max_roots:
                break

    return roots


def _children(graph: GraphDataset, node_id: str, *, edge_type: str = "hierarchy:child", limit: int = 20) -> list[str]:
    out: list[str] = []
    for e in graph.iter_out_edges(node_id, edge_type=edge_type):
        out.append(str(e.dst))
        if len(out) >= limit:
            break
    return out


def _parents(graph: GraphDataset, node_id: str, *, edge_type: str = "hierarchy:child", limit: int = 20) -> list[str]:
    out: list[str] = []
    for e in graph.iter_in_edges(node_id, edge_type=edge_type):
        out.append(str(e.src))
        if len(out) >= limit:
            break
    return out


def _parent_chain(graph: GraphDataset, start: str, *, edge_type: str = "hierarchy:child", max_hops: int = 6) -> list[str]:
    chain = [start]
    cur = start
    for _ in range(max_hops):
        ps = _parents(graph, cur, edge_type=edge_type, limit=1)
        if not ps:
            break
        cur = ps[0]
        chain.append(cur)
    return chain


def _label_search(graph: GraphDataset, query: str, *, limit: int = 10) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    hits: list[str] = []
    for n in graph.store.iter_nodes():
        label = _label(n)
        if label and q in label.lower():
            hits.append(str(n.id))
            if len(hits) >= limit:
                break
    return hits


@dataclass
class _Path:
    node_ids: list[str]


def _shortest_path_undirected(
    graph: GraphDataset,
    src: str,
    dst: str,
    *,
    edge_type: str = "hierarchy:child",
    max_visits: int = 50_000,
) -> _Path | None:
    if src == dst:
        return _Path([src])

    q: deque[str] = deque([src])
    parent: dict[str, str | None] = {src: None}
    visits = 0

    while q and visits < max_visits:
        cur = q.popleft()
        visits += 1

        # Treat hierarchy edges as undirected for connectivity checks.
        for e in graph.iter_out_edges(cur, edge_type=edge_type):
            nxt = str(e.dst)
            if nxt not in parent:
                parent[nxt] = cur
                if nxt == dst:
                    q.clear()
                    break
                q.append(nxt)

        for e in graph.iter_in_edges(cur, edge_type=edge_type):
            nxt = str(e.src)
            if nxt not in parent:
                parent[nxt] = cur
                if nxt == dst:
                    q.clear()
                    break
                q.append(nxt)

    if dst not in parent:
        return None

    path: list[str] = []
    cur: str | None = dst
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return _Path(path)


def main() -> int:
    snapshot_path = _env_path("PZ_CMS_V1_SNAPSHOT", DEFAULT_SNAPSHOT)
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot not found: {snapshot_path}")

    print(f"Loading graph: {snapshot_path}")
    graph = GraphDataset.load(snapshot_path)

    nodes_n = graph.store.count_nodes()
    edges_n, top_edge_types = _edge_type_counts_with_total(graph)
    print(f"nodes={nodes_n} edges={edges_n}")

    print("\nTop edge types:")
    for t, c in top_edge_types:
        print(f"  {t}: {c}")

    print("\nRoot candidates:")
    roots = _find_roots(graph)
    for rid in roots[:8]:
        print(" ", _node_preview(graph, rid))

    if roots:
        root = roots[0]
        kids = _children(graph, root, limit=12)
        print(f"\nChildren of root {root} (hierarchy:child, up to 12):")
        for cid in kids:
            print(" ", _node_preview(graph, cid))

    # Label searches that tend to exist in CMS docs.
    terms = [
        "Tier0",
        "Grid",
        "Couch",
        "Production",
        "Reprocessing",
        "Data",
        "Agent",
    ]

    picked: list[str] = []
    print("\nLabel search samples:")
    for term in terms:
        hits = _label_search(graph, term, limit=3)
        if hits:
            picked.append(hits[0])
        print(f"  {term!r}: {len(hits)} hits")
        for hid in hits:
            print("   -", _node_preview(graph, hid))

    if picked:
        nid = picked[0]
        chain = _parent_chain(graph, nid, max_hops=8)
        print(f"\nParent chain from {nid} (hierarchy:child reversed):")
        for i, cid in enumerate(chain, start=0):
            print(f"  depth=-{i} ", _node_preview(graph, cid))

    if len(picked) >= 2:
        a, b = picked[0], picked[1]
        print(f"\nShortest undirected hierarchy path between {a} and {b} (may be None):")
        path = _shortest_path_undirected(graph, a, b)
        if path is None:
            print("  (no path found under hierarchy:child)")
        else:
            print(f"  hops={len(path.node_ids) - 1}")
            for pid in path.node_ids[:12]:
                print(" ", _node_preview(graph, pid))
            if len(path.node_ids) > 12:
                print(f"  … ({len(path.node_ids) - 12} more)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

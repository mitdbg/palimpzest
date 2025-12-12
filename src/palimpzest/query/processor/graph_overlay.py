from __future__ import annotations

from collections import Counter
from typing import Any

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge
from palimpzest.utils.hash_helpers import hash_for_id


def add_shortcut_overlay_edges_from_traversals(
    *,
    graph: GraphDataset,
    traversal_record_states: list[dict[str, Any]],
    edge_type: str = "overlay:shortcut",
    min_hops: int = 2,
    min_count: int = 2,
    max_edges: int = 200,
) -> int:
    """Materialize overlay shortcut edges from traversal paths.

    A "shortcut" edge is added from the first to the last node in a path, when:
    - path has at least `min_hops` hops (i.e. len(path_node_ids) >= min_hops + 1)
    - the (src,dst) pair appears at least `min_count` times across traversals

    Returns the number of edges added.
    """

    if min_hops < 1:
        raise ValueError("min_hops must be >= 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    if max_edges <= 0:
        raise ValueError("max_edges must be > 0")

    pair_counts: Counter[tuple[str, str]] = Counter()
    for state in traversal_record_states:
        path_node_ids = state.get("path_node_ids") or []
        if not isinstance(path_node_ids, list) or len(path_node_ids) < (min_hops + 1):
            continue
        src = str(path_node_ids[0])
        dst = str(path_node_ids[-1])
        if src == dst:
            continue
        pair_counts[(src, dst)] += 1

    added = 0
    for (src, dst), count in pair_counts.most_common():
        if count < min_count:
            break
        edge_id = hash_for_id(f"{edge_type}:{src}->{dst}")
        if graph.has_edge(edge_id):
            continue
        graph.add_edge(
            GraphEdge(
                id=edge_id,
                src=src,
                dst=dst,
                type=edge_type,
                directed=True,
                attrs={"count": int(count), "min_hops": int(min_hops)},
            ),
            overwrite=False,
        )
        added += 1
        if added >= max_edges:
            break

    return added


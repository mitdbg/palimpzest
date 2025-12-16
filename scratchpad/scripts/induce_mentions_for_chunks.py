from __future__ import annotations

import argparse
import time
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.core.data.induction import InductionLogEntry


def _count_chunk_mentions(graph: GraphDataset, *, chunk_type: str, edge_type: str) -> int:
    node_type: dict[str, str | None] = {n.id: n.type for n in graph.store.iter_nodes()}
    out = 0
    for e in graph.store.iter_edges():
        if e.type != edge_type:
            continue
        if node_type.get(e.src) == chunk_type or node_type.get(e.dst) == chunk_type:
            out += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Induce chunk->concept mention edges (predicate induction) for a chunked GraphDataset snapshot. "
            "Uses the text_anchor generator with a dst_types filter to keep candidate generation scalable."
        )
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.json"),
        help="Input chunked graph snapshot path (JSON).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.with_chunk_mentions.json"),
        help="Output graph snapshot path (JSON).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")

    ap.add_argument("--edge-type", type=str, default="mentions", help="Edge type to induce (default: mentions).")
    ap.add_argument(
        "--chunk-type",
        type=str,
        default="chunk",
        help="Node type for chunks (used for reporting only).",
    )
    ap.add_argument(
        "--dst-type",
        action="append",
        dest="dst_types",
        default=["concept"],
        help="Restrict target node types (repeatable). Default: concept.",
    )
    ap.add_argument(
        "--source-text-field",
        type=str,
        default="text",
        help="Source text field on chunk nodes (default: text).",
    )
    ap.add_argument(
        "--target-field",
        action="append",
        dest="target_fields",
        default=["label", "attrs.name", "attrs.metadata.name"],
        help="Target string fields to match against (repeatable).",
    )
    ap.add_argument(
        "--min-anchor-len",
        type=int,
        default=5,
        help="Minimum token length used for the anchor index (default: 5).",
    )
    ap.add_argument(
        "--boundaries",
        action="store_true",
        help="Use word-boundary matching (default: False).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="incremental",
        choices=["full", "incremental"],
        help="Induction run mode (default: incremental; targets chunk nodes via node revision).",
    )

    args = ap.parse_args()

    if args.out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_path} (pass --overwrite)")

    t0 = time.perf_counter()
    graph = GraphDataset.load(args.in_path)
    load_s = time.perf_counter() - t0

    before_edges = sum(1 for _ in graph.store.iter_edges())
    before_chunk_mentions = _count_chunk_mentions(graph, chunk_type=args.chunk_type, edge_type=args.edge_type)

    spec_id = graph.add_predicate_induction(
        edge_type=args.edge_type,
        generator_kind="text_anchor",
        generator_params={
            "source_text_field": args.source_text_field,
            "target_fields": list(args.target_fields),
            "min_anchor_len": int(args.min_anchor_len),
            "dst_types": list(args.dst_types),
        },
        predicates=[
            {
                "kind": "text_contains",
                "params": {
                    "source_field": args.source_text_field,
                    "target_fields": list(args.target_fields),
                    "boundaries": bool(args.boundaries),
                },
            }
        ],
        predicate_mode="all",
        symmetric=False,
        incremental_mode="source",
        allow_self_edges=False,
    )

    # Target only chunk nodes when using incremental mode.
    # In our saved snapshots, non-chunk nodes have revision=0; chunk nodes were added later and have revision>0.
    entry = graph.induction_log().get(spec_id)
    if entry is None:
        raise RuntimeError("Expected induction log entry to exist after add_predicate_induction")
    graph.induction_log().upsert(
        InductionLogEntry(
            spec=entry.spec,
            processed_node_ids=sorted(graph.store.get_node_ids()),
            last_revision=0,
        )
    )

    t1 = time.perf_counter()
    out = graph.run_induction(spec_id, mode=args.mode)
    run_s = time.perf_counter() - t1

    after_edges = sum(1 for _ in graph.store.iter_edges())
    after_chunk_mentions = _count_chunk_mentions(graph, chunk_type=args.chunk_type, edge_type=args.edge_type)

    t2 = time.perf_counter()
    graph.save(args.out_path, include_overlay=True)
    save_s = time.perf_counter() - t2

    print(
        "Induced chunk mentions and saved graph:\n"
        f"  in:  {args.in_path}\n"
        f"  out: {args.out_path}\n"
        f"  edge_type={args.edge_type} dst_types={args.dst_types}\n"
        f"  edges: {before_edges} -> {after_edges} (+{after_edges - before_edges})\n"
        f"  {args.edge_type} touching {args.chunk_type}: {before_chunk_mentions} -> {after_chunk_mentions} (+{after_chunk_mentions - before_chunk_mentions})\n"
        f"  timing_s: load={load_s:.2f} run={run_s:.2f} save={save_s:.2f} total={load_s + run_s + save_s:.2f}"
    )


if __name__ == "__main__":
    main()

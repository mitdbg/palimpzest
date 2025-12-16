from __future__ import annotations

import argparse
import time
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.query.processor.config import QueryProcessorConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk a GraphDataset and save a snapshot with overlay edges.")
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.json"),
        help="Input graph snapshot path (JSON).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.json"),
        help="Output graph snapshot path (JSON).",
    )
    ap.add_argument(
        "--only-type",
        action="append",
        dest="only_types",
        default=None,
        help=(
            "If provided, only chunk nodes whose 'type' is in this list. "
            "Repeatable (e.g. --only-type cms_block --only-type jira_ticket)."
        ),
    )
    ap.add_argument(
        "--input-col",
        type=str,
        default="text",
        help="Which input field to chunk (default: text). For Jira nodes in this dataset, try --input-col label.",
    )
    ap.add_argument(
        "--output-col",
        type=str,
        default="text",
        help="Which output field to write chunk text into (default: text).",
    )
    ap.add_argument("--chunker", type=str, default="markdown", help="Chunker kind (markdown|token|recursive_character|character).")
    ap.add_argument("--chunk-size", type=int, default=4000, help="Chunk size (tokens for token chunker; chars otherwise).")
    ap.add_argument("--chunk-overlap", type=int, default=800, help="Chunk overlap (tokens for token chunker; chars otherwise).")
    ap.add_argument("--progress", action="store_true", help="Show progress during execution.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")
    args = ap.parse_args()

    if args.out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_path} (pass --overwrite)")

    t0 = time.perf_counter()
    graph = GraphDataset.load(args.in_path)
    load_s = time.perf_counter() - t0

    nodes_before = len(graph)
    edges_before = sum(1 for _ in graph.store.iter_edges())

    config = QueryProcessorConfig(verbose=False, progress=args.progress)

    ds_src = graph
    if args.only_types:
        only = set(args.only_types)
        ds_src = ds_src.filter(lambda r, only=only: r.get("type") in only, depends_on=["type"])

    ds = ds_src.chunk(
        input_col=args.input_col,
        output_col=args.output_col,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunker_kind=args.chunker,
        graph=graph,
        edge_policy="has_and_next",
        has_chunk_edge_type="overlay:has_chunk",
        next_chunk_edge_type="overlay:next_chunk",
        chunk_node_type="chunk",
        overwrite_nodes=False,
        overwrite_edges=False,
    )

    t1 = time.perf_counter()
    # Exhaust the dataset to apply side effects.
    _ = list(ds.run(config))
    run_s = time.perf_counter() - t1

    nodes_after = len(graph)
    edges_after = sum(1 for _ in graph.store.iter_edges())

    graph.save(args.out_path, include_overlay=True)
    save_s = time.perf_counter() - (t1 + run_s)

    print(
        "Saved chunked graph:\n"
        f"  in:   {args.in_path}\n"
        f"  out:  {args.out_path}\n"
        f"  chunker={args.chunker} chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}\n"
        f"  input_col={args.input_col} output_col={args.output_col} only_types={args.only_types}\n"
        f"  nodes: {nodes_before} -> {nodes_after} (+{nodes_after - nodes_before})\n"
        f"  edges: {edges_before} -> {edges_after} (+{edges_after - edges_before})\n"
        f"  timing_s: load={load_s:.2f} run={run_s:.2f} save={save_s:.2f} total={load_s + run_s + save_s:.2f}"
    )


if __name__ == "__main__":
    main()

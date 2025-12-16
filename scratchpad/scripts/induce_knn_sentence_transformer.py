from __future__ import annotations

import argparse
import time
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Embed graph node text with sentence-transformers and induce sim:knn edges (top-k). "
            "For larger graphs, the kNN inducer uses Chroma/HNSW to avoid O(N^2) memory."
        )
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.with_chunk_mentions.json"),
        help="Input graph snapshot path (JSON).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("data/cms_knowledge_graph.chunked_markdown.with_chunk_mentions.with_knn10.json"),
        help="Output graph snapshot path (JSON).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")

    ap.add_argument("--edge-type", type=str, default="sim:knn", help="Edge type for kNN edges.")
    ap.add_argument("--k", type=int, default=10, help="Top-k neighbors per node.")
    ap.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    ap.add_argument("--device", type=str, default=None, help="Optional sentence-transformers device override.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings.")
    ap.add_argument("--embedding-key", type=str, default="embedding", help="Embedding storage key (default: embedding).")
    ap.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "incremental"],
        help="Induction run mode.",
    )
    ap.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Recompute embeddings even if they already exist.",
    )
    ap.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")

    args = ap.parse_args()

    if args.out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_path} (pass --overwrite)")

    t0 = time.perf_counter()
    graph = GraphDataset.load(args.in_path)
    load_s = time.perf_counter() - t0

    nodes = len(graph)
    edges_before = sum(1 for _ in graph.store.iter_edges())

    t1 = time.perf_counter()
    emb_stats = graph.ensure_sentence_transformer_embeddings(
        model_name=args.model_name,
        device=args.device,
        normalize=not args.no_normalize,
        embedding_key=args.embedding_key,
        overwrite=args.overwrite_embeddings,
        batch_size=args.batch_size,
    )
    embed_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    out = graph.run_knn_similarity_sentence_transformer_topk(
        edge_type=args.edge_type,
        k=args.k,
        model_name=args.model_name,
        device=args.device,
        normalize=not args.no_normalize,
        embedding_key=args.embedding_key,
        include_overlay=True,
        mode=args.mode,
        overwrite_embeddings=False,
        batch_size=args.batch_size,
    )
    induce_s = time.perf_counter() - t2

    edges_after = sum(1 for _ in graph.store.iter_edges())

    t3 = time.perf_counter()
    graph.save(args.out_path, include_overlay=True)
    save_s = time.perf_counter() - t3

    print(
        "Embedded + induced kNN edges and saved graph:\n"
        f"  in:   {args.in_path}\n"
        f"  out:  {args.out_path}\n"
        f"  nodes: {nodes}\n"
        f"  edge_type={args.edge_type} k={args.k}\n"
        f"  embeddings: {emb_stats}\n"
        f"  induced_edges_records: {len(out)}\n"
        f"  edges: {edges_before} -> {edges_after} (+{edges_after - edges_before})\n"
        f"  timing_s: load={load_s:.2f} embed={embed_s:.2f} induce={induce_s:.2f} save={save_s:.2f} total={load_s + embed_s + induce_s + save_s:.2f}"
    )


if __name__ == "__main__":
    main()

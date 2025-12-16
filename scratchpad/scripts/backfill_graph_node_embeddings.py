"""Backfill node embeddings into GraphDataset node attrs (Option B).

This script loads a graph snapshot, computes embeddings for each node's text using
`SentenceTransformerEmbeddingModel`, and writes the resulting vector to
`node.attrs[embedding_key]`.

Why attrs (Option B)?
- Supports multiple embedding variants (title/body/metadata/etc.) side-by-side.
- Works with kNN induction via `embedding_key`.

Example:
  /Users/jason/projects/mit/palimpzest/.venv/bin/python \
    scratchpad/scripts/backfill_graph_node_embeddings.py \
    --in-graph CURRENT_WORKSTREAM/exports/cms_graph.snapshot.json \
    --out-graph CURRENT_WORKSTREAM/exports/cms_graph.with_embeddings.json \
    --embedding-key embedding_st \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 64

Notes:
- Uses `graph.upsert_node(...)` so graph revision increments and incremental
  induction can detect changes.
- By default, node text is chosen using `default_node_text`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset
from palimpzest.core.data.graph_store import GraphNode
from palimpzest.graphrag.retrieval import (
    SentenceTransformerEmbeddingConfig,
    SentenceTransformerEmbeddingModel,
    default_node_text,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill graph node embeddings into attrs.")
    p.add_argument("--in-graph", required=True, help="Path to input GraphDataset snapshot JSON")
    p.add_argument("--out-graph", required=True, help="Path to write output snapshot JSON")
    p.add_argument(
        "--embedding-key",
        default="embedding_st",
        help="Node attrs key to store embeddings under (Option B)",
    )
    p.add_argument(
        "--model-name",
        default=SentenceTransformerEmbeddingConfig().model_name,
        help="SentenceTransformer model name",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing attrs[embedding_key] if present",
    )
    p.add_argument(
        "--include-overlay",
        action="store_true",
        help="Include overlay edges when saving (default: false)",
    )
    return p.parse_args()


def _batched(items: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> int:
    args = _parse_args()

    in_path = Path(args.in_graph)
    out_path = Path(args.out_graph)
    embedding_key: str = args.embedding_key
    overwrite: bool = bool(args.overwrite)

    graph = GraphDataset.load(in_path)

    model = SentenceTransformerEmbeddingModel(
        config=SentenceTransformerEmbeddingConfig(model_name=args.model_name)
    )

    to_embed: list[tuple[str, str]] = []
    skipped_empty = 0
    skipped_existing = 0

    # Snapshot iteration avoids surprises if the underlying store mutates while we upsert.
    snapshot = graph.to_snapshot()
    for node in snapshot.nodes:
        if not overwrite and embedding_key in (node.attrs or {}):
            skipped_existing += 1
            continue

        text = default_node_text(node)
        if not text:
            skipped_empty += 1
            continue

        to_embed.append((node.id, text))

    if not to_embed:
        print(
            json.dumps(
                {
                    "status": "no-op",
                    "graph_id": graph.graph_id,
                    "node_count": len(snapshot.nodes),
                    "skipped_empty": skipped_empty,
                    "skipped_existing": skipped_existing,
                    "embedded": 0,
                    "embedding_key": embedding_key,
                },
                indent=2,
                sort_keys=True,
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(out_path, include_overlay=bool(args.include_overlay))
        return 0

    embedded = 0
    batches = _batched(to_embed, args.batch_size)
    for batch_idx, batch in enumerate(batches, start=1):
        ids = [node_id for (node_id, _t) in batch]
        texts = [t for (_node_id, t) in batch]
        vectors = model.embed_texts(texts)
        if vectors.shape[0] != len(ids):
            raise ValueError("EmbeddingModel returned mismatched row count")

        for node_id, vec in zip(ids, vectors, strict=True):
            node = graph.get_node(node_id)
            new_attrs = dict(node.attrs or {})
            new_attrs[embedding_key] = [float(x) for x in vec.tolist()]
            updated: GraphNode = node.model_copy(update={"attrs": new_attrs})
            graph.upsert_node(updated)
            embedded += 1

        if batch_idx == 1 or batch_idx == len(batches) or batch_idx % 25 == 0:
            print(f"Embedded {embedded}/{len(to_embed)} nodes…")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(out_path, include_overlay=bool(args.include_overlay))

    print(
        json.dumps(
            {
                "status": "ok",
                "graph_id": graph.graph_id,
                "initial_revision": snapshot.revision,
                "final_revision": graph.revision,
                "node_count": len(snapshot.nodes),
                "skipped_empty": skipped_empty,
                "skipped_existing": skipped_existing,
                "embedded": embedded,
                "embedding_key": embedding_key,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "overwrite": overwrite,
                "out_graph": str(out_path),
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

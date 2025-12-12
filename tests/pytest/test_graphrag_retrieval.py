from __future__ import annotations

import numpy as np

from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode
from palimpzest.graphrag.retrieval import VectorIndex


class _ToyEmbedding:
    """Deterministic tiny embedding model for tests."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vocab = ["tier0", "data", "production", "reprocessing", "cmsdm"]
        rows: list[list[float]] = []
        for t in texts:
            tl = (t or "").lower()
            rows.append([1.0 if w in tl else 0.0 for w in vocab])
        arr = np.asarray(rows, dtype=np.float32)
        # normalize (cosine space)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms


def test_vector_index_search_picks_relevant_node() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="n1", label="A", text="Tier0 operations overview"))
    g.add_node(GraphNode(id="n2", label="B", text="Data management best practices"))
    g.add_node(GraphNode(id="n3", label="C", text="CMSDM project"))

    ix = VectorIndex(graph=g, embedding_model=_ToyEmbedding())
    hits = ix.search(query="tier0", k=2)

    assert hits
    assert hits[0][0] == "n1"
    assert hits[0][1] >= hits[-1][1]


def test_vector_index_search_empty_query_returns_empty() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="n1", label="A", text="hello"))

    ix = VectorIndex(graph=g, embedding_model=_ToyEmbedding())
    assert ix.search(query="", k=5) == []

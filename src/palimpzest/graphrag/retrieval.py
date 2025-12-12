from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode


class EmbeddingModel(Protocol):
    """Embeds texts into a dense vector space."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a 2D array of shape (len(texts), dim)."""


@dataclass(frozen=True)
class SentenceTransformerEmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    normalize: bool = True


class SentenceTransformerEmbeddingModel:
    def __init__(self, *, config: SentenceTransformerEmbeddingConfig | None = None) -> None:
        self.config = config or SentenceTransformerEmbeddingConfig()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbeddingModel. "
                "Install with `pip install sentence-transformers`."
            ) from e

        self._model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        arr = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        out = np.asarray(arr, dtype=np.float32)
        if self.config.normalize:
            out = _l2_normalize(out)
        return out


@dataclass(frozen=True)
class OpenAIEmbeddingConfig:
    model_name: str = "text-embedding-3-small"


class OpenAIEmbeddingModel:
    def __init__(self, *, config: OpenAIEmbeddingConfig | None = None) -> None:
        self.config = config or OpenAIEmbeddingConfig()
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("openai package is required for OpenAIEmbeddingModel") from e

        self._client = OpenAI()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        # OpenAI supports batch embedding.
        resp = self._client.embeddings.create(model=self.config.model_name, input=texts)
        vectors: list[list[float]] = [d.embedding for d in resp.data]
        out = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(out)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def default_node_text(node: GraphNode) -> str:
    """Best-effort text used for embeddings/reranking."""

    if isinstance(node.text, str) and node.text.strip():
        return node.text.strip()
    if isinstance(node.label, str) and node.label.strip():
        return node.label.strip()
    md = (node.attrs or {}).get("metadata")
    if isinstance(md, dict):
        name = md.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return ""


class VectorIndex:
    """In-memory cosine-similarity search over graph nodes."""

    def __init__(
        self,
        *,
        graph: GraphDataset,
        embedding_model: EmbeddingModel,
        node_text_fn=default_node_text,
    ) -> None:
        self._graph = graph
        self._embedding_model = embedding_model
        self._node_ids: list[str] = []
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None

        node_ids: list[str] = []
        texts: list[str] = []
        for node in graph.to_snapshot().nodes:
            node_id = node.id
            txt = node_text_fn(node)
            if not txt:
                continue
            node_ids.append(node_id)
            texts.append(txt)

        self._node_ids = node_ids
        self._texts = texts
        # Build once.
        self._embeddings = embedding_model.embed_texts(texts)
        if self._embeddings.shape[0] != len(self._node_ids):
            raise ValueError("EmbeddingModel returned mismatched row count")

    @property
    def size(self) -> int:
        return len(self._node_ids)

    def search(self, *, query: str, k: int) -> list[tuple[str, float]]:
        q = (query or "").strip()
        if not q:
            return []
        if k <= 0:
            return []
        if self._embeddings is None or self._embeddings.size == 0:
            return []

        q_emb = self._embedding_model.embed_texts([q])
        if q_emb.shape[0] != 1:
            raise ValueError("EmbeddingModel returned invalid query embedding")

        # cosine sim because embeddings are normalized
        sims = (self._embeddings @ q_emb[0]).astype(np.float32)
        k = min(int(k), sims.shape[0])
        if k <= 0:
            return []

        # partial top-k
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        # sort by score desc
        idx = idx[np.argsort(-sims[idx])]
        return [(self._node_ids[i], float(sims[i])) for i in idx]


class Reranker(Protocol):
    def score(self, *, query: str, docs: list[str]) -> list[float]:
        """Return relevance scores aligned to docs (higher is better)."""


@dataclass(frozen=True)
class HFRerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    device: str | None = None
    max_length: int = 512
    local_files_only: bool = True
    trust_remote_code: bool = False


class HFReranker:
    """HuggingFace cross-encoder reranker.

    This requires `transformers` + `torch`.
    """

    def __init__(self, *, config: HFRerankerConfig | None = None) -> None:
        self.config = config or HFRerankerConfig()

        # Prefer sentence-transformers CrossEncoder when available (this matches how
        # many HF rerankers are intended to be used).
        self._cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            # NOTE: CrossEncoder doesn't expose local_files_only directly; it forwards
            # to transformers under the hood. This may still download unless the model
            # is already present in the HF cache.
            self._cross_encoder = CrossEncoder(
                self.config.model_name,
                device=self.config.device or "cpu",
                trust_remote_code=self.config.trust_remote_code,
            )
            return
        except Exception:
            self._cross_encoder = None

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "transformers+torch are required for HFReranker. "
                "Install with `pip install transformers torch`."
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.device:
            self._model.to(self.config.device)
        self._model.eval()

    def score(self, *, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []

        if self._cross_encoder is not None:
            pairs = [(query, d) for d in docs]
            scores = self._cross_encoder.predict(pairs)
            return [float(s) for s in scores]

        tok = self._tokenizer(
            [query] * len(docs),
            docs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        if self.config.device:
            tok = {k: v.to(self.config.device) for k, v in tok.items()}

        with self._torch.no_grad():
            out = self._model(**tok)
            logits = out.logits

        # Common rerankers output a single logit per pair; some output multi-class logits.
        if logits.ndim == 2:
            if logits.shape[1] == 1:
                logits = logits[:, 0]
            else:
                # Heuristic: treat the last logit as "more relevant".
                logits = logits[:, -1]

        scores = logits.detach().float().cpu().numpy().tolist()
        return [float(s) for s in scores]


def stable_softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

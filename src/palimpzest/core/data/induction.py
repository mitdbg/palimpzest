from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from palimpzest.utils.hash_helpers import hash_for_id


class CandidateGenerator(Protocol):
    def generator_id(self) -> str: ...

    def generate_pairs(
        self,
        *,
        node_ids: list[str],
        impacted_node_ids: set[str],
        score_pair: ScorePairFn,
    ) -> Iterable[tuple[str, str]]: ...


class Decider(Protocol):
    def decider_id(self) -> str: ...

    def score_pair(self, *, src_id: str, dst_id: str) -> float: ...


class ScorePairFn(Protocol):
    def __call__(self, *, src_id: str, dst_id: str) -> float: ...


class CandidateGeneratorSpec(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class DeciderSpec(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class InductionSpec(BaseModel):
    """Serializable, persisted induction spec.

    Backwards compatibility:
    - Accepts legacy payloads with top-level `kind` + `params` by converting them into generator/decider specs.
    """

    schema_version: int = 1
    edge_type: str
    include_overlay: bool = True
    symmetric: bool = True
    allow_self_edges: bool = False
    overwrite: bool = False
    generator: CandidateGeneratorSpec
    decider: DeciderSpec

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "generator" in data and "decider" in data:
            return data

        # Legacy: {kind, edge_type, params, ...}
        kind = data.get("kind")
        params = data.get("params", {}) or {}
        edge_type = data.get("edge_type")
        include_overlay = data.get("include_overlay", True)

        if kind == "knn_similarity":
            gen = {
                "kind": "knn_bruteforce",
                "params": {
                    "embedding_key": params.get("embedding_key", "embedding"),
                    "k": params.get("k", 10),
                    "threshold": params.get("threshold"),
                },
            }
            dec = {"kind": "cosine_similarity", "params": {"embedding_key": params.get("embedding_key", "embedding")}}
            return {
                "schema_version": 1,
                "edge_type": edge_type,
                "include_overlay": include_overlay,
                "symmetric": True,
                "allow_self_edges": False,
                "overwrite": False,
                "generator": gen,
                "decider": dec,
            }

        # Legacy: text_mentions (from earlier iterations) -> predicate induction
        if kind == "text_mentions":
            gen = {"kind": "text_anchor", "params": data.get("params", {}) or {}}
            dec = {
                "kind": "predicate",
                "params": {
                    "kind": "text_contains",
                    "params": {"source_field": "text", "target_fields": ["label", "attrs.name", "attrs.metadata.name"], "boundaries": True},
                },
            }
            return {
                "schema_version": 1,
                "edge_type": edge_type,
                "include_overlay": include_overlay,
                "symmetric": False,
                "allow_self_edges": False,
                "overwrite": False,
                "generator": gen,
                "decider": dec,
            }

        return data

    def spec_id(self) -> str:
        payload = self.model_dump(mode="json")
        return hash_for_id(json.dumps(payload, sort_keys=True))


class InductionLogEntry(BaseModel):
    spec: InductionSpec
    processed_node_ids: list[str] = Field(default_factory=list, description="Nodes covered by the last successful run.")


class InductionLog(BaseModel):
    entries: list[InductionLogEntry] = Field(default_factory=list)

    def get(self, spec_id: str) -> InductionLogEntry | None:
        for entry in self.entries:
            if entry.spec.spec_id() == spec_id:
                return entry
        return None

    def upsert(self, entry: InductionLogEntry) -> None:
        spec_id = entry.spec.spec_id()
        for i, existing in enumerate(self.entries):
            if existing.spec.spec_id() == spec_id:
                self.entries[i] = entry
                return
        self.entries.append(entry)


class KnnBruteForceCandidateGenerator:
    def __init__(
        self,
        *,
        k: int | None,
        threshold: float | None,
        symmetric: bool,
        allow_self_edges: bool,
    ) -> None:
        if (k is None) == (threshold is None):
            raise ValueError("kNN candidate generation requires exactly one of: k, threshold")
        if k is not None and k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.threshold = threshold
        self.symmetric = symmetric
        self.allow_self_edges = allow_self_edges

    def generator_id(self) -> str:
        payload = {
            "kind": "knn_bruteforce",
            "k": self.k,
            "threshold": self.threshold,
            "symmetric": self.symmetric,
            "allow_self_edges": self.allow_self_edges,
        }
        return hash_for_id(json.dumps(payload, sort_keys=True))

    def generate_pairs(
        self,
        *,
        node_ids: list[str],
        impacted_node_ids: set[str],
        score_pair: ScorePairFn,
    ) -> Iterable[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()

        for u in impacted_node_ids:
            scored: list[tuple[float, str]] = []
            for v in node_ids:
                if not self.allow_self_edges and v == u:
                    continue
                score = score_pair(src_id=u, dst_id=v)
                if self.threshold is not None:
                    if score >= self.threshold:
                        scored.append((score, v))
                else:
                    scored.append((score, v))

            scored.sort(reverse=True, key=lambda t: t[0])
            if self.k is not None:
                scored = scored[: self.k]

            for _score, v in scored:
                if not self.allow_self_edges and u == v:
                    continue
                pairs.add((u, v))
                if self.symmetric:
                    pairs.add((v, u))

        return sorted(pairs)


class CosineSimilarityDecider:
    def __init__(self, *, embedding_for_node_id: callable) -> None:
        self._embedding_for_node_id = embedding_for_node_id

    def decider_id(self) -> str:
        # The embedding provider is not serializable; the spec should capture embedding_key/model.
        return "cosine_similarity"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b, strict=True):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na**0.5) * (nb**0.5))

    def score_pair(self, *, src_id: str, dst_id: str) -> float:
        return self._cosine_similarity(self._embedding_for_node_id(src_id), self._embedding_for_node_id(dst_id))

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from palimpzest.core.data.iter_dataset import IterDataset
from palimpzest.core.lib.schemas import create_schema_from_fields


class DictListDataset(IterDataset):
    """A small, pandas-free in-memory dataset for list[dict] values."""

    def __init__(self, *, id: str, vals: list[dict[str, Any]], schema: type[BaseModel] | list[dict] | None = None) -> None:
        self.vals = vals
        if schema is None:
            schema = create_schema_from_fields([{"name": k, "type": object, "description": k} for k in sorted({k for r in vals for k in r})])
        super().__init__(id=id, schema=schema)

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.vals[idx])


class NodePairListDataset(IterDataset):
    """Indexable dataset over explicit (src_id, dst_id) pairs without materializing list[dict]."""

    def __init__(
        self,
        *,
        id: str,
        pairs: list[tuple[str, str]],
        schema: type[BaseModel] | list[dict] | None = None,
    ) -> None:
        self.pairs = pairs
        if schema is None:
            schema = [
                {"name": "src_node_id", "type": str, "description": "Source node id"},
                {"name": "dst_node_id", "type": str, "description": "Destination node id"},
            ]
        super().__init__(id=id, schema=schema)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src, dst = self.pairs[idx]
        return {"src_node_id": src, "dst_node_id": dst}


class AllPairsNodePairDataset(IterDataset):
    """Indexable, O(1)-memory candidate generator for src_ids × dst_ids (optionally excluding self edges)."""

    def __init__(
        self,
        *,
        id: str,
        src_ids: list[str],
        dst_ids: list[str],
        allow_self_edges: bool,
        schema: type[BaseModel] | list[dict] | None = None,
    ) -> None:
        self.src_ids = src_ids
        self.dst_ids = dst_ids
        self.allow_self_edges = bool(allow_self_edges)
        self._dst_index_by_id = {n: i for i, n in enumerate(dst_ids)}
        if not self.allow_self_edges:
            missing = [s for s in src_ids if s not in self._dst_index_by_id]
            if missing:
                raise ValueError(f"AllPairsNodePairDataset requires all src_ids to exist in dst_ids when allow_self_edges=False; missing={missing[:5]}")
        if schema is None:
            schema = [
                {"name": "src_node_id", "type": str, "description": "Source node id"},
                {"name": "dst_node_id", "type": str, "description": "Destination node id"},
            ]
        super().__init__(id=id, schema=schema)

    def __len__(self) -> int:
        n_dst = len(self.dst_ids)
        per_src = n_dst if self.allow_self_edges else max(0, n_dst - 1)
        return len(self.src_ids) * per_src

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        n_dst = len(self.dst_ids)
        per_src = n_dst if self.allow_self_edges else n_dst - 1
        src_pos = idx // per_src
        off = idx % per_src
        src_id = self.src_ids[src_pos]

        if self.allow_self_edges:
            dst_id = self.dst_ids[off]
            return {"src_node_id": src_id, "dst_node_id": dst_id}

        self_pos = self._dst_index_by_id[src_id]
        dst_pos = off + 1 if off >= self_pos else off
        dst_id = self.dst_ids[dst_pos]
        return {"src_node_id": src_id, "dst_node_id": dst_id}


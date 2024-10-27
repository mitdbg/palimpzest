import hashlib
from typing import Any, Type

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.corelib.schemas import Schema


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""

    def __init__(
        self,
        schema: Type[Schema],
        parent_id: str | None = None,
        scan_idx: int | None = None,
        cardinality_idx: int | None = None,
    ):
        # schema for the data record
        self.schema = schema
        # dynamic properties
        self._data = {}

        # NOTE: Record ids are hashed based on:
        # 0. their schema (keys)
        # 1. their parent record id(s) (or scan_idx if there is no parent record)
        # 2. their index in the fan out (if this is in a one-to-many operation)
        #
        # We currently do NOT hash just based on record content (i.e. schema (key, value) pairs)
        # because multiple outputs for a given operation may have the exact same
        # schema (key, value) pairs.
        #
        # We may revisit this hashing scheme in the future.

        # unique identifier for the record
        id_str = (
            str(schema) + (parent_id if parent_id is not None else str(scan_idx))
            if cardinality_idx is None
            else str(schema) + str(cardinality_idx) + (parent_id if parent_id is not None else str(scan_idx))
        )
        self._id = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]
        self._parent_id = parent_id

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ["schema", "_data"]:
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            pass
        elif name in self._data:
            return self._data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _as_json_str(self, include_bytes: bool = True, *args, **kwargs):
        """Return a JSON representation of this DataRecord"""
        record_dict = self._as_dict(include_bytes)
        return self.schema().as_json_str(record_dict, *args, **kwargs)

    def _as_dict(self, include_bytes: bool = True):
        """Return a dictionary representation of this DataRecord"""
        dct = self._data.copy()
        if not include_bytes:
            for k, v in dct.items():
                if isinstance(v, bytes) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], bytes)):
                    dct[k] = "<bytes>"
        return dct

    def __str__(self):
        items = (f"{k}={str(v)[:15]!r}..." for k, v in sorted(self._data.items()))
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return isinstance(other, DataRecord) and self._data == other._data and self.schema == other.schema

    def get_fields(self):
        return list(self._data.keys())

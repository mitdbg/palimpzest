import hashlib
from typing import Type

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.corelib import Schema


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
            str(schema)
            + (parent_id if parent_id is not None else str(scan_idx))
            if cardinality_idx is None
            else str(schema)
            + str(cardinality_idx)
            + (parent_id if parent_id is not None else str(scan_idx))
        )
        self._id = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[
            :MAX_ID_CHARS
        ]
        self._parent_id = parent_id

    def _asJSONStr(self, include_bytes: bool = True, *args, **kwargs):
        """Return a JSON representation of this DataRecord"""
        record_dict = self._asDict(include_bytes)
        return self.schema().asJSONStr(record_dict, *args, **kwargs)

    def _asDict(self, include_bytes: bool = True):
        """Return a dictionary representation of this DataRecord"""
        dct = {k: self.__dict__[k] for k in self._getFields()}
        if not include_bytes:
            for k in dct:
                if isinstance(dct[k], bytes) or (
                    isinstance(dct[k], list)
                    and len(dct[k]) > 0
                    and isinstance(dct[k][0], bytes)
                ):
                    dct[k] = "<bytes>"
        return dct

    def __str__(self):
        keys = sorted(self.__dict__.keys())
        items = (
            "{}={!r}...".format(k, str(self.__dict__[k])[:15]) for k in keys
        )
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    # NOTE: the method is called _getFields instead of getFields to avoid it being picked up as a data record attribute;
    #       in the future we will come up with a less ugly fix -- but for now do not remove the _ even though it's not private
    def _getFields(self):
        return [
            k
            for k in self.__dict__.keys()
            if not k.startswith("_") and k != "schema"
        ]

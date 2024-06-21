from palimpzest.corelib import Schema

import hashlib

# DEFINITIONS
MAX_UUID_CHARS = 10


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""

    def __init__(
        self,
        schema: Schema,
        parent_uuid: str = None,
        scan_idx: int = None,
        cardinality_idx: int = None,
    ):
        # schema for the data record
        self.schema = schema

        # TODO: this uuid should be a hash of the parent_uuid and/or the record index in the current operator
        #       this way we can compare records across plans (e.g. for determining majority answer when gathering
        #       samples from plans in parallel)
        # unique identifier for the record
        # self._uuid = str(uuid.uuid4())[:MAX_UUID_CHARS]
        uuid_str = (
            str(schema) + (parent_uuid if parent_uuid is not None else str(scan_idx))
            if cardinality_idx is None
            else str(schema)
            + str(cardinality_idx)
            + (parent_uuid if parent_uuid is not None else str(scan_idx))
        )
        self._uuid = hashlib.sha256(uuid_str.encode("utf-8")).hexdigest()[
            :MAX_UUID_CHARS
        ]
        self._parent_uuid = parent_uuid

    def __getitem__(self, key):
        return super().__getattr__(key)

    def _asJSONStr(self, include_bytes: bool = True, *args, **kwargs):
        """Return a JSON representation of this DataRecord"""
        record_dict = self._asDict(include_bytes)
        return self.schema().asJSONStr(record_dict, *args, **kwargs)

    def _asDict(self, include_bytes: bool = True):
        """Return a dictionary representation of this DataRecord"""
        dct = {k: self.__dict__[k] for k in self.getFields()}
        if not include_bytes:
            for k in dct:
                if isinstance(dct[k], bytes) or (
                    isinstance(dct[k], list) and isinstance(dct[k][0], bytes)
                ):
                    dct[k] = "<bytes>"
        return dct

    def __str__(self):
        keys = sorted(self.__dict__.keys())
        items = ("{}={!r}...".format(k, str(self.__dict__[k])[:15]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def getFields(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_") and k != "schema"]

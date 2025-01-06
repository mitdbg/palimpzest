from __future__ import annotations

import hashlib
from typing import Any

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.core.lib.schemas import Schema
from palimpzest.core.data.dataclasses import RecordOpStats


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""

    def __init__(
        self,
        schema: Schema,
        source_id: int | str,
        parent_id: str | None = None,
        cardinality_idx: int | None = None,
    ):
        # check that source_id is provided
        assert source_id is not None, "Every DataRecord must be constructed with a source_id"

        # schema for the data record
        self.schema = schema

        # dynamic properties
        self._data = {}

        # the source record(s) from which this DataRecord is derived
        self._source_id = str(source_id)

        # the id of the parent record(s) from which this DataRecord is derived
        self._parent_id = parent_id

        # store the cardinality index
        self._cardinality_idx = cardinality_idx

        # NOTE: Record ids are hashed based on:
        # 0. their schema (keys)
        # 1. their parent record id(s) (or source_id if there is no parent record)
        # 2. their index in the fan out (if this is in a one-to-many operation)
        #
        # We currently do NOT hash just based on record content (i.e. schema (key, value) pairs)
        # because multiple outputs for a given operation may have the exact same
        # schema (key, value) pairs.
        #
        # We may revisit this hashing scheme in the future.

        # unique identifier for the record
        id_str = (
            str(schema) + (parent_id if parent_id is not None else self._source_id)
            if cardinality_idx is None
            else str(schema) + str(cardinality_idx) + (parent_id if parent_id is not None else self._source_id)
        )
        self._id = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]


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


    def __getitem__(self, key):
        return self.__getattr__(key)


    def __str__(self):
        items = (f"{k}={str(v)[:15]!r}..." for k, v in sorted(self._data.items()))
        return "{}({})".format(type(self).__name__, ", ".join(items))


    def __eq__(self, other):
        return isinstance(other, DataRecord) and self._data == other._data and self.schema == other.schema


    def __hash__(self):
        return hash(self.as_json_str())


    def _copy(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        # make new record which has parent_record as its parent (and the same source_id)
        new_dr = DataRecord(
            self.schema,
            source_id=self._source_id,
            parent_id=self._id,
            cardinality_idx=self._cardinality_idx,
        )

        # get the set of fields to copy from the parent record
        copy_fields = project_cols if project_cols is not None else self.get_fields()
        copy_fields = [field.split(".")[-1] for field in copy_fields]

        # copy fields from the parent
        for field in copy_fields:
            field_value = getattr(self, field)
            if (
                not include_bytes
                and isinstance(field_value, bytes)
                or (isinstance(field_value, list) and len(field_value) > 0 and isinstance(field_value[0], bytes))
            ):
                continue

            # set attribute
            setattr(new_dr, field, field_value)

        return new_dr


    @staticmethod
    def from_parent(
        schema: Schema,
        parent_record: DataRecord,
        project_cols: list[str] | None = None,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # make new record which has parent_record as its parent (and the same source_id)
        new_dr = DataRecord(
            schema,
            source_id=parent_record._source_id,
            parent_id=parent_record._id,
            cardinality_idx=cardinality_idx,
        )

        # get the set of fields to copy from the parent record
        copy_fields = project_cols if project_cols is not None else parent_record.get_fields()

        # copy fields from the parent
        for field in copy_fields:
            setattr(new_dr, field, getattr(parent_record, field))

        return new_dr


    @staticmethod
    def from_agg_parents(
        schema: Schema,
        parent_records: DataRecordSet,
        project_cols: list[str] | None = None,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # TODO: we can implement this once we support having multiple parent ids
        pass


    @staticmethod
    def from_join_parents(
        left_schema: Schema,
        right_schema: Schema,
        left_parent_record: DataRecord,
        right_parent_record: DataRecord,
        project_cols: list[str] | None = None,
        cardinality_idx: int = None,
    ) -> DataRecord:
        # TODO: we can implement this method if/when we add joins
        pass


    def as_json_str(self, include_bytes: bool = True, project_cols: list[str] | None = None, *args, **kwargs):
        """Return a JSON representation of this DataRecord"""
        record_dict = self.as_dict(include_bytes, project_cols)
        return self.schema().as_json_str(record_dict, *args, **kwargs)


    def as_dict(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        """Return a dictionary representation of this DataRecord"""
        dct = self._data.copy()

        if project_cols is not None and len(project_cols) > 0:
            project_fields = set(field.split(".")[-1] for field in project_cols)
            dct = {k: v for k, v in dct.items() if k in project_fields}

        if not include_bytes:
            for k, v in dct.items():
                if isinstance(v, bytes) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], bytes)):
                    dct[k] = "<bytes>"
        return dct


    def get_fields(self):
        return list(self._data.keys())


class DataRecordSet:
    """
    A DataRecordSet contains a list of DataRecords and a reference to the parent_id
    and source_id that these records were derived from. It also contains the RecordOpStats
    associated with generating these DataRecords.

    Thus, there is an assumption that a DataRecordSet consists of the output from
    executing a single operator on a single input record.

    We explicitly check that this is True, by making sure that the records passed into
    the DataRecordSet all share the same parent_id.
    """
    def __init__(self, data_records: list[DataRecord], record_op_stats: list[RecordOpStats]):
        # check that all data_records are derived from the same parent record
        if len(data_records) > 0:
            parent_id = data_records[0]._parent_id
            error_msg = "DataRecordSet must be constructed from the output of executing a single operator on a single input."
            assert all([dr._parent_id == parent_id for dr in data_records]), error_msg

        # set data_records, parent_id, and source_id; note that it is possible for
        # data_records to be an empty list in the event of a failed convert
        self.data_records = data_records
        self.parent_id = data_records[0]._parent_id if len(data_records) > 0 else None
        self.source_id = data_records[0]._source_id if len(data_records) > 0 else None

        # set statistics for generating these records
        self.record_op_stats = record_op_stats


    def __getitem__(self, slice):
        return self.data_records[slice]


    def __len__(self):
        return len(self.data_records)


    def __iter__(self):
        yield from self.data_records

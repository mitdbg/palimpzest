from __future__ import annotations

import hashlib
import json
from typing import Any

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.corelib.fields import Field
from palimpzest.corelib.schemas import Schema
from palimpzest.dataclasses import RecordOpStats


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

        # mapping from field names to Field objects; effectively a mapping from a field name to its type
        self.field_types: dict[str, Field] = {}

        # mapping from field names to their values
        self.field_values: dict[str, Any] = {}

        # the source record(s) from which this DataRecord is derived
        self.source_id = str(source_id)

        # the id of the parent record(s) from which this DataRecord is derived
        self.parent_id = parent_id

        # store the cardinality index
        self.cardinality_idx = cardinality_idx

        # indicator variable which may be flipped by filter operations to signal when a record has been filtered out
        self.passed_operator = True

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
            str(schema) + (parent_id if parent_id is not None else self.source_id)
            if cardinality_idx is None
            else str(schema) + str(cardinality_idx) + (parent_id if parent_id is not None else self.source_id)
        )
        self.id = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]


    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ["schema", "field_values"]:
            super().__setattr__(name, value)
        else:
            self.field_values[name] = value


    def __getattr__(self, name: str) -> Any:
        if name == "field_values":
            pass
        elif name in self.field_values:
            return self.field_values[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def __getitem__(self, field: str) -> Any:
        return self.__getattr__(field)


    def __setitem__(self, field: str, value: Any) -> None:
        self.__setattr__(field, value)


    def __str__(self):
        items = (f"{k}={str(v)[:15]!r}..." for k, v in sorted(self.field_values.items()))
        return "{}({})".format(type(self).__name__, ", ".join(items))


    def __eq__(self, other):
        return isinstance(other, DataRecord) and self.field_values == other.field_values and self.schema == other.schema


    def __hash__(self):
        return hash(self.as_json_str())


    def __iter__(self):
        yield from self.field_values.items()


    def get_field_names(self):
        return list(self.field_types.keys())


    def get_field_type(self, field_name: str) -> Field:
        return self.field_types[field_name]


    def copy(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        # make new record which has parent_record as its parent (and the same source_id)
        new_dr = DataRecord(
            self.schema,
            source_id=self.source_id,
            parent_id=self.id,
            cardinality_idx=self.cardinality_idx,
        )

        # get the set of fields to copy from the parent record
        copy_field_names = project_cols if project_cols is not None else self.get_field_names()
        copy_field_names = [field.split(".")[-1] for field in copy_field_names]

        # copy field types and values from the parent
        for field_name in copy_field_names:
            field_type = self.get_field_type(field_name)
            field_value = self[field_name]
            if (
                not include_bytes
                and isinstance(field_value, bytes)
                or (isinstance(field_value, list) and len(field_value) > 0 and isinstance(field_value[0], bytes))
            ):
                continue

            # set field and value
            new_dr.field_types[field_name] = field_type
            new_dr[field_name] = field_value

        return new_dr


    @staticmethod
    def from_parent(
        schema: Schema,
        parent_record: DataRecord,
        project_cols: list[str] | None = None,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # project_cols must be None or contain at least one column
        assert project_cols is None or len(project_cols) > 1, "must have at least one column if using projection"

        # if project_cols is None, then the new schema is a union of the provided schema and parent_record.schema;
        # otherwise, it's a ProjectSchema
        new_schema = schema.union(parent_record.schema)
        if project_cols is not None:
            new_schema = new_schema.project(project_cols)

        # make new record which has parent_record as its parent (and the same source_id)
        new_dr = DataRecord(
            new_schema,
            source_id=parent_record.source_id,
            parent_id=parent_record.id,
            cardinality_idx=cardinality_idx,
        )

        # get the set of fields and field descriptions to copy from the parent record
        copy_field_names = project_cols if project_cols is not None else parent_record.get_field_names()
        copy_field_names = [field.split(".")[-1] for field in copy_field_names]

        # copy fields from the parent
        for field_name in copy_field_names:
            new_dr.field_types[field_name] = parent_record.get_field_type(field_name)
            new_dr[field_name] = parent_record[field_name]

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


    def as_json_str(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        """Return a JSON representation of this DataRecord"""
        record_dict = self.as_dict(include_bytes, project_cols)
        record_dict = {
            field_name: self.schema.field_to_json(field_name, field_value)
            for field_name, field_value in self.field_values.items()
        }
        return json.dumps(record_dict, indent=2)


    def as_dict(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        """Return a dictionary representation of this DataRecord"""
        dct = self.field_values.copy()

        if project_cols is not None and len(project_cols) > 0:
            project_field_names = set(field.split(".")[-1] for field in project_cols)
            dct = {k: v for k, v in dct.items() if k in project_field_names}

        if not include_bytes:
            for k, v in dct.items():
                if isinstance(v, bytes) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], bytes)):
                    dct[k] = "<bytes>"

        return dct


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
            parent_id = data_records[0].parent_id
            error_msg = "DataRecordSet must be constructed from the output of executing a single operator on a single input."
            assert all([dr.parent_id == parent_id for dr in data_records]), error_msg

        # set data_records, parent_id, and source_id; note that it is possible for
        # data_records to be an empty list in the event of a failed convert
        self.data_records = data_records
        self.parent_id = data_records[0].parent_id if len(data_records) > 0 else None
        self.source_id = data_records[0].source_id if len(data_records) > 0 else None

        # set statistics for generating these records
        self.record_op_stats = record_op_stats


    def __getitem__(self, slice):
        return self.data_records[slice]


    def __len__(self):
        return len(self.data_records)


    def __iter__(self):
        yield from self.data_records

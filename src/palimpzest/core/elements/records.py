from __future__ import annotations

import json
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from palimpzest.core.data import context
from palimpzest.core.lib.schemas import (
    AUDIO_FIELD_TYPES,
    IMAGE_FIELD_TYPES,
    AudioBase64,
    AudioFilepath,
    ImageBase64,
    ImageFilepath,
    ImageURL,
    project,
    union_schemas,
)
from palimpzest.core.models import ExecutionStats, PlanStats, RecordOpStats
from palimpzest.utils.hash_helpers import hash_for_id


class DataRecord:
    """A DataRecord is a single record of data matching some schema defined by a BaseModel."""

    def __init__(
        self,
        data_item: BaseModel,
        source_indices: str | int | list[str | int],
        parent_ids: str | list[str] | None = None,
        cardinality_idx: int | None = None,
    ):
        # check that source_indices are provided
        assert source_indices is not None, "Every DataRecord must be constructed with source index (or indices)"

        # normalize to list[str]
        if not isinstance(source_indices, list):
            source_indices = [source_indices]

        # normalize to list[str]
        if isinstance(parent_ids, str):
            parent_ids = [parent_ids]

        # data for the data record
        self._data_item = data_item

        # the index in the root Dataset from which this DataRecord is derived;
        # each source index takes the form: f"{root_dataset.id}-{idx}"
        self._source_indices = sorted(source_indices)

        # the id(s) of the parent record(s) from which this DataRecord is derived
        self._parent_ids = parent_ids

        # store the cardinality index
        self._cardinality_idx = cardinality_idx

        # indicator variable which may be flipped by filter operations to signal when a record has been filtered out
        self._passed_operator = True

        # NOTE: Record ids are hashed based on:
        # 0. their schema (keys)
        # 1. their parent record id(s) (or source_indices if there is no parent record)
        # 2. their index in the fan out (if this is in a one-to-many operation)
        #
        # We currently do NOT hash just based on record content (i.e. schema (key, value) pairs)
        # because multiple outputs for a given operation may have the exact same
        # schema (key, value) pairs.
        #
        # We may revisit this hashing scheme in the future.

        # unique identifier for the record
        schema_fields = sorted(list(type(data_item).model_fields))
        id_str = (
            str(schema_fields) + str(parent_ids) if parent_ids is not None else str(self._source_indices)
            if cardinality_idx is None
            else str(schema_fields) + str(cardinality_idx) + str(parent_ids) if parent_ids is not None else str(self._source_indices)
        )
        self._id = hash_for_id(id_str)


    # TODO: raise an exception if one of these fields is present in the schema
    # - put these in a constant list up top
    # - import the constant list in Dataset (if possible) and check at plan creation time
    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ["_data_item", "_source_indices", "_parent_ids", "_cardinality_idx", "_passed_operator", "_id"]:
            super().__setattr__(name, value)
        else:
            setattr(self._data_item, name, value)


    def __getattr__(self, name: str) -> Any:
        return getattr(self._data_item, name)


    def __getitem__(self, field: str) -> Any:
        return getattr(self._data_item, field)


    def __setitem__(self, field: str, value: Any) -> None:
        setattr(self._data_item, field, value)


    def __str__(self, truncate: int | None = 15) -> str:
        if truncate is not None:
            items = (f"{k}={str(v)[:truncate]!r}{'...' if len(str(v)) > truncate else ''}" for k, v in sorted(self._data_item.model_dump().items()))
        else:
            items = (f"{k}={v!r}" for k, v in sorted(self._data_item.model_dump().items()))
        return "{}({})".format(type(self).__name__, ", ".join(items))


    def __repr__(self) -> str:
        return self.__str__(truncate=None)


    def __eq__(self, other):
        return isinstance(other, DataRecord) and self._data_item == other._data_item


    def __hash__(self):
        return hash(self.to_json_str(bytes_to_str=True, sorted=True))


    def __iter__(self):
        yield from self._data_item.__iter__()


    def get_field_names(self):
        return list(type(self._data_item).model_fields.keys())


    def get_field_type(self, field_name: str) -> FieldInfo:
        return type(self._data_item).model_fields[field_name]

    @property
    def schema(self) -> type[BaseModel]:
        return type(self._data_item)

    def copy(self) -> DataRecord:
        # get the set of fields to copy from the parent record
        copy_field_names = [field.split(".")[-1] for field in self.get_field_names()]

        # copy field types and values from the parent
        data_item = {field_name: self[field_name] for field_name in copy_field_names}

        # make copy of the current record
        new_dr = DataRecord(
            self.schema(**data_item),
            source_indices=self._source_indices,
            parent_ids=self._parent_ids,
            cardinality_idx=self._cardinality_idx,
        )

        # copy the passed_operator attribute
        new_dr._passed_operator = self._passed_operator

        return new_dr

    @staticmethod
    def from_parent(
        schema: type[BaseModel],
        data_item: dict,
        parent_record: DataRecord,
        project_cols: list[str] | None = None,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # if project_cols is None, then the new schema is a union of the provided schema and parent_record.schema;
        # if project_cols is an empty list, then the new schema is simply the provided schema
        # otherwise, it's a ProjectSchema
        new_schema = None
        if project_cols is None:
            new_schema = union_schemas([schema, parent_record.schema])
        elif project_cols == []:
            new_schema = schema
        else:
            new_schema = union_schemas([schema, parent_record.schema])
            new_schema = project(new_schema, project_cols)

        # get the set of fields and field descriptions to copy from the parent record
        copy_field_names = parent_record.get_field_names() if project_cols is None else project_cols
        copy_field_names = [field.split(".")[-1] for field in copy_field_names]

        # copy fields from the parent
        data_item.update({field_name: parent_record[field_name] for field_name in copy_field_names})

        # corner-case: wrap values in lists if the new schema expects a list but the data item has a single value
        for field_name, field_info in new_schema.model_fields.items():
            field_should_be_list = hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ is list
            field_is_not_list = field_name in data_item and not isinstance(data_item[field_name], list)
            if field_should_be_list and field_is_not_list:
                data_item[field_name] = [data_item[field_name]]

        # make new record which has parent_record as its parent (and the same source_indices)
        new_dr = DataRecord(
            new_schema(**data_item),
            source_indices=parent_record._source_indices,
            parent_ids=[parent_record._id],
            cardinality_idx=cardinality_idx,
        )

        return new_dr

    @staticmethod
    def from_agg_parents(
        data_item: BaseModel,
        parent_records: DataRecordSet,
        cardinality_idx: int | None = None,
    ) -> DataRecord:
        # flatten source indices from all parents
        source_indices = [
            source_idx
            for parent_record in parent_records
            for source_idx in parent_record._source_indices
        ]

        # make new record which has all parent records as its parents
        return DataRecord(
            data_item,
            source_indices=source_indices,
            parent_ids=[parent_record._id for parent_record in parent_records],
            cardinality_idx=cardinality_idx,
        )

    @staticmethod
    def from_join_parents(
        schema: type[BaseModel],
        left_parent_record: DataRecord | None,
        right_parent_record: DataRecord | None,
        project_cols: list[str] | None = None,
        cardinality_idx: int = None,
    ) -> DataRecord:
        # get the set of fields and field descriptions to copy from the parent record(s)
        left_copy_field_names = [] if left_parent_record is None else (
            left_parent_record.get_field_names()
            if project_cols is None
            else [col for col in project_cols if col in left_parent_record.get_field_names()]
        )
        right_copy_field_names = [] if right_parent_record is None else (
            right_parent_record.get_field_names()
            if project_cols is None
            else [col for col in project_cols if col in right_parent_record.get_field_names()]
        )
        left_copy_field_names = [field.split(".")[-1] for field in left_copy_field_names]
        right_copy_field_names = [field.split(".")[-1] for field in right_copy_field_names]

        # copy fields from the parents
        data_item = {field_name: left_parent_record[field_name] for field_name in left_copy_field_names}
        for field_name in right_copy_field_names:
            new_field_name = field_name
            if field_name in left_copy_field_names:
                new_field_name = f"{field_name}_right"
            data_item[new_field_name] = right_parent_record[field_name]

        # for any missing fields in the schema, set them to None
        for field_name in schema.model_fields:
            if field_name not in data_item:
                data_item[field_name] = None

        # make new record which has left and right parent record as its parents
        left_parent_source_indices = [] if left_parent_record is None else list(left_parent_record._source_indices)
        right_parent_source_indices = [] if right_parent_record is None else list(right_parent_record._source_indices)
        left_parent_record_id = [] if left_parent_record is None else [left_parent_record._id]
        right_parent_record_id = [] if right_parent_record is None else [right_parent_record._id]
        new_dr = DataRecord(
            schema(**data_item),
            source_indices=left_parent_source_indices + right_parent_source_indices,
            parent_ids=left_parent_record_id + right_parent_record_id,
            cardinality_idx=cardinality_idx,
        )

        return new_dr

    @staticmethod
    def to_df(records: list[DataRecord], project_cols: list[str] | None = None) -> pd.DataFrame:
        if len(records) == 0:
            return pd.DataFrame()

        fields = records[0].get_field_names()
        if project_cols is not None and len(project_cols) > 0:
            fields = [field for field in fields if field in project_cols]

        # convert Context --> str
        for record in records:
            for k in fields:
                if isinstance(record[k], context.Context):
                    record[k] = record[k].description

        return pd.DataFrame([
            {k: record[k] for k in fields}
            for record in records
        ])

    def to_json_str(self, include_bytes: bool = True, bytes_to_str: bool = False, project_cols: list[str] | None = None, sorted: bool = False):
        """Return a JSON representation of this DataRecord"""
        record_dict = self.to_dict(include_bytes, bytes_to_str, project_cols, sorted)
        return json.dumps(record_dict, indent=2)

    def to_dict(self, include_bytes: bool = True, bytes_to_str: bool = False, project_cols: list[str] | None = None, _sorted: bool = False, mask_filepaths: bool = False):
        """Return a dictionary representation of this DataRecord"""
        # TODO(chjun): In case of numpy types, the json.dumps will fail. Convert to native types.
        # Better ways to handle this.
        field_values = {
            k: v.description if isinstance(v, context.Context) else v
            for k, v in self._data_item.model_dump().items()
        }
        dct = pd.Series(field_values).to_dict()

        if project_cols is not None and len(project_cols) > 0:
            project_field_names = set(field.split(".")[-1] for field in project_cols)
            dct = {k: v for k, v in dct.items() if k in project_field_names}

        if not include_bytes:
            bytes_field_types = [bytes, list[bytes], bytes | None, list[bytes] | None, bytes | Any, list[bytes] | Any]
            bytes_field_types += AUDIO_FIELD_TYPES + IMAGE_FIELD_TYPES
            for k in dct:
                field_type = self.get_field_type(k)
                if field_type.annotation in bytes_field_types:
                    dct[k] = "<bytes>"

        if bytes_to_str:
            for k, v in dct.items():
                if isinstance(v, bytes):
                    dct[k] = v.decode("utf-8")
                elif isinstance(v, list) and len(v) > 0 and any([isinstance(elt, bytes) for elt in v]):
                    dct[k] = [elt.decode("utf-8") if isinstance(elt, bytes) else elt for elt in v]

        if _sorted:
            dct = dict(sorted(dct.items()))

        if mask_filepaths:
            for k in dct:
                field_type = self.get_field_type(k)
                if field_type.annotation in [AudioBase64, AudioFilepath, ImageBase64, ImageFilepath, ImageURL]:
                    dct[k] = "<bytes>"

        return deepcopy(dct)


class DataRecordSet:
    """
    A DataRecordSet contains a list of DataRecords that share the same schema, same parent(s), and same source(s).

    We explicitly check that this is True.

    The record_op_stats could be empty if the DataRecordSet is not from executing an operator.
    """
    def __init__(
            self,
            data_records: list[DataRecord],
            record_op_stats: list[RecordOpStats],
            field_to_score_fn: dict[str, str | callable] | None = None,
            input: int | DataRecord | list[DataRecord] | tuple[list[DataRecord]] | None = None,
        ):
        # set data_records, parent_ids, and source_indices; note that it is possible for
        # data_records to be an empty list in the event of a failed convert
        self.data_records = data_records
        self.parent_ids = data_records[0]._parent_ids if len(data_records) > 0 else None
        self.source_indices = data_records[0]._source_indices if len(data_records) > 0 else None
        self.schema = data_records[0].schema if len(data_records) > 0 else None

        # the input to the operator which produced the data_records; type is tuple[DataRecord] | tuple[int]
        # - for scan operators, input is a singleton tuple[int] which wraps the source_idx, e.g.: (source_idx,)
        # - for join operators, input is a tuple with one entry for the left input DataRecord and one entry for the right input DataRecord
        # - for aggregate operators, input is a tuple with all the input DataRecords to the aggregation
        # - for all other operaotrs, input is a singleton tuple[DataRecord] which wraps the single input
        self.input = input

        # set statistics for generating these records
        self.record_op_stats = record_op_stats

        # assign field_to_score_fn if provided
        self.field_to_score_fn = {} if field_to_score_fn is None else field_to_score_fn

    def get_total_cost(self) -> float:
        return sum([record_op_stats.cost_per_record for record_op_stats in self.record_op_stats])

    def get_field_to_score_fn(self) -> dict[str, str | callable]:
        return self.field_to_score_fn

    def __getitem__(self, slice) -> DataRecord | list[DataRecord]:
        return self.data_records[slice]

    def __len__(self) -> int:
        return len(self.data_records)

    def __iter__(self) -> Generator[DataRecord]:
        yield from self.data_records


class DataRecordCollection:
    """
    A DataRecordCollection contains a list of DataRecords.

    This is a wrapper class for list[DataRecord] to support more advanced features for output of execute().

    The difference between DataRecordSet and DataRecordCollection 

    Goal: 
        DataRecordSet is a set of DataRecords that share the same schema, same parents, and same sources.
        DataRecordCollection is a general wrapper for list[DataRecord].
    
    Usage:
        DataRecordSet is used for the output of executing an operator.
        DataRecordCollection is used for the output of executing a query, we definitely could extend it to support more advanced features for output of execute().
    """
    def __init__(self, data_records: list[DataRecord], execution_stats: ExecutionStats | None = None, plan_stats: PlanStats | None = None):
        self.data_records = data_records
        self.execution_stats = execution_stats
        self.plan_stats = plan_stats
        self.executed_plans = self._get_executed_plans()

    def __iter__(self) -> Generator[DataRecord]:
        """Allow iterating directly over the data records"""
        yield from self.data_records

    def __len__(self):
        """Return the number of records in the collection"""
        return len(self.data_records)

    def to_df(self, cols: list[str] | None = None):
        return DataRecord.to_df(self.data_records, cols)

    def _get_executed_plans(self):
        if self.plan_stats is not None:
            return [self.plan_stats.plan_str]
        elif self.execution_stats is not None:
            return list(self.execution_stats.plan_strs.values())
        else:
            return None

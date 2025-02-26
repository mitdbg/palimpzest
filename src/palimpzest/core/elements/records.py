from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pandas as pd

from palimpzest.core.data.dataclasses import ExecutionStats, PlanStats, RecordOpStats
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import Schema
from palimpzest.utils.hash_helpers import hash_for_id


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""

    def __init__(
        self,
        schema: Schema,
        source_idx: int,
        parent_id: str | None = None,
        cardinality_idx: int | None = None,
    ):
        # check that source_idx is provided
        assert source_idx is not None, "Every DataRecord must be constructed with a source_idx"

        # schema for the data record
        self.schema = schema

        # mapping from field names to Field objects; effectively a mapping from a field name to its type        
        self.field_types: dict[str, Field] = schema.field_map()

        # mapping from field names to their values
        self.field_values: dict[str, Any] = {}

        # the index in the DataReader from which this DataRecord is derived
        self.source_idx = source_idx

        # the id of the parent record(s) from which this DataRecord is derived
        self.parent_id = parent_id

        # store the cardinality index
        self.cardinality_idx = cardinality_idx

        # indicator variable which may be flipped by filter operations to signal when a record has been filtered out
        self.passed_operator = True

        # NOTE: Record ids are hashed based on:
        # 0. their schema (keys)
        # 1. their parent record id(s) (or source_idx if there is no parent record)
        # 2. their index in the fan out (if this is in a one-to-many operation)
        #
        # We currently do NOT hash just based on record content (i.e. schema (key, value) pairs)
        # because multiple outputs for a given operation may have the exact same
        # schema (key, value) pairs.
        #
        # We may revisit this hashing scheme in the future.

        # unique identifier for the record
        id_str = (
            str(schema) + (parent_id if parent_id is not None else str(self.source_idx))
            if cardinality_idx is None
            else str(schema) + str(cardinality_idx) + str(parent_id if parent_id is not None else str(self.source_idx))
        )
        # TODO(Jun): build-in id should has a special name, the current self.id is too general which would conflict with user defined schema too easily.
        # the options: built_in_id, generated_id
        self.id = hash_for_id(id_str)


    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ["schema", "field_types", "field_values", "source_idx", "parent_id", "cardinality_idx", "passed_operator", "id"]:
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


    def __str__(self, truncate: int | None = 15) -> str:
        if truncate is not None:
            items = (f"{k}={str(v)[:truncate]!r}{'...' if len(str(v)) > truncate else ''}" for k, v in sorted(self.field_values.items()))
        else:
            items = (f"{k}={v!r}" for k, v in sorted(self.field_values.items()))
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __repr__(self) -> str:
        return self.__str__(truncate=None)

    def __eq__(self, other):
        return isinstance(other, DataRecord) and self.field_values == other.field_values and self.schema.get_desc() == other.schema.get_desc()


    def __hash__(self):
        return hash(self.to_json_str())


    def __iter__(self):
        yield from self.field_values.items()


    def get_field_names(self):
        return list(self.field_values.keys())


    def get_field_type(self, field_name: str) -> Field:
        return self.field_types[field_name]


    def copy(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        # make new record which has parent_record as its parent (and the same source_idx)
        new_dr = DataRecord(
            self.schema,
            source_idx=self.source_idx,
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

        # make new record which has parent_record as its parent (and the same source_idx)
        new_dr = DataRecord(
            new_schema,
            source_idx=parent_record.source_idx,
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


    @staticmethod
    def from_df(df: pd.DataFrame, schema: Schema | None = None) -> list[DataRecord]:
        """Create a list of DataRecords from a pandas DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            schema (Schema, optional): Schema for the DataRecords. If None, will be derived from DataFrame  
        
        Returns:
            list[DataRecord]: List of DataRecord instances
        """
        if df is None:
            raise ValueError("DataFrame is None!")

        records = []
        if schema is None:
            schema = Schema.from_df(df)

        field_map = schema.field_map()
        for source_idx, row in df.iterrows():
            row_dict = row.to_dict()
            record = DataRecord(schema=schema, source_idx=source_idx)
            record.field_values = row_dict
            record.field_types = {field_name: field_map[field_name] for field_name in row_dict}
            records.append(record)

        return records

    @staticmethod
    def to_df(records: list[DataRecord], project_cols: list[str] | None = None) -> pd.DataFrame:
        if len(records) == 0:
            return pd.DataFrame()

        fields = records[0].get_field_names()
        if project_cols is not None and len(project_cols) > 0:
            fields = [field for field in fields if field in project_cols]

        return pd.DataFrame([
            {k: record[k] for k in fields}
            for record in records
        ])

    def to_json_str(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        """Return a JSON representation of this DataRecord"""
        record_dict = self.to_dict(include_bytes, project_cols)
        record_dict = {
            field_name: self.schema.field_to_json(field_name, field_value)
            for field_name, field_value in record_dict.items()
        }
        return json.dumps(record_dict, indent=2)

    def to_dict(self, include_bytes: bool = True, project_cols: list[str] | None = None):
        """Return a dictionary representation of this DataRecord"""
        # TODO(chjun): In case of numpy types, the json.dumps will fail. Convert to native types.
        # Better ways to handle this.
        dct = pd.Series(self.field_values).to_dict()

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
    A DataRecordSet contains a list of DataRecords that share the same schema, same parent_id, and same source_idx.

    We explicitly check that this is True.

    The record_op_stats could be empty if the DataRecordSet is not from executing an operator.
    """
    def __init__(self, data_records: list[DataRecord], record_op_stats: list[RecordOpStats]):
        # check that all data_records are derived from the same parent record
        if len(data_records) > 0:
            parent_id = data_records[0].parent_id
            error_msg = "DataRecordSet must be constructed from the output of executing a single operator on a single input."
            assert all([dr.parent_id == parent_id for dr in data_records]), error_msg

        # set data_records, parent_id, and source_idx; note that it is possible for
        # data_records to be an empty list in the event of a failed convert
        self.data_records = data_records
        self.parent_id = data_records[0].parent_id if len(data_records) > 0 else None
        self.source_idx = data_records[0].source_idx if len(data_records) > 0 else None

        # set statistics for generating these records
        self.record_op_stats = record_op_stats

    def get_total_cost(self):
        return sum([record_op_stats.cost_per_record for record_op_stats in self.record_op_stats])

    def __getitem__(self, slice):
        return self.data_records[slice]


    def __len__(self):
        return len(self.data_records)


    def __iter__(self):
        yield from self.data_records


class DataRecordCollection:
    """
    A DataRecordCollection contains a list of DataRecords.

    This is a wrapper class for list[DataRecord] to support more advanced features for output of execute().

    The difference between DataRecordSet and DataRecordCollection 

    Goal: 
        DataRecordSet is a set of DataRecords that share the same schema, same parent_id, and same source_idx.
        DataRecordCollection is a general wrapper for list[DataRecord].
    
    Usage:
        DataRecordSet is used for the output of executing an operator.
        DataRecordCollection is used for the output of executing a query, we definitely could extend it to support more advanced features for output of execute().
    """
    # TODO(Jun): consider to have stats_manager class to centralize stats management.
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

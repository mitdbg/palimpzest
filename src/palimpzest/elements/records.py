from __future__ import annotations
from palimpzest.constants import MAX_ID_CHARS
from palimpzest.dataclasses import RecordOpStats
from palimpzest.corelib import Schema

from typing import List, Optional, Union

import hashlib


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""

    def __init__(
        self,
        schema: Schema,
        source_id: Union[int, str],
        parent_id: str = None,
        cardinality_idx: int = None,
    ):
        # check that source_id is provided
        assert source_id is not None, "Every DataRecord must be constructed with a source_id"

        # schema for the data record
        self.schema = schema

        # the source record(s) from which this DataRecord is derived
        self._source_id = str(source_id)

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
            else str(schema)
            + str(cardinality_idx)
            + (parent_id if parent_id is not None else self._source_id)
        )
        self._id = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]
        self._parent_id = parent_id

    @staticmethod
    def fromParent(
        schema: Schema,
        parent_record: DataRecord,
        project_cols: Optional[List[str]] = None,
        cardinality_idx: Optional[int] = None,
    ) -> DataRecord:
        # make new record which has parent_record as its parent (and the same source_id)
        new_dr = DataRecord(
            schema,
            source_id=parent_record._source_id,
            parent_id=parent_record._id,
            cardinality_idx=cardinality_idx,
        )

        # get the set of fields to copy from the parent record
        copy_fields = project_cols if project_cols is not None else parent_record._getFields()

        # copy fields from the parent
        for field in copy_fields:
            setattr(new_dr, field, getattr(parent_record, field))

        return new_dr

    @staticmethod
    def fromAggParents(
        schema: Schema,
        parent_records: DataRecordSet,
        project_cols: Optional[List[str]] = None,
        cardinality_idx: Optional[int] = None,
    ) -> DataRecord:
        # TODO: we can implement this once we support having multiple parent ids
        pass

    @staticmethod
    def fromJoinParents(
        left_schema: Schema,
        right_schema: Schema,
        left_parent_record: DataRecord,
        right_parent_record: DataRecord,
        project_cols: Optional[List[str]] = None,
        cardinality_idx: Optional[int] = None,
    ) -> DataRecord:
        # TODO: we can implement this method if/when we add joins
        pass

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
                    isinstance(dct[k], list) and len(dct[k]) > 0 and isinstance(dct[k][0], bytes)
                ):
                    dct[k] = "<bytes>"
        return dct

    def __str__(self):
        keys = sorted(self.__dict__.keys())
        items = ("{}={!r}...".format(k, str(self.__dict__[k])[:15]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self._asJSONStr())

    # NOTE: the method is called _getFields instead of getFields to avoid it being picked up as a data record attribute;
    #       in the future we will come up with a less ugly fix -- but for now do not remove the _ even though it's not private
    def _getFields(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_") and k != "schema"]


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
    def __init__(self, data_records: List[DataRecord], record_op_stats: List[RecordOpStats]):
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
        for dr in self.data_records:
            yield dr

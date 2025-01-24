from __future__ import annotations

import time
from abc import ABC, abstractmethod

from palimpzest.constants import (
    LOCAL_SCAN_TIME_PER_KB,
    MEMORY_SCAN_TIME_PER_KB,
    Cardinality,
)
from palimpzest.core.data.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator


class DataSourcePhysicalOp(PhysicalOperator, ABC):
    """
    Physical operators which implement DataSources require slightly more information
    in order to accurately compute naive cost estimates. Thus, we use a slightly
    modified abstract base class for these operators.
    """

    def __init__(self, dataset_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id

    def __str__(self):
        op = f"{self.op_name()}({self.dataset_id}) -> {self.output_schema}\n"
        op += f"    ({', '.join(self.output_schema.field_names())[:30]})\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"dataset_id": self.dataset_id, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"dataset_id": self.dataset_id, **op_params}

    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Cardinality,
        input_record_size_in_bytes: int | float,
    ) -> OperatorCostEstimates:
        """
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - quality

        For the implemented operator. These will be used by the CostModel
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("Abstract method")
    
    @abstractmethod
    def get_datasource(self):
        raise NotImplementedError("Abstract method")
        

class MarshalAndScanDataOp(DataSourcePhysicalOp):
    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_record_size_in_bytes: int | float,
        dataset_type: str,
    ) -> OperatorCostEstimates:
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        per_record_size_kb = input_record_size_in_bytes / 1024.0
        time_per_record = (
            LOCAL_SCAN_TIME_PER_KB * per_record_size_kb
            if dataset_type in ["dir", "file"]
            else MEMORY_SCAN_TIME_PER_KB * per_record_size_kb
        )

        # estimate output cardinality
        cardinality = source_op_cost_estimates.cardinality

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        """
        This function takes the candidate -- which is a DataRecord with a SourceRecord schema --
        and invokes its get_item_fn on the given idx to return the next DataRecord from the DataSource.
        """
        start_time = time.time()
        records = candidate.get_item_fn(candidate.idx)
        end_time = time.time()

        # if records is a DataRecord (instead of a list) wrap it in a list
        if isinstance(records, DataRecord):
            records = [records]

        # assert that every element of records is a DataRecord and has a source_id
        for dr in records:
            assert isinstance(dr, DataRecord), "Output from DataSource.get_item() must be a DataRecord or List[DataRecord]"

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_id=record.id,
                record_parent_id=record.parent_id,
                record_source_id=record.source_id,
                record_state=record.to_dict(include_bytes=False),
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct and return DataRecordSet object
        record_set = DataRecordSet(records, record_op_stats_lst)

        return record_set
    
    def get_datasource(self):
        return self.datadir.get_registered_dataset(self.dataset_id)
    
    def get_datasource_type(self):
        return self.datadir.get_registered_dataset_type(self.dataset_id)


class CacheScanDataOp(DataSourcePhysicalOp):
    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_record_size_in_bytes: int | float,
    ):
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        per_record_size_kb = input_record_size_in_bytes / 1024.0
        time_per_record = LOCAL_SCAN_TIME_PER_KB * per_record_size_kb

        # estimate output cardinality
        cardinality = source_op_cost_estimates.cardinality

        # for now, assume no cost per record for reading from cache
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()
        records = candidate.get_item_fn(candidate.idx)
        end_time = time.time()

        # if records is a DataRecord (instead of a list) wrap it in a list
        if isinstance(records, DataRecord):
            records = [records]

        # assert that every element of records is a DataRecord and has a source_id
        for dr in records:
            assert isinstance(dr, DataRecord), "Output from DataSource.get_item() must be a DataRecord or List[DataRecord]"

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_id=record.id,
                record_parent_id=record.parent_id,
                record_source_id=record.source_id,
                record_state=record.to_dict(include_bytes=False),
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct and eturn DataRecordSet object
        record_set = DataRecordSet(records, record_op_stats_lst)

        return record_set

    def get_datasource(self):
        return self.datadir.get_cached_result(self.dataset_id)

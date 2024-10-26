from __future__ import annotations

import time
from typing import List, Union

from palimpzest.constants import (
    LOCAL_SCAN_TIME_PER_KB,
    MEMORY_SCAN_TIME_PER_KB,
    NAIVE_EST_ONE_TO_MANY_SELECTIVITY,
    Cardinality,
)
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.elements.records import DataRecord
from palimpzest.operators.physical import DataRecordsWithStats, PhysicalOperator


class DataSourcePhysicalOp(PhysicalOperator):
    """
    Physical operators which implement DataSources require slightly more information
    in order to accurately compute naive cost estimates. Thus, we use a slightly
    modified abstract base class for these operators.
    """

    def __init__(self, dataset_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id

    def __str__(self):
        op = f"{self.op_name()}({self.dataset_id}) -> {self.outputSchema}\n"
        op += f"    ({', '.join(self.outputSchema.fieldNames())[:30]})\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"dataset_id": self.dataset_id, **copy_kwargs}

    def get_op_params(self):
        return {"outputSchema": self.outputSchema, "dataset_id": self.dataset_id}

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.outputSchema == other.outputSchema
            and self.dataset_id == other.dataset_id
        )

    def naiveCostEstimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
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


class MarshalAndScanDataOp(DataSourcePhysicalOp):
    def naiveCostEstimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
        dataset_type: str,
    ) -> OperatorCostEstimates:
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        perRecordSizeInKb = input_record_size_in_bytes / 1024.0
        timePerRecord = (
            LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb
            if dataset_type in ["dir", "file"]
            else MEMORY_SCAN_TIME_PER_KB * perRecordSizeInKb
        )

        # estimate output cardinality
        cardinality = (
            source_op_cost_estimates.cardinality
            if input_cardinality == Cardinality.ONE_TO_ONE
            else source_op_cost_estimates.cardinality * NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        )

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        """
        This function takes the candidate -- which is a DataRecord with a SourceRecord schema --
        and invokes its get_item_fn on the given idx to return the next DataRecord from the DataSource.
        """
        start_time = time.time()
        output = candidate.get_item_fn(candidate.idx)
        records = [output] if candidate.cardinality == Cardinality.ONE_TO_ONE else output
        end_time = time.time()

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_id=record._id,
                record_parent_id=record._parent_id,
                record_state=record._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return records, record_op_stats_lst


class CacheScanDataOp(DataSourcePhysicalOp):
    def naiveCostEstimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
    ):
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        perRecordSizeInKb = input_record_size_in_bytes / 1024.0
        timePerRecord = LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb

        # estimate output cardinality
        cardinality = (
            source_op_cost_estimates.cardinality
            if input_cardinality == Cardinality.ONE_TO_ONE
            else source_op_cost_estimates.cardinality * NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        )

        # for now, assume no cost per record for reading from cache
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        start_time = time.time()
        output = candidate.get_item_fn(candidate.idx)
        records = [output] if candidate.cardinality == Cardinality.ONE_TO_ONE else output
        end_time = time.time()

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_id=record._id,
                record_parent_id=record._parent_id,
                record_state=record._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return records, record_op_stats_lst

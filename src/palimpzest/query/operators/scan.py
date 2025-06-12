from __future__ import annotations

import time
from abc import ABC, abstractmethod

from palimpzest.constants import (
    LOCAL_SCAN_TIME_PER_KB,
    MEMORY_SCAN_TIME_PER_KB,
    Cardinality,
)
from palimpzest.core.data.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.core.data.datareaders import DataReader, DirectoryReader, FileReader
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator


class ScanPhysicalOp(PhysicalOperator, ABC):
    """
    Physical operators which implement DataReaders require slightly more information
    in order to accurately compute naive cost estimates. Thus, we use a slightly
    modified abstract base class for these operators.
    """

    def __init__(self, datareader: DataReader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datareader = datareader

    def __str__(self):
        op = f"{self.op_name()}({self.datareader}) -> {self.output_schema}\n"
        op += f"    ({', '.join(self.output_schema.field_names())[:30]})\n"
        return op

    def get_id_params(self):
        return super().get_id_params()

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"datareader": self.datareader, **op_params}

    @abstractmethod
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
        execution data alone -- thus ScanPhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        pass

    def __call__(self, idx: int) -> DataRecordSet:
        """
        This function invokes `self.datareader.__getitem__` on the given `idx` to retrieve the next data item.
        It then returns this item as a DataRecord wrapped in a DataRecordSet.
        """
        start_time = time.time()
        item = self.datareader[idx]
        end_time = time.time()

        # TODO: remove once validation data is refactored
        item_field_dict = item.get("fields", item)

        # check that item covers fields in output schema
        output_field_names = self.output_schema.field_names()
        assert all([field in item_field_dict for field in output_field_names]), f"Some fields in DataReader schema not present in item!\n - DataReader fields: {output_field_names}\n - Item fields: {list(item.keys())}"

        # construct a DataRecord from the item
        dr = DataRecord(self.output_schema, source_idx=idx)
        for field in output_field_names:
            setattr(dr, field, item_field_dict[field])

        # create RecordOpStats objects
        record_op_stats = RecordOpStats(
            record_id=dr.id,
            record_parent_id=dr.parent_id,
            record_source_idx=dr.source_idx,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=(end_time - start_time),
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
 
        # construct and return DataRecordSet object
        return DataRecordSet([dr], [record_op_stats])
        

class MarshalAndScanDataOp(ScanPhysicalOp):
    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_record_size_in_bytes: int | float,
    ) -> OperatorCostEstimates:
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        per_record_size_kb = input_record_size_in_bytes / 1024.0
        time_per_record = (
            LOCAL_SCAN_TIME_PER_KB * per_record_size_kb
            if isinstance(self.datareader, (DirectoryReader, FileReader))
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


class CacheScanDataOp(ScanPhysicalOp):
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

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from palimpzest.constants import LOCAL_SCAN_TIME_PER_KB
from palimpzest.core.data import context
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator


class ScanPhysicalOp(PhysicalOperator, ABC):
    """
    Physical operators which implement root Datasets require slightly more information
    in order to accurately compute naive cost estimates. Thus, we use a slightly
    modified abstract base class for these operators.
    """
    # datasource: IterDataset
    def __init__(self, datasource: Any, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasource = datasource

    def __str__(self):
        op = f"{self.op_name()}({self.datasource}) -> {self.output_schema}\n"
        op += f"    ({', '.join(list(self.output_schema.model_fields))[:30]})\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"datasource_id": self.datasource.id, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"datasource": self.datasource, **op_params}

    @abstractmethod
    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
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
        This function invokes `self.datasource.__getitem__` on the given `idx` to retrieve the next data item.
        It then returns this item as a DataRecord wrapped in a DataRecordSet.
        """
        start_time = time.time()
        item = self.datasource[idx]
        end_time = time.time()

        # check that item covers fields in output schema
        output_field_names = list(self.output_schema.model_fields)
        assert all([field in item for field in output_field_names]), f"Some fields in Dataset schema not present in item!\n - Dataset fields: {output_field_names}\n - Item fields: {list(item.keys())}"

        # construct a DataRecord from the item
        data_item = self.output_schema(**{field: item[field] for field in output_field_names})
        dr = DataRecord(data_item, source_indices=[f"{self.datasource.id}-{idx}"])

        # create RecordOpStats objects
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
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

        # TODO: cannot do the first computation b/c we cannot import iter_dataset; possibly revisit
        # time_per_record = (
        #     MEMORY_SCAN_TIME_PER_KB * per_record_size_kb
        #     if isinstance(self.datasource, (iter_dataset.MemoryDataset))
        #     else LOCAL_SCAN_TIME_PER_KB * per_record_size_kb
        # )
        time_per_record = LOCAL_SCAN_TIME_PER_KB * per_record_size_kb

        # estimate output cardinality
        cardinality = source_op_cost_estimates.cardinality

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0,
            quality=1.0,
        )


class ContextScanOp(PhysicalOperator):
    """
    Physical operator which facillitates the loading of a Context for processing.
    """

    def __init__(self, context: context.Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    def __str__(self):
        op = f"{self.op_name()}({self.context}) -> {self.output_schema}\n"
        op += f"    ({', '.join(list(self.output_schema.model_fields))[:30]})\n"
        return op

    def get_id_params(self):
        return super().get_id_params()

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"context": self.context, **op_params}

    def naive_cost_estimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
    ):
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        time_per_record = LOCAL_SCAN_TIME_PER_KB * 1.0

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=1.0,
            time_per_record=time_per_record,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, *args, **kwargs) -> DataRecordSet:
        """
        This function returns the context as a DataRecord wrapped in a DataRecordSet.
        """
        # construct a DataRecord from the context
        start_time = time.time()
        dr = DataRecord(self.output_schema(), source_indices=[f"{self.context.id}-{0}"])
        dr.context = self.context
        end_time = time.time()

        # create RecordOpStats objects
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
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

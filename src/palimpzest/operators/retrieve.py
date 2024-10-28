from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib import *
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.elements import *
from palimpzest.operators import PhysicalOperator

import time


class RetrieveOp(PhysicalOperator):
    def __init__(self, index, search_attr, output_attr, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.search_attr = search_attr
        self.output_attr = output_attr
        self.k = k

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.index == other.index
            and self.search_attr == other.search_attr
            and self.output_attr == other.output_attr
            and self.k == other.k
        )

    def __str__(self):
        op = super().__str__()
        op += f"    Retrieve: {str(self.index)} with top {self.k}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {
            "index": self.index,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **copy_kwargs,
        }

    def get_op_params(self):
        return {
            "index": self.index,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
        }

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        query = getattr(candidate, self.search_attr)
        
        top_k_strs = self.index.search(query, k=self.k)

        output_dr = DataRecord.fromParent(self.outputSchema, parent_record=candidate)
        setattr(output_dr, self.output_attr, top_k_strs)

        duration_secs = time.time() - start_time

        answer = (
            {
                field_name: getattr(output_dr, field_name)
                for field_name in self.outputSchema.fieldNames()
            },
        )

        record_op_stats = RecordOpStats(
            record_id=output_dr._id,
            record_parent_id=output_dr._parent_id,
            record_source_id=output_dr._source_id,
            record_state=output_dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=duration_secs,
            cost_per_record=0.0,
            answer=answer,
            input_fields=self.inputSchema.fieldNames(),
            generated_fields=self.outputSchema.fieldNames(),
            fn_call_duration_secs=duration_secs,  # TODO(Siva): Currently tracking retrieval time in fn_call_duration_secs
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        drs = [output_dr]
        record_op_stats_lst = [record_op_stats]

        # construct record set
        record_set = DataRecordSet(drs, record_op_stats_lst)

        return record_set

    def naiveCostEstimates(
        self, source_op_cost_estimates: OperatorCostEstimates
    ) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the Retrieve operation. These estimates assume
        that the Retrieve (1) has no cost and (2) has perfect quality.
        """
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.001,  # estimate 1 ms single-threaded execution for index lookup
            cost_per_record=0.0,
            quality=1.0,
        )

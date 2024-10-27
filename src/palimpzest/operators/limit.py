from __future__ import annotations

from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.elements.records import DataRecord
from palimpzest.operators.physical import DataRecordsWithStats, PhysicalOperator


class LimitScanOp(PhysicalOperator):
    def __init__(self, limit: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit

    def __str__(self):
        op = super().__str__()
        op += f"    Limit: {self.limit}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"limit": self.limit, **copy_kwargs}

    def get_op_params(self):
        return {
            "output_schema": self.output_schema,
            "limit": self.limit,
        }

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.limit == other.limit
            and self.output_schema == other.output_schema
            and self.input_schema == other.input_schema
        )

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the limit takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=min(self.limit, source_op_cost_estimates.cardinality),
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> list[DataRecordsWithStats]:
        # NOTE: execution layer ensures that no more than self.limit
        #       records are returned to the user by this operator.
        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
            record_state=candidate._as_dict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
        )

        return [candidate], [record_op_stats]

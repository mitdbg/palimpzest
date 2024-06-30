from __future__ import annotations

from palimpzest.corelib import Schema
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.elements import DataRecord
from palimpzest.operators import logical, PhysicalOperator, DataRecordsWithStats

from typing import List


class LimitScanOp(PhysicalOperator):
    implemented_op = logical.LimitScan
    final = True

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Schema,
        limit: int,
        targetCacheId: str = None,
        shouldProfile=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            shouldProfile=shouldProfile,
        )
        self.limit = limit
        self.targetCacheId = targetCacheId

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.limit == other.limit
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Limit: "
            + str(self.limit)
            + ")"
        )

    def copy(self):
        return LimitScanOp(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            limit=self.limit,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "limit": self.limit,
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the limit takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=min(self.limit, source_op_cost_estimates.cardinality),
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        # NOTE: SimpleExecution.execute_dag ensures that no more than self.limit
        #       records are returned to the user by this operator.
        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            record_state=candidate._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
        )

        return [candidate], [record_op_stats]

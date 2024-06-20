from __future__ import annotations

from palimpzest.constants import NAIVE_EST_NUM_GROUPS
from palimpzest.corelib import Number, Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import AggregateFunction, DataRecord, GroupBySig
from palimpzest.operators import logical, PhysicalOperator, DataRecordsWithStats

from typing import List

import time


class AggregateOp(PhysicalOperator):
    """
    Aggregate operators accept a list of candidate DataRecords as input to their
    __call__ methods. Thus, we use a slightly modified abstract base class for
    these operators.
    """
    def __call__(self, candidates: List[DataRecord]) -> List[DataRecordsWithStats]:
        raise NotImplementedError("Using __call__ from abstract method")


class ApplyGroupByOp(AggregateOp):
    implemented_op = logical.GroupByAggregate

    def __init__(
        self,
        inputSchema: Schema,
        gbySig: GroupBySig,
        targetCacheId: str = None,
        shouldProfile=False,
        *args, **kwargs,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=gbySig.outputSchema(),
            shouldProfile=shouldProfile,
        )
        self.inputSchema = inputSchema
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId
        self.shouldProfile = shouldProfile

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.gbySig == other.gbySig
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return f"{self.op_name()}({str(self.gbySig)})"

    def copy(self):
        return ApplyGroupByOp(
            self.inputSchema, self.gbySig, self.targetCacheId, self.shouldProfile
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "gbySig": str(GroupBySig.serialize(self.gbySig)),
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the groupby takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=NAIVE_EST_NUM_GROUPS,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    @staticmethod
    def agg_init(func):
        if func.lower() == "count":
            return 0
        elif func.lower() == "average":
            return (0, 0)
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_merge(func, state, val):
        if func.lower() == "count":
            return state + 1
        elif func.lower() == "average":
            sum, cnt = state
            return (sum + val, cnt + 1)
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_final(func, state):
        if func.lower() == "count":
            return state
        elif func.lower() == "average":
            sum, cnt = state
            return float(sum) / cnt
        else:
            raise Exception("Unknown agg function " + func)

    def __call__(self, candidates: List[DataRecord]) -> List[DataRecordsWithStats]:
        start_time = time.time()

        # build group array
        aggState = {}
        for candidate in candidates:
            group = ()
            for f in self.gbySig.gbyFields:
                if not hasattr(candidate, f):
                    raise TypeError(
                        f"ApplyGroupByOp record missing expected field {f}"
                    )
                group = group + (getattr(candidate, f),)
            if group in aggState:
                state = aggState[group]
            else:
                state = []
                for fun in self.gbySig.aggFuncs:
                    state.append(ApplyGroupByOp.agg_init(fun))
            for i in range(0, len(self.gbySig.aggFuncs)):
                fun = self.gbySig.aggFuncs[i]
                if not hasattr(candidate, self.gbySig.aggFields[i]):
                    raise TypeError(
                        f"ApplyGroupByOp record missing expected field {self.gbySig.aggFields[i]}"
                    )
                field = getattr(candidate, self.gbySig.aggFields[i])
                state[i] = ApplyGroupByOp.agg_merge(fun, state[i], field)
            aggState[group] = state

        # return list of data records (one per group)
        drs = []
        gbyFields = self.gbySig.gbyFields
        aggFields = self.gbySig.getAggFieldNames()
        for g in aggState.keys():
            dr = DataRecord(self.gbySig.outputSchema())
            for i in range(0, len(g)):
                k = g[i]
                setattr(dr, gbyFields[i], k)
            vals = aggState[g]
            for i in range(0, len(vals)):
                v = ApplyGroupByOp.agg_final(self.gbySig.aggFuncs[i], vals[i])
                setattr(dr, aggFields[i], v)

            drs.append(dr)

        # create RecordOpStats objects
        total_time = time.time() - start_time
        record_op_stats_lst = []
        for dr in drs:
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=total_time / len(drs),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return drs, record_op_stats_lst


class ApplyCountAggregateOp(AggregateOp):
    implemented_op = logical.ApplyAggregateFunction

    def __init__(
        self,
        inputSchema: Schema,
        aggFunction: AggregateFunction,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema, outputSchema=Number, shouldProfile=shouldProfile
        )
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunction == other.aggFunction
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Function: "
            + str(self.aggFunction)
            + ")"
        )

    def copy(self):
        return ApplyCountAggregateOp(
            inputSchema=self.inputSchema,
            aggFunction=self.aggFunction,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "aggFunction": str(self.aggFunction)
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: List[DataRecord]) -> List[DataRecordsWithStats]:
        start_time = time.time()

        # NOTE: this will set the parent_uuid to be the uuid of the final source record;
        #       in the near future we may want to have parent_uuid accept a list of uuids
        dr = DataRecord(Number, parent_uuid=candidates[-1]._uuid)
        dr.value = len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return [dr], [record_op_stats]


class ApplyAverageAggregateOp(AggregateOp):
    implemented_op = logical.ApplyAggregateFunction

    def __init__(
        self,
        inputSchema: Schema,
        aggFunction: AggregateFunction,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema, outputSchema=Number, shouldProfile=shouldProfile
        )
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

        if not inputSchema == Number:
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunction == other.aggFunction
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Function: "
            + str(self.aggFunction)
            + ")"
        )

    def copy(self):
        return ApplyAverageAggregateOp(
            inputSchema=self.inputSchema,
            aggFunction=self.aggFunction,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "aggFunction": str(self.aggFunction)
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: List[DataRecord]) -> List[DataRecordsWithStats]:
        start_time = time.time()

        # NOTE: this will set the parent_uuid to be the uuid of the final source record;
        #       in the near future we may want to have parent_uuid accept a list of uuids
        dr = DataRecord(Number, parent_uuid=candidates[-1]._uuid)
        dr.value = sum(list(map(lambda c: float(c.value), candidates))) / len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return [dr], [record_op_stats]

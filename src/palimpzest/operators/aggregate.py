from __future__ import annotations

from palimpzest.constants import NAIVE_EST_NUM_GROUPS, AggFunc
from palimpzest.corelib import Number
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord, DataRecordSet, GroupBySig
from palimpzest.operators import PhysicalOperator

import time


class AggregateOp(PhysicalOperator):
    """
    Aggregate operators accept a list of candidate DataRecords as input to their
    __call__ methods. Thus, we use a slightly modified abstract base class for
    these operators.
    """
    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
        raise NotImplementedError("Using __call__ from abstract method")


class ApplyGroupByOp(AggregateOp):

    def __init__(self, gbySig: GroupBySig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gbySig = gbySig

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.gbySig == other.gbySig
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        op = super().__str__()
        op += f"    Group-by Signature: {str(self.gbySig)}\n"
        return op
    
    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"gbySig": self.gbySig, **copy_kwargs}

    def get_op_params(self):
        """
        We identify the operation by its outputSchema and group by signature.
        inputSchema is ignored as it depends on how the Optimizer orders operations.
        """
        return {
            "outputSchema": self.outputSchema,
            "gbySig": str(self.gbySig.serialize()),
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

    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
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
            # NOTE: this will set the parent_id and source_id to be the id of the final source record;
            #       in the near future we may want to have parent_id accept a list of ids
            dr = DataRecord.fromParent(
                schema=self.gbySig.outputSchema(),
                parent_record=candidates[-1],
                project_cols=[],
            )
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
                record_id=dr._id,
                record_parent_id=dr._parent_id,
                record_source_id=dr._source_id,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=total_time / len(drs),
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_op_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct and return DataRecordSet
        return DataRecordSet(drs, record_op_stats_lst)


class CountAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use aggFunc here (yet)

    def __init__(self, aggFunc: AggFunc, *args, **kwargs):
        kwargs["outputSchema"] = Number
        super().__init__(*args, **kwargs)
        self.aggFunc = aggFunc

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunc == other.aggFunc
        )

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.aggFunc)}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"aggFunc": self.aggFunc, **copy_kwargs}

    def get_op_params(self):
        """
        We identify the operation by its aggregation function.
        inputSchema is ignored as it depends on how the Optimizer orders operations.
        """
        return {"aggFunc": str(self.aggFunc)}

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
        start_time = time.time()

        # NOTE: this will set the parent_id and source_id to be the id of the final source record;
        #       in the near future we may want to have parent_id accept a list of ids
        dr = DataRecord.fromParent(schema=Number, parent_record=candidates[-1], project_cols=[])
        dr.value = len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_id=dr._parent_id,
            record_source_id=dr._source_id,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])


class AverageAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use aggFunc here (yet)

    def __init__(self, aggFunc: AggFunc, *args, **kwargs):
        kwargs["outputSchema"] = Number
        super().__init__(*args, **kwargs)
        self.aggFunc = aggFunc

        if not self.inputSchema == Number:
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunc == other.aggFunc
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.aggFunc)}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"aggFunc": self.aggFunc, **copy_kwargs}

    def get_op_params(self):
        """
        We identify the operation by its aggregation function.
        inputSchema is ignored as it depends on how the Optimizer orders operations.
        """
        return {"aggFunc": str(self.aggFunc)}

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
        start_time = time.time()

        # NOTE: this will set the parent_id and source_id to be the id of the final source record;
        #       in the near future we may want to have parent_id accept a list of ids
        dr = DataRecord.fromParent(schema=Number, parent_record=candidates[-1], project_cols=[])
        dr.value = sum(list(map(lambda c: float(c.value), candidates))) / len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_id=dr._parent_id,
            record_source_id=dr._source_id,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])

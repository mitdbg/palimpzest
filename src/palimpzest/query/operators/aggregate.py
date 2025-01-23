from __future__ import annotations

import time

from palimpzest.constants import NAIVE_EST_NUM_GROUPS, AggFunc
from palimpzest.core.data.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Number
from palimpzest.query.operators.physical import PhysicalOperator


class AggregateOp(PhysicalOperator):
    """
    Aggregate operators accept a list of candidate DataRecords as input to their
    __call__ methods. Thus, we use a slightly modified abstract base class for
    these operators.
    """
    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
        raise NotImplementedError("Using __call__ from abstract method")


class ApplyGroupByOp(AggregateOp):
    def __init__(self, group_by_sig: GroupBySig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_by_sig = group_by_sig

    def __str__(self):
        op = super().__str__()
        op += f"    Group-by Signature: {str(self.group_by_sig)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"group_by_sig": str(self.group_by_sig.serialize()), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"group_by_sig": self.group_by_sig, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
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
        agg_state = {}
        for candidate in candidates:
            group = ()
            for f in self.group_by_sig.group_by_fields:
                if not hasattr(candidate, f):
                    raise TypeError(f"ApplyGroupByOp record missing expected field {f}")
                group = group + (getattr(candidate, f),)
            if group in agg_state:
                state = agg_state[group]
            else:
                state = []
                for fun in self.group_by_sig.agg_funcs:
                    state.append(ApplyGroupByOp.agg_init(fun))
            for i in range(0, len(self.group_by_sig.agg_funcs)):
                fun = self.group_by_sig.agg_funcs[i]
                if not hasattr(candidate, self.group_by_sig.agg_fields[i]):
                    raise TypeError(f"ApplyGroupByOp record missing expected field {self.group_by_sig.agg_fields[i]}")
                field = getattr(candidate, self.group_by_sig.agg_fields[i])
                state[i] = ApplyGroupByOp.agg_merge(fun, state[i], field)
            agg_state[group] = state

        # return list of data records (one per group)
        drs = []
        group_by_fields = self.group_by_sig.group_by_fields
        agg_fields = self.group_by_sig.get_agg_field_names()
        for g in agg_state:
            dr = DataRecord(self.group_by_sig.output_schema())
            # NOTE: this will set the parent_id and source_id to be the id of the final source record;
            #       in the near future we may want to have parent_id accept a list of ids
            dr = DataRecord.from_parent(
                schema=self.group_by_sig.output_schema(),
                parent_record=candidates[-1],
                project_cols=[],
            )
            for i in range(0, len(g)):
                k = g[i]
                setattr(dr, group_by_fields[i], k)
            vals = agg_state[g]
            for i in range(0, len(vals)):
                v = ApplyGroupByOp.agg_final(self.group_by_sig.agg_funcs[i], vals[i])
                setattr(dr, agg_fields[i], v)

            drs.append(dr)

        # create RecordOpStats objects
        total_time = time.time() - start_time
        record_op_stats_lst = []
        for dr in drs:
            record_op_stats = RecordOpStats(
                record_id=dr.id,
                record_parent_id=dr.parent_id,
                record_source_id=dr.source_id,
                record_state=dr.to_dict(include_bytes=False),
                op_id=self.get_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=total_time / len(drs),
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct and return DataRecordSet
        return DataRecordSet(drs, record_op_stats_lst)


class AverageAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        kwargs["output_schema"] = Number
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

        if not self.input_schema.get_desc() == Number.get_desc():
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
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
        dr = DataRecord.from_parent(schema=Number, parent_record=candidates[-1], project_cols=[])
        dr.value = sum(list(map(lambda c: float(c.value), candidates))) / len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr.id,
            record_parent_id=dr.parent_id,
            record_source_id=dr.source_id,
            record_state=dr.to_dict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return DataRecordSet([dr], [record_op_stats])


class CountAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        kwargs["output_schema"] = Number
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: DataRecordSet) -> DataRecordSet:
        start_time = time.time()

        # NOTE: this will set the parent_id to be the id of the final source record;
        #       in the near future we may want to have parent_id accept a list of ids
        dr = DataRecord.from_parent(schema=Number, parent_record=candidates[-1], project_cols=[])
        dr.value = len(candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr.id,
            record_parent_id=dr.parent_id,
            record_source_id=dr.source_id,
            record_state=dr.to_dict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])

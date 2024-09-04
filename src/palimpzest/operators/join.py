from __future__ import annotations

from palimpzest.corelib.schemas import Schema, Field
from palimpzest.generators.generators import DSPyGenerator, ImageTextGenerator
from .physical import PhysicalOperator, DataRecordsWithStats

from palimpzest.constants import *
from palimpzest.dataclasses import GenerationStats, RecordOpStats
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord

from typing import List

import base64
import time


class JoinOp(PhysicalOperator):
    def __init__(self, left: Schema, right:Schema, on: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert on in left.fieldNames(), "Left schema does not contain the field to join on"
        assert on in right.fieldNames(), "Right schema does not contain the field to join on"
        self.left = left
        self.right = right
        self.on = on
        self.inputSchema = left
        self.outputSchema = left + right

    def __str__(self):
        op = super().__str__()
        op += f"    Join: {self.left.className}-{self.right.className} on {self.on}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"left": self.left,
                "right":self.right,
                "on":self.on,
                **copy_kwargs}

    def get_op_params(self):
        return {
            "left": self.left,
            "right": self.right,
            "on": self.on,
        }

    def __eq__(self, other: JoinOp):
        return (
            isinstance(other, self.__class__)
            and self.left == other.left
            and self.right == other.right
            and self.on == other.on
        )


class NonLLMJoin(JoinOp):

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = 0.1 #NAIVE_EST_JOIN_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate 1 ms single-threaded execution for filter function
        time_per_record = 0.001 / self.max_workers

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidates: DataRecord) -> List[DataRecordsWithStats]:
        start_time = time.time()

        breakpoint()
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
                record_id=dr._id,
                record_parent_id=dr._parent_id,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=total_time / len(drs),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return drs, record_op_stats_lst

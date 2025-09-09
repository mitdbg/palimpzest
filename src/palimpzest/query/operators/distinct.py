from __future__ import annotations

from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator


class DistinctOp(PhysicalOperator):
    def __init__(self, distinct_cols: list[str], distinct_seen: set | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distinct_cols = distinct_cols
        self._distinct_seen = set() if distinct_seen is None else distinct_seen

    def __str__(self):
        op = super().__str__()
        op += f"    Distinct Cols: {self.distinct_cols}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"distinct_cols": self.distinct_cols, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"distinct_cols": self.distinct_cols, "distinct_seen": self._distinct_seen, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # assume applying the distinct operator takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        # create new DataRecord
        dr = DataRecord.from_parent(schema=candidate.schema, data_item={}, parent_record=candidate)

        # output record only if it has not been seen before
        record_str = dr.to_json_str(project_cols=self.distinct_cols, bytes_to_str=True, sorted=True)
        record_hash = f"{hash(record_str)}"
        dr._passed_operator = record_hash not in self._distinct_seen
        if dr._passed_operator:
            self._distinct_seen.add(record_hash)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
            passed_operator=dr._passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])

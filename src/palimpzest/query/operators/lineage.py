from __future__ import annotations

from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator

class ExplodeLineageOp(PhysicalOperator):
    """
    Physical operator that explodes a record into multiple records based on its lineage.
    For each parent_id in the input record, it emits a new record with fields:
    - src_node_id: the id of the input record (the "Super-Node")
    - dst_node_id: the id of the parent record (the "Child")
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return super().__str__()

    def get_id_params(self):
        return super().get_id_params()

    def get_op_params(self):
        return super().get_op_params()

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # Assume average fan-out of 10 for now (N-to-1 aggregation reversed)
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality * 10,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        parent_ids = candidate._parent_ids or []
        src_node_id = candidate._id
        
        results = []
        stats = []
        
        for parent_id in parent_ids:
            data_item = {
                "src_node_id": src_node_id,
                "dst_node_id": parent_id,
            }
            
            dr = DataRecord.from_parent(
                schema=self.output_schema,
                data_item=data_item,
                parent_record=candidate,
                project_cols=[], # Don't copy fields, just lineage
            )
            
            results.append(dr)
            
            stats.append(
                RecordOpStats(
                    record_id=dr._id,
                    record_parent_ids=dr._parent_ids,
                    record_source_indices=dr._source_indices,
                    record_state=dr.to_dict(include_bytes=False),
                    full_op_id=self.get_full_op_id(),
                    logical_op_id=self.logical_op_id,
                    op_name=self.op_name(),
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    op_details={},
                )
            )
            
        return DataRecordSet(results, stats, input=candidate)

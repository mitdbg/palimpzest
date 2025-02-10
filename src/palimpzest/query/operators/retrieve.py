from __future__ import annotations

import os
import time

from palimpzest.core.data.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator


class RetrieveOp(PhysicalOperator):
    def __init__(self, index, search_func, search_attr, output_attr, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.search_func = search_func
        self.search_attr = search_attr
        self.output_attr = output_attr
        self.k = k

    def __str__(self):
        op = super().__str__()
        op += f"    Retrieve: {str(self.index)} with top {self.k}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "index": self.index,
            "search_func": self.search_func,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
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

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        query = getattr(candidate, self.search_attr)

        try:
            top_k_results = self.search_func(self.index, query, self.k)
        except Exception:
            top_k_results = ["error-in-retrieve"]
            os.makedirs("retrieve-errors", exist_ok=True)
            ts = time.time()
            with open(f"retrieve-errors/error-{ts}.txt", "w") as f:
                f.write(str(query))

        output_dr = DataRecord.from_parent(self.output_schema, parent_record=candidate)
        setattr(output_dr, self.output_attr, top_k_results)

        duration_secs = time.time() - start_time
        answer = {self.output_attr: top_k_results}
        record_state = output_dr.to_dict(include_bytes=False)

        # NOTE: right now this should be equivalent to [self.output_attr], but in the future we may
        #       want to support the RetrieveOp generating multiple fields. (Also, the function will
        #       return the full field name (as opposed to the short field name))
        generated_fields = self.get_fields_to_generate(candidate, self.input_schema, self.output_schema)

        record_op_stats = RecordOpStats(
            record_id=output_dr.id,
            record_parent_id=output_dr.parent_id,
            record_source_id=output_dr.source_id,
            record_state=record_state,
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=duration_secs,
            cost_per_record=0.0,
            answer=answer,
            input_fields=self.input_schema.field_names(),
            generated_fields=generated_fields,
            fn_call_duration_secs=duration_secs,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        drs = [output_dr]
        record_op_stats_lst = [record_op_stats]

        # construct record set
        record_set = DataRecordSet(drs, record_op_stats_lst)

        return record_set

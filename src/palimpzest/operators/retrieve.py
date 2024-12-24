from __future__ import annotations

import os
import time

from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.elements.records import DataRecord, DataRecordSet
from palimpzest.operators.physical import PhysicalOperator


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

        top_k_results, top_k_result_doc_ids = [], []
        if isinstance(query, str):
            results = self.index.search(query, k=self.k)
            top_k_results = [result["content"] for result in results]

            # This is hacky, fix this later.
            top_k_result_doc_ids = list({result["document_id"] for result in results})

        elif isinstance(query, list):
            try:
                # retrieve top entry for each query
                results = self.index.search(query, k=1)

                # filter for the top-k entries
                results = [result[0] if isinstance(result, list) else result for result in results]
                sorted_results = sorted(results, key=lambda result: result["score"], reverse=True)
                top_k_results = [result["content"] for result in sorted_results[:self.k]]
                top_k_result_doc_ids = [result["document_id"] for result in sorted_results[:self.k]]
            except Exception:
                os.makedirs("retrieve-errors", exist_ok=True)
                ts = time.time()
                with open(f"retrieve-errors/error-{ts}.txt", "w") as f:
                    f.write(str(query))

                top_k_results = ["error-in-retrieve"]
                top_k_result_doc_ids = ["error-in-retrieve"]

        output_dr = DataRecord.from_parent(self.output_schema, parent_record=candidate)
        setattr(output_dr, self.output_attr, top_k_results)
        output_dr._evidence_file_ids = top_k_result_doc_ids

        duration_secs = time.time() - start_time

        # answer = (
        #     {
        #         field_name: getattr(output_dr, field_name)
        #         for field_name in self.output_schema.field_names()
        #     },
        # )
        answer = {self.output_attr: top_k_results}
        record_state = output_dr.as_dict(include_bytes=False)
        record_state["_evidence_file_ids"] = top_k_result_doc_ids
        generated_fields = self._generate_field_names(candidate, self.input_schema, self.output_schema)
        
        record_op_stats = RecordOpStats(
            record_id=output_dr._id,
            record_parent_id=output_dr._parent_id,
            record_source_id=output_dr._source_id,
            record_state=record_state,
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=duration_secs,
            cost_per_record=0.0,
            answer=answer,
            input_fields=self.input_schema.field_names(),
            generated_fields=generated_fields,
            fn_call_duration_secs=duration_secs,  # TODO(Siva): Currently tracking retrieval time in fn_call_duration_secs
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        drs = [output_dr]
        record_op_stats_lst = [record_op_stats]

        # construct record set
        record_set = DataRecordSet(drs, record_op_stats_lst)

        return record_set

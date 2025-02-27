from __future__ import annotations

import time
from typing import Callable

from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.fields import Field
from palimpzest.query.operators.physical import PhysicalOperator


class MapOp(PhysicalOperator):
    def __init__(self, udf: Callable | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.udf = udf

    def __str__(self):
        op = super().__str__()
        op += f"    UDF: {self.udf.__name__}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"udf": self.udf, **id_params}

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"udf": self.udf, **op_params}

        return op_params

    def _create_record_set(
        self,
        record: DataRecord,
        generation_stats: GenerationStats,
        total_time: float,
    ) -> DataRecordSet:
        """
        Given an input DataRecord and a determination of whether it passed the filter or not,
        construct the resulting RecordSet.
        """
        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=record.id,
            record_parent_id=record.parent_id,
            record_source_idx=record.source_idx,
            record_state=record.to_dict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=total_time,
            cost_per_record=0.0,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            answer=record.to_dict(include_bytes=False),
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([record], [record_op_stats])

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the Map operation. These estimates assume that the map UDF
        (1) has no cost and (2) has perfect quality.
        """
        # estimate 1 ms single-threaded execution for udf function
        time_per_record = 0.001

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    def map(self, candidate: DataRecord, fields: dict[str, Field]) -> tuple[dict[str, list], GenerationStats]:
        # apply UDF to input record
        start_time = time.time()
        field_answers = {}
        try:
            # execute the UDF function
            field_answers = self.udf(candidate.to_dict())

            # answer should be a dictionary
            assert isinstance(field_answers, dict), (
                "UDF must return a dictionary mapping each input field to its value for map operations"
            )

            if self.verbose:
                print(f"{self.udf.__name__}")

        except Exception as e:
            print(f"Error invoking user-defined function for map: {e}")
            raise e

        # create generation stats object containing the time spent executing the UDF function
        generation_stats = GenerationStats(fn_call_duration_secs=time.time() - start_time)

        return field_answers, generation_stats


    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        """
        This method converts an input DataRecord into an output DataRecordSet. The output DataRecordSet contains the
        DataRecord(s) output by the operator's convert() method and their corresponding RecordOpStats objects.
        Some subclasses may override this __call__method to implement their own custom logic.
        """
        start_time = time.time()

        # execute the map operation
        field_answers: dict[str, list]
        fields = {field: field_type for field, field_type in self.output_schema.field_map().items()}
        field_answers, generation_stats = self.map(candidate=candidate, fields=fields)
        assert all([field in field_answers for field in fields]), "Not all fields are present in output of map!"

        # construct DataRecord from field_answers
        dr = DataRecord.from_parent(schema=self.output_schema, parent_record=candidate)
        for field_name, field_value in field_answers.items():
            dr[field_name] = field_value

        # construct and return DataRecordSet
        record_set = self._create_record_set(
            record=dr,
            generation_stats=generation_stats,
            total_time=time.time() - start_time,
        )

        return record_set

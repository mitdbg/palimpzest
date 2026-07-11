from __future__ import annotations

import threading
import time

from pydantic.fields import FieldInfo

from palimpzest.constants import (
    NAIVE_EST_FILTER_SELECTIVITY,
    NAIVE_EST_NUM_INPUT_TOKENS,
    Cardinality,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates
from palimpzest.query.generators.generators import get_json_from_answer
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator


class BatchedOperator(PhysicalOperator):
    def flush(self) -> DataRecordSet:
        raise NotImplementedError(
            "flush method must be implemented by BatchedOperator subclasses"
        )


class BatchedFilter(LLMFilter, BatchedOperator):
    def __init__(
        self,
        batch_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self._buffer: list[DataRecord] = []
        self._buffer_lock = threading.Lock()
        self.flushed = False

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"batch_size": self.batch_size, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"batch_size": self.batch_size, **op_params}

    def set_flushed(self):
        self.flushed = True

    def has_pending_batch(self) -> bool:
        with self._buffer_lock:
            return len(self._buffer) > 0

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS if not self.is_image_op() else (765 / 10)
        batch_num_input_tokens = 0.1 * est_input_tokens + (0.9*self.batch_size * est_input_tokens)

        # NOTE: the output often generates an entire reasoning sentence, thus the true value may be higher
        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        batch_est_num_output_tokens = self.batch_size + (0.25 * est_input_tokens)

        # get est. of conversion time per batch from model card
        model_conversion_time_per_batch = (
            self.model.get_seconds_per_output_token() * batch_est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per batch from model card
        usd_per_input_token = (
            self.model.get_usd_per_audio_input_token()
            if self.is_audio_op()
            else self.model.get_usd_per_input_token()
        )
        model_conversion_usd_per_batch = (
            usd_per_input_token * batch_num_input_tokens
            + self.model.get_usd_per_output_token() * batch_est_num_output_tokens
        )

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (self.model.get_overall_score() / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_batch / self.batch_size,
            cost_per_record=model_conversion_usd_per_batch / self.batch_size,
            quality=quality,
        )

    def _build_filter_condition(self) -> str:
        return (
            f"{self.filter_obj.filter_condition}\n"
            "Return a JSON array of booleans with one entry per input record in order."
        )

    def _parse_batch_answer(self, completion_text: str, expected_len: int) -> list[bool]:
        try:
            parsed = get_json_from_answer(completion_text, self.model, Cardinality.ONE_TO_MANY)
            if not isinstance(parsed, list) or len(parsed) != expected_len:
                raise Exception

            if not all(isinstance(value, bool) for value in parsed):
                raise Exception
            return parsed

        except Exception:
            return [False] * expected_len

    def _process_batch(self, candidates: list[DataRecord]) -> DataRecordSet:
        if len(candidates) == 0:
            return DataRecordSet([], [])

        start_time = time.time()
        input_fields = self.get_input_fields()
        filter_condition = self._build_filter_condition()
        gen_kwargs = {"project_cols": input_fields, "filter_condition": filter_condition}

        fields = {
            "passed_operator": FieldInfo(
                annotation=bool,
                description="Whether the record passed the filter operation",
            )
        }
        expected_len = len(candidates)

        def parse_answer(completion_text: str) -> list[bool]:
            return self._parse_batch_answer(completion_text, expected_len)

        field_answers, _, generation_stats, _ = self.generator(
            candidates,
            fields,
            parse_answer=parse_answer,
            **gen_kwargs,
        )

        answers = field_answers if isinstance(field_answers, list) else []
        if len(answers) != expected_len or any(not isinstance(value, bool) for value in answers):
            answers = [False] * expected_len

        elapsed = time.time() - start_time
        per_record_stats = generation_stats / expected_len
        per_record_time = elapsed / expected_len

        records, record_op_stats = [], []
        for candidate, passed_operator in zip(candidates, answers):
            record_set = self._create_record_set(
                candidate,
                passed_operator,
                per_record_stats,
                per_record_time,
                {"passed_operator": passed_operator},
            )
            records.extend(record_set.data_records)
            record_op_stats.extend(record_set.record_op_stats)

        return DataRecordSet(records, record_op_stats, input=candidates)

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        with self._buffer_lock:
            if self.flushed:
                self.flushed = False
            self._buffer.append(candidate)
            if len(self._buffer) < self.batch_size:
                # return empty result until we have a full batch to process
                return DataRecordSet([], [])
            else:
                batch = list(self._buffer)
                self._buffer.clear()

        # release the lock before processing
        return self._process_batch(batch)

    def flush(self) -> DataRecordSet:
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return DataRecordSet([], [])
            batch = list(self._buffer)
            self._buffer.clear()

        return self._process_batch(batch)

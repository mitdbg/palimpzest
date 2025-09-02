from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic.fields import FieldInfo

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_JOIN_SELECTIVITY,
    NAIVE_EST_NUM_INPUT_TOKENS,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator


class JoinOp(PhysicalOperator, ABC):
    def __init__(self, condition: str, desc: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.input_schema == self.output_schema, "Input and output schemas must match for JoinOp"
        self.condition = condition
        self.desc = desc

    def __str__(self):
        op = super().__str__()
        op += f"    Condition: {self.condition}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"condition": self.condition, "desc": self.desc, **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"condition": self.condition, "desc": self.desc, **op_params}

    @abstractmethod
    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        pass


class BlockingNestedLoopsJoin(JoinOp):
    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.JOIN,
        join_parallelism: int = 64,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)
        self.join_idx = 0

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }

        return op_params

    def get_model_name(self):
        return self.model.value

    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = 2 * NAIVE_EST_NUM_INPUT_TOKENS
        if self.is_image_op():
            est_num_input_tokens = 2 * 765 / 10  # 1024x1024 image is 765 tokens

        # NOTE: the output often generates an entire reasoning sentence, thus the true value may be higher
        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        usd_per_input_token = (
            MODEL_CARDS[self.model.value]["usd_per_audio_input_token"]
            if self.is_audio_op()
            else MODEL_CARDS[self.model.value]["usd_per_input_token"]
        )
        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_JOIN_SELECTIVITY
        cardinality = selectivity * (left_source_op_cost_estimates.cardinality * right_source_op_cost_estimates.cardinality)

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["overall"] / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def _process_join_candidate_pair(
        self,
        left_candidate: DataRecord,
        right_candidate: DataRecord,
        gen_kwargs: dict,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        start_time = time.time()

        # generate output; NOTE: FieldInfo is used to indicate the output type; thus, the desc is not needed
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the records satisfy the join condition")}
        field_answers, _, generation_stats, _ = self.generator(left_candidate, fields, right_candidate=right_candidate, **gen_kwargs)

        # determine whether or not the join was satisfied
        passed_operator = field_answers["passed_operator"]

        # compute output record and add to output_records
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator

        # compute record stats and add to output_record_op_stats
        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=self.get_model_name(),
            join_condition=self.condition,
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            answer=field_answers,
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return [join_dr], [record_op_stats]

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord]) -> tuple[DataRecordSet, int]:
        # get the set of input fields from both records in the join
        input_fields = self.get_input_fields()

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "join_condition": self.condition}

        # apply the generator to each pair of candidates
        output_records, output_record_op_stats, num_inputs_processed = [], [], 0
        total_join_candidates = len(left_candidates) * len(right_candidates)
        with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
            futures = []
            for candidate in left_candidates:
                for right_candidate in right_candidates:
                    futures.append(executor.submit(self._process_join_candidate_pair, candidate, right_candidate, gen_kwargs))
                    num_inputs_processed += 1

            for future in as_completed(futures):
                self.join_idx += 1
                join_output_records, join_output_record_op_stats = future.result()
                output_records.extend(join_output_records)
                output_record_op_stats.extend(join_output_record_op_stats)
                print(f"{self.join_idx}/{total_join_candidates} JOINED")

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed


class NestedLoopsJoin(JoinOp):
    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.JOIN,
        join_parallelism: int = 64,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)
        self.join_idx = 0

        # maintain list(s) of input records for the join
        self._left_input_records: list[DataRecord] = []
        self._right_input_records: list[DataRecord] = []

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }

        return op_params

    def get_model_name(self):
        return self.model.value

    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = 2 * NAIVE_EST_NUM_INPUT_TOKENS
        if self.is_image_op():
            est_num_input_tokens = 2 * 765 / 10  # 1024x1024 image is 765 tokens

        # NOTE: the output often generates an entire reasoning sentence, thus the true value may be higher
        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        usd_per_input_token = (
            MODEL_CARDS[self.model.value]["usd_per_audio_input_token"]
            if self.is_audio_op()
            else MODEL_CARDS[self.model.value]["usd_per_input_token"]
        )
        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_JOIN_SELECTIVITY
        cardinality = selectivity * (left_source_op_cost_estimates.cardinality * right_source_op_cost_estimates.cardinality)

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["overall"] / 100.0)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def _process_join_candidate_pair(
        self,
        left_candidate: DataRecord,
        right_candidate: DataRecord,
        gen_kwargs: dict,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        start_time = time.time()

        # generate output; NOTE: FieldInfo is used to indicate the output type; thus, the desc is not needed
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the records satisfy the join condition")}
        field_answers, _, generation_stats, _ = self.generator(left_candidate, fields, right_candidate=right_candidate, **gen_kwargs)

        # determine whether or not the join was satisfied
        passed_operator = field_answers["passed_operator"]

        # compute output record and add to output_records
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator

        # compute record stats and add to output_record_op_stats
        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=self.get_model_name(),
            join_condition=self.condition,
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            answer=field_answers,
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return [join_dr], [record_op_stats]

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord]) -> tuple[DataRecordSet | None, int]:
        # get the set of input fields from both records in the join
        input_fields = self.get_input_fields()

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "join_condition": self.condition}

        # apply the generator to each pair of candidates
        output_records, output_record_op_stats, num_inputs_processed = [], [], 0
        with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
            futures = []
            # join new left candidates with new right candidates
            for candidate in left_candidates:
                for right_candidate in right_candidates:
                    futures.append(executor.submit(self._process_join_candidate_pair, candidate, right_candidate, gen_kwargs))
                    num_inputs_processed += 1

            # join new left candidates with stored right input records
            for candidate in left_candidates:
                for right_candidate in self._right_input_records:
                    futures.append(executor.submit(self._process_join_candidate_pair, candidate, right_candidate, gen_kwargs))
                    num_inputs_processed += 1

            # join new right candidates with stored left input records
            for candidate in self._left_input_records:
                for right_candidate in right_candidates:
                    futures.append(executor.submit(self._process_join_candidate_pair, candidate, right_candidate, gen_kwargs))
                    num_inputs_processed += 1

            # collect results as they complete
            for future in as_completed(futures):
                self.join_idx += 1
                join_output_records, join_output_record_op_stats = future.result()
                output_records.extend(join_output_records)
                output_record_op_stats.extend(join_output_record_op_stats)
                print(f"{self.join_idx} JOINED")

        # store input records to join with new records added later
        self._left_input_records.extend(left_candidates)
        self._right_input_records.extend(right_candidates)

        # return None if no output records were produced
        if len(output_records) == 0:
            return None, num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from PIL import Image
from pydantic.fields import FieldInfo
from sentence_transformers import SentenceTransformer

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_JOIN_SELECTIVITY,
    NAIVE_EST_NUM_INPUT_TOKENS,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import AUDIO_FIELD_TYPES, IMAGE_FIELD_TYPES, ImageFilepath
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator


class Singleton:
     def __new__(cls, *args, **kw):
         if not hasattr(cls, '_instance'):
             orig = super(Singleton, cls)  # noqa: UP008
             cls._instance = orig.__new__(cls, *args, **kw)
         return cls._instance

class Locks(Singleton):
    model = None
    clip_lock = threading.Lock()
    exec_lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str):
        with cls.clip_lock:
            if cls.model is None:
                cls.model = SentenceTransformer(model_name)
            return cls.model

def compute_similarity(left_embedding: list[float], right_embedding: list[float]) -> float:
    """
    Compute the similarity between two embeddings using cosine similarity.
    """
    return np.dot(left_embedding, right_embedding) / (norm(left_embedding) * norm(right_embedding))


class JoinOp(PhysicalOperator, ABC):
    def __init__(
        self,
        condition: str,
        how: str = "inner",
        on: list[str] | None = None,
        join_parallelism: int = 64,
        retain_inputs: bool = True,
        desc: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.input_schema == self.output_schema, "Input and output schemas must match for JoinOp"
        self.condition = condition
        self.how = how
        self.on = on
        self.join_parallelism = join_parallelism
        self.retain_inputs = retain_inputs
        self.desc = desc
        self.join_idx = 0
        self.finished = False

        # maintain list(s) of input records for the join
        self._left_input_records: list[DataRecord] = []
        self._right_input_records: list[DataRecord] = []

        # maintain set of left/right record ids that have been joined (for left/right/outer joins)
        self._left_joined_record_ids: set[str] = set()
        self._right_joined_record_ids: set[str] = set()

    def __str__(self):
        op = super().__str__()
        op += f"    Condition: {self.condition}\n"
        op += f"    How: {self.how}\n"
        op += f"    On: {self.on}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "condition": self.condition,
            "join_parallelism": self.join_parallelism,
            "desc": self.desc,
            "how": self.how,
            "on": self.on,
            **id_params,
        }
        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "condition": self.condition,
            "join_parallelism": self.join_parallelism,
            "retain_inputs": self.retain_inputs,
            "desc": self.desc,
            "how": self.how,
            "on": self.on,
            **op_params,
        }
        return op_params

    def _compute_unmatched_records(self) -> DataRecordSet:
        """Helper function to compute unmatched records for left/right/outer joins."""
        def join_unmatched_records(input_records: list[DataRecord] | list[tuple[DataRecord, list[float]]], joined_record_ids: set[str], left: bool = True):
            records, record_op_stats_lst = [], []
            for record in input_records:
                start_time = time.time()
                record = record[0] if isinstance(record, tuple) else record
                if record._id not in joined_record_ids:
                    unmatched_dr = (
                        DataRecord.from_join_parents(self.output_schema, record, None)
                        if left
                        else DataRecord.from_join_parents(self.output_schema, None, record)
                    )
                    unmatched_dr._passed_operator = True

                    # compute record stats and add to output_record_op_stats
                    time_per_record = time.time() - start_time
                    record_op_stats = RecordOpStats(
                        record_id=unmatched_dr._id,
                        record_parent_ids=unmatched_dr._parent_ids,
                        record_source_indices=unmatched_dr._source_indices,
                        record_state=unmatched_dr.to_dict(include_bytes=False),
                        full_op_id=self.get_full_op_id(),
                        logical_op_id=self.logical_op_id,
                        op_name=self.op_name(),
                        time_per_record=time_per_record,
                        cost_per_record=0.0,
                        model_name=self.get_model_name(),
                        join_condition=str(self.on),
                        fn_call_duration_secs=time_per_record,
                        answer={"passed_operator": True},
                        passed_operator=True,
                        op_details={k: str(v) for k, v in self.get_id_params().items()},
                    )
                    records.append(unmatched_dr)
                    record_op_stats_lst.append(record_op_stats)
            return records, record_op_stats_lst

        records, record_op_stats = [], []
        if self.how == "left":
            records, record_op_stats = join_unmatched_records(self._left_input_records, self._left_joined_record_ids, left=True)

        elif self.how == "right":
            records, record_op_stats = join_unmatched_records(self._right_input_records, self._right_joined_record_ids, left=False)

        elif self.how == "outer":
            records, record_op_stats = join_unmatched_records(self._left_input_records, self._left_joined_record_ids, left=True)
            right_records, right_record_op_stats = join_unmatched_records(self._right_input_records, self._right_joined_record_ids, left=False)
            records.extend(right_records)
            record_op_stats.extend(right_record_op_stats)

        return DataRecordSet(records, record_op_stats)

    @abstractmethod
    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        pass

    def set_finished(self):
        """Mark the operator as finished after computing left/right/outer join logic."""
        self.finished = True

class RelationalJoin(JoinOp):

    def get_model_name(self):
        return None
    
    def _process_join_candidate_pair(self, left_candidate, right_candidate) -> tuple[DataRecord, RecordOpStats]:
        start_time = time.time()

        # determine whether or not the join was satisfied
        passed_operator = all(
            left_candidate[field] == right_candidate[field]
            for field in self.on
        )

        # handle different join types
        if self.how == "left" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
        elif self.how == "right" and passed_operator:
            self._right_joined_record_ids.add(right_candidate._id)
        elif self.how == "outer" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
            self._right_joined_record_ids.add(right_candidate._id)

        # compute output record and add to output_records
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator

        # compute record stats and add to output_record_op_stats
        time_per_record = time.time() - start_time
        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time_per_record,
            cost_per_record=0.0,
            model_name=self.get_model_name(),
            join_condition=str(self.on),
            fn_call_duration_secs=time_per_record,
            answer={"passed_operator": passed_operator},
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return join_dr, record_op_stats

    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates):
        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_JOIN_SELECTIVITY
        cardinality = selectivity * (left_source_op_cost_estimates.cardinality * right_source_op_cost_estimates.cardinality)

        # estimate 1 ms execution time per input record pair
        time_per_record = 0.001 * (left_source_op_cost_estimates.cardinality + right_source_op_cost_estimates.cardinality)

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord], final: bool = False) -> tuple[DataRecordSet, int]:
        # create the set of candidates to join
        join_candidates = []
        for candidate in left_candidates:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))
            for right_candidate in self._right_input_records:
                join_candidates.append((candidate, right_candidate))
        for candidate in self._left_input_records:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))

        # apply the join logic to each pair of candidates
        output_records, output_record_op_stats = [], []
        with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
            futures = [
                executor.submit(self._process_join_candidate_pair, candidate, right_candidate)
                for candidate, right_candidate in join_candidates
            ]
  
            # collect results as they complete
            for future in as_completed(futures):
                self.join_idx += 1
                join_output_record, join_output_record_op_stats = future.result()
                output_records.append(join_output_record)
                output_record_op_stats.append(join_output_record_op_stats)

        # compute the number of inputs processed
        num_inputs_processed = len(join_candidates)

        # store input records to join with new records added later
        if self.retain_inputs:
            self._left_input_records.extend(left_candidates)
            self._right_input_records.extend(right_candidates)
        
        # if this is the final call, then add in any left/right/outer join records that did not match
        if final:
            return self._compute_unmatched_records(), 0

        # return empty DataRecordSet if no output records were produced
        if len(output_records) == 0:
            return DataRecordSet([], []), num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed



class LLMJoin(JoinOp):
    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.JOIN,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)

    def __str__(self):
        op = super().__str__()
        op += f"    Model: {self.model.value}\n"
        op += f"    Reasoning Effort: {self.reasoning_effort}\n"
        op += f"    Prompt Strategy: {self.prompt_strategy.value}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }
        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }
        return op_params

    def get_model_name(self):
        return self.model.value

    def _process_join_candidate_pair(
        self,
        left_candidate: DataRecord,
        right_candidate: DataRecord,
        gen_kwargs: dict,
    ) -> tuple[DataRecord, RecordOpStats]:
        start_time = time.time()

        # generate output; NOTE: FieldInfo is used to indicate the output type; thus, the desc is not needed
        fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the records satisfy the join condition")}
        field_answers, _, generation_stats, _ = self.generator(left_candidate, fields, right_candidate=right_candidate, **gen_kwargs)

        # determine whether or not the join was satisfied
        passed_operator = field_answers["passed_operator"]

        # handle different join types
        if self.how == "left" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
        elif self.how == "right" and passed_operator:
            self._right_joined_record_ids.add(right_candidate._id)
        elif self.how == "outer" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
            self._right_joined_record_ids.add(right_candidate._id)

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
            total_embedding_input_tokens=generation_stats.total_embedding_input_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            total_embedding_cost=generation_stats.total_embedding_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            answer=field_answers,
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return join_dr, record_op_stats


class NestedLoopsJoin(LLMJoin):

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

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord], final: bool = False) -> tuple[DataRecordSet, int]:
        # get the set of input fields from both records in the join
        input_fields = self.get_input_fields()

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "join_condition": self.condition}

        # create the set of candidates to join
        join_candidates = []
        for candidate in left_candidates:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))
            for right_candidate in self._right_input_records:
                join_candidates.append((candidate, right_candidate))
        for candidate in self._left_input_records:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))

        # apply the generator to each pair of candidates
        output_records, output_record_op_stats = [], []
        with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
            futures = [
                executor.submit(self._process_join_candidate_pair, candidate, right_candidate, gen_kwargs)
                for candidate, right_candidate in join_candidates
            ]
  
            # collect results as they complete
            for future in as_completed(futures):
                self.join_idx += 1
                join_output_record, join_output_record_op_stats = future.result()
                output_records.append(join_output_record)
                output_record_op_stats.append(join_output_record_op_stats)
                print(f"{self.join_idx} JOINED")

        # compute the number of inputs processed
        num_inputs_processed = len(join_candidates)

        # store input records to join with new records added later
        if self.retain_inputs:
            self._left_input_records.extend(left_candidates)
            self._right_input_records.extend(right_candidates)

        # if this is the final call, then add in any left/right/outer join records that did not match
        if final:
            return self._compute_unmatched_records(), 0

        # return empty DataRecordSet if no output records were produced
        if len(output_records) == 0:
            return DataRecordSet([], []), num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed


class EmbeddingJoin(LLMJoin):
    # NOTE: we currently do not support audio joins as embedding models for audio seem to have
    # specialized use cases (e.g., speech-to-text) with strict requirements on things like e.g. sample rate
    def __init__(
        self,
        num_samples: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.samples_drawn = 0

        # compute whether all fields are text fields
        self.text_only = all([
            field.annotation not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
            for field_name, field in self.input_schema.model_fields.items()
            if field_name.split(".")[-1] in self.get_input_fields()
        ])
        self.embedding_model = Model.TEXT_EMBEDDING_3_SMALL if self.text_only else Model.CLIP_VIT_B_32
        self.locks = Locks()

        # keep track of embedding costs that could not be amortized if no output records were produced
        self.residual_embedding_cost = 0.0

        # crude adjustment factor for naive estimation in unoptimized setting
        self.naive_quality_adjustment = 0.6

        # maintain list(s) of input records and their embeddings for the join
        self._left_input_records: list[tuple[DataRecord, list[float]]] = []
        self._right_input_records: list[tuple[DataRecord, list[float]]] = []

        # maintain lowest and highest embedding similarities for matching and non-matching pairs
        self.min_matching_sim = float("inf")
        self.max_non_matching_sim = float("-inf")

    def __str__(self):
        op = super().__str__()
        op += f"    Num Samples: {self.num_samples}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "num_samples": self.num_samples,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "num_samples": self.num_samples,
            **op_params,
        }

        return op_params

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
            MODEL_CARDS[self.embedding_model.value]["seconds_per_output_token"] * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = MODEL_CARDS[self.embedding_model.value]["usd_per_input_token"] * est_num_input_tokens

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_JOIN_SELECTIVITY
        cardinality = selectivity * (left_source_op_cost_estimates.cardinality * right_source_op_cost_estimates.cardinality)

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["overall"] / 100.0) * self.naive_quality_adjustment

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def _compute_embeddings(self, candidates: list[DataRecord], input_fields: list[str]) -> tuple[np.ndarray, GenerationStats]:
        # return empty array and empty stats if no candidates  
        if len(candidates) == 0:
            return np.zeros((0, 512)), GenerationStats()

        start_time = time.time()
        total_embedding_input_tokens = 0
        embeddings = None
        if self.text_only:
            client = OpenAI()
            inputs = [dr.to_json_str(bytes_to_str=True, project_cols=input_fields, sorted=True) for dr in candidates]
            response = client.embeddings.create(input=inputs, model=self.embedding_model.value)
            total_embedding_input_tokens = response.usage.total_tokens
            embeddings = np.array([item.embedding for item in response.data])
        else:
            model = self.locks.get_model(self.embedding_model.value)
            embeddings = np.zeros((len(candidates), 512))  # CLIP embeddings are 512-dimensional
            num_input_fields_present = 0
            for field in input_fields:
                field_inputs = []
                for candidate in candidates:
                    if field not in candidate.get_field_names():
                        continue
                    num_input_fields_present += 1
                    field_type = candidate.get_field_type(field)
                    if field_type in [ImageFilepath]:
                        field_inputs.append(Image.open(candidate[field]))
                    else:
                        field_inputs.append(str(candidate[field]))
                
                if len(field_inputs) > 0:
                    embeddings += model.encode(field_inputs, convert_to_numpy=True)

            # average embeddings over input fields present in candidates
            embeddings /= num_input_fields_present

        # compute cost of embedding(s)
        model_card = MODEL_CARDS[self.embedding_model.value]
        total_embedding_cost = model_card["usd_per_input_token"] * total_embedding_input_tokens
        embedding_gen_stats = GenerationStats(
            model_name=self.embedding_model.value,
            total_input_tokens=0.0,
            total_output_tokens=0.0,
            total_embedding_input_tokens=total_embedding_input_tokens,
            total_input_cost=0.0,
            total_output_cost=0.0,
            total_embedding_cost=total_embedding_cost,
            cost_per_record=total_embedding_cost,
            llm_call_duration_secs=time.time() - start_time,
            total_llm_calls=1,
            total_embedding_llm_calls=len(candidates),
        )

        return embeddings, embedding_gen_stats

    def _process_join_candidate_pair(self, left_candidate, right_candidate, gen_kwargs, embedding_sim):
        output_record, output_record_op_stats = super()._process_join_candidate_pair(left_candidate, right_candidate, gen_kwargs)
        return output_record, output_record_op_stats, embedding_sim

    def _process_join_candidate_with_sim(self, left_candidate: DataRecord, right_candidate: DataRecord, embedding_sim: float, passed_operator: bool) -> tuple[DataRecord, RecordOpStats]:
        # compute output record and add to output_records
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator

        # handle different join types
        if self.how == "left" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
        elif self.how == "right" and passed_operator:
            self._right_joined_record_ids.add(right_candidate._id)
        elif self.how == "outer" and passed_operator:
            self._left_joined_record_ids.add(left_candidate._id)
            self._right_joined_record_ids.add(right_candidate._id)

        # NOTE: embedding costs are amortized over all records and added at the end of __call__
        # compute record stats and add to output_record_op_stats
        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=0.0,
            cost_per_record=0.0,
            model_name=self.get_model_name(),
            join_condition=self.condition,
            answer={"passed_operator": passed_operator},
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return join_dr, record_op_stats, embedding_sim

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord], final: bool = False) -> tuple[DataRecordSet, int]:
        # get the set of input fields from both records in the join
        input_fields = self.get_input_fields()

        # compute the embeding for each candidate
        left_embeddings, left_embedding_gen_stats = self._compute_embeddings(left_candidates, input_fields)
        right_embeddings, right_embedding_gen_stats = self._compute_embeddings(right_candidates, input_fields)
        total_embedding_cost = left_embedding_gen_stats.cost_per_record + right_embedding_gen_stats.cost_per_record + self.residual_embedding_cost
        self.residual_embedding_cost = 0.0

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "join_condition": self.condition}

        # TODO: add embeddings to join candidates
        # create the set of candidates to join
        join_candidates = []
        for candidate, embedding in zip(left_candidates, left_embeddings):
            for right_candidate, right_embedding in zip(right_candidates, right_embeddings):
                embedding_sim = compute_similarity(embedding, right_embedding)
                join_candidates.append((candidate, right_candidate, embedding_sim))
            for right_candidate, right_embedding in self._right_input_records:
                embedding_sim = compute_similarity(embedding, right_embedding)
                join_candidates.append((candidate, right_candidate, embedding_sim))
        for candidate, embedding in self._left_input_records:
            for right_candidate, right_embedding in zip(right_candidates, right_embeddings):
                embedding_sim = compute_similarity(embedding, right_embedding)
                join_candidates.append((candidate, right_candidate, embedding_sim))

        # prepare list of output records and their stats
        output_records, output_record_op_stats, num_inputs_processed = [], [], 0

        # draw samples until num_samples is reached
        with self.locks.exec_lock:
            if self.samples_drawn < self.num_samples:
                samples_to_draw = min(self.num_samples - self.samples_drawn, len(join_candidates))
                join_candidate_samples = join_candidates[:samples_to_draw]
                join_candidates = join_candidates[samples_to_draw:]

                # apply the generator to each pair of candidates
                with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
                    futures = [
                        executor.submit(self._process_join_candidate_pair, left_candidate, right_candidate, gen_kwargs, embedding_sim)
                        for left_candidate, right_candidate, embedding_sim in join_candidate_samples
                    ]

                    # collect results as they complete
                    similarities, joined = [], []
                    for future in as_completed(futures):
                        self.join_idx += 1
                        join_output_record, join_output_record_op_stats, embedding_sim = future.result()
                        output_records.append(join_output_record)
                        output_record_op_stats.append(join_output_record_op_stats)
                        similarities.append(embedding_sim)
                        joined.append(join_output_record._passed_operator)
                        print(f"{self.join_idx} JOINED")

                    # sort join results by embedding similarity
                    sorted_sim_join_tuples = sorted(zip(similarities, joined), key=lambda x: x[0])

                    # compute threshold below which no records joined
                    for embedding_sim, records_joined in sorted_sim_join_tuples:
                        if records_joined:
                            break
                        if not records_joined and embedding_sim > self.max_non_matching_sim:
                            self.max_non_matching_sim = embedding_sim

                    # compute threshold above which all records joined
                    for embedding_sim, records_joined in reversed(sorted_sim_join_tuples):
                        if not records_joined:
                            break
                        if records_joined and embedding_sim < self.min_matching_sim:
                            self.min_matching_sim = embedding_sim

                # update samples drawn and num_inputs_processed
                self.samples_drawn += samples_to_draw
                num_inputs_processed += samples_to_draw

        # process remaining candidates based on embedding similarity
        if len(join_candidates) > 0:
             assert self.samples_drawn >= self.num_samples, "All samples should have been drawn before processing remaining candidates"
             with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
                futures = []
                for left_candidate, right_candidate, embedding_sim in join_candidates:
                    # if the embedding similarity is lower than the threshold below which no records joined,
                    # then we can skip the LLM call and mark the records as not joined
                    if embedding_sim < self.max_non_matching_sim:
                        futures.append(executor.submit(self._process_join_candidate_with_sim, left_candidate, right_candidate, embedding_sim, passed_operator=False))

                    # if the embedding similarity is higher than the threshold above which all records joined,
                    # then we can skip the LLM call and mark the records as joined
                    elif embedding_sim > self.min_matching_sim:
                        futures.append(executor.submit(self._process_join_candidate_with_sim, left_candidate, right_candidate, embedding_sim, passed_operator=True))

                    # otherwise, we will process the LLM call
                    else:
                        futures.append(executor.submit(self._process_join_candidate_pair, left_candidate, right_candidate, gen_kwargs, embedding_sim))

                    num_inputs_processed += 1

                # collect results as they complete
                similarities, joined = [], []
                for future in as_completed(futures):
                    self.join_idx += 1
                    join_output_record, join_output_record_op_stats, embedding_sim = future.result()
                    output_records.append(join_output_record)
                    output_record_op_stats.append(join_output_record_op_stats)
                    similarities.append(embedding_sim)
                    joined.append(join_output_record._passed_operator)
                    print(f"{self.join_idx} JOINED")

                ### update thresholds if there are llm calls which incrementally squeeze the boundaries ###
                # sort join results by embedding similarity
                sorted_sim_join_tuples = sorted(zip(similarities, joined), key=lambda x: x[0])

                # potentially update threshold below which no records joined
                for embedding_sim, records_joined in sorted_sim_join_tuples:
                    if records_joined:
                        break
                    if not records_joined and embedding_sim > self.max_non_matching_sim:
                        self.max_non_matching_sim = embedding_sim

                # potentially update threshold above which all records joined
                for embedding_sim, records_joined in reversed(sorted_sim_join_tuples):
                    if not records_joined:
                        break
                    if records_joined and embedding_sim < self.min_matching_sim:
                        self.min_matching_sim = embedding_sim

        # amortize embedding costs over all output records and add to each record's op stats
        amortized_embedding_cost = total_embedding_cost / len(output_record_op_stats) if len(output_record_op_stats) > 0 else 0.0
        for record_op_stats in output_record_op_stats:
            record_op_stats.cost_per_record += amortized_embedding_cost
            record_op_stats.total_embedding_cost = amortized_embedding_cost

        # store input records to join with new records added later
        if self.retain_inputs:
            self._left_input_records.extend(zip(left_candidates, left_embeddings))
            self._right_input_records.extend(zip(right_candidates, right_embeddings))

        # if this is the final call, then add in any left/right/outer join records that did not match
        if final:
            return self._compute_unmatched_records(), 0

        # return empty DataRecordSet if no output records were produced
        if len(output_records) == 0:
            self.residual_embedding_cost = total_embedding_cost
            return DataRecordSet([], []), num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed

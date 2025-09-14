from __future__ import annotations

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


def compute_similarity(left_embedding: list[float], right_embedding: list[float]) -> float:
    """
    Compute the similarity between two embeddings using cosine similarity.
    """
    return np.dot(left_embedding, right_embedding) / (norm(left_embedding) * norm(right_embedding))


class JoinOp(PhysicalOperator, ABC):
    def __init__(
        self,
        condition: str,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.JOIN,
        join_parallelism: int = 64,
        reasoning_effort: str | None = None,
        retain_inputs: bool = True,
        desc: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.input_schema == self.output_schema, "Input and output schemas must match for JoinOp"
        self.condition = condition
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.retain_inputs = retain_inputs
        self.desc = desc
        self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base, Cardinality.ONE_TO_ONE, self.desc, self.verbose)
        self.join_idx = 0

        # maintain list(s) of input records for the join
        self._left_input_records: list[DataRecord] = []
        self._right_input_records: list[DataRecord] = []

    def __str__(self):
        op = super().__str__()
        op += f"    Condition: {self.condition}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "condition": self.condition,
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            "desc": self.desc,
            **id_params,
        }
        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "condition": self.condition,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            "retain_inputs": self.retain_inputs,
            "desc": self.desc,
            **op_params,
        }
        return op_params

    def get_model_name(self):
        return self.model.value

    @abstractmethod
    def naive_cost_estimates(self, left_source_op_cost_estimates: OperatorCostEstimates, right_source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        pass

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

        return join_dr, record_op_stats


class NestedLoopsJoin(JoinOp):

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

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord]) -> tuple[DataRecordSet, int]:
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

        # return empty DataRecordSet if no output records were produced
        if len(output_records) == 0:
            return DataRecordSet([], []), num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed


class EmbeddingJoin(JoinOp):
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
        total_input_tokens = 0
        embeddings = None
        if self.text_only:
            client = OpenAI()
            inputs = [dr.to_json_str(bytes_to_str=True, project_cols=input_fields, sorted=True) for dr in candidates]
            response = client.embeddings.create(input=inputs, model=self.embedding_model.value)
            total_input_tokens = response.usage.total_tokens
            embeddings = np.array([item.embedding for item in response.data])
        else:
            model = SentenceTransformer(self.embedding_model.value)
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
        total_input_cost = model_card["usd_per_input_token"] * total_input_tokens
        embedding_gen_stats = GenerationStats(
            model_name=self.embedding_model.value,
            total_input_tokens=total_input_tokens,
            total_output_tokens=0.0,
            total_input_cost=total_input_cost,
            total_output_cost=0.0,
            cost_per_record=total_input_cost,
            llm_call_duration_secs=time.time() - start_time,
            total_llm_calls=1,
            total_embedding_llm_calls=len(candidates),
        )

        return embeddings, embedding_gen_stats

    def _process_join_candidate_pair(self, left_candidate, right_candidate, gen_kwargs, embedding_sim):
        output_record, output_record_op_stats = super()._process_join_candidate_pair(left_candidate, right_candidate, gen_kwargs)
        return output_record, output_record_op_stats, embedding_sim

    def _process_join_candidate_with_sim(self, left_candidate: DataRecord, right_candidate: DataRecord, passed_operator: bool) -> tuple[DataRecord, RecordOpStats]:
        # compute output record and add to output_records
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator

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

        return join_dr, record_op_stats

    def __call__(self, left_candidates: list[DataRecord], right_candidates: list[DataRecord]) -> tuple[DataRecordSet, int]:
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
                for future in as_completed(futures):
                    self.join_idx += 1
                    join_output_record, join_output_record_op_stats, embedding_sim = future.result()
                    output_records.append(join_output_record)
                    output_record_op_stats.append(join_output_record_op_stats)
                    print(f"{self.join_idx} JOINED")

                    # update similarity thresholds
                    records_joined = join_output_record._passed_operator
                    if not records_joined and embedding_sim > self.max_non_matching_sim:
                        self.max_non_matching_sim = embedding_sim
                    if records_joined and embedding_sim < self.min_matching_sim:
                        self.min_matching_sim = embedding_sim
            
            # update samples drawn and num_inputs_processed
            self.samples_drawn += samples_to_draw
            num_inputs_processed += samples_to_draw

        # process remaining candidates based on embedding similarity
        if len(join_candidates) > 0:
             assert self.samples_drawn == self.num_samples, "All samples should have been drawn before processing remaining candidates"
             with ThreadPoolExecutor(max_workers=self.join_parallelism) as executor:
                futures = []
                for left_candidate, right_candidate, embedding_sim in join_candidates:
                    llm_call_needed = self.min_matching_sim <= embedding_sim <= self.max_non_matching_sim

                    if llm_call_needed:
                        futures.append(executor.submit(self._process_join_candidate_pair, left_candidate, right_candidate, gen_kwargs, embedding_sim))

                    elif embedding_sim < self.min_matching_sim:
                        self.join_idx += 1
                        output_record, record_op_stats = self._process_join_candidate_with_sim(left_candidate, right_candidate, passed_operator=False)
                        output_records.append(output_record)
                        output_record_op_stats.append(record_op_stats)
                        print(f"{self.join_idx} SKIPPED (low sim: {embedding_sim:.4f} < {self.min_matching_sim:.4f})")

                    elif embedding_sim > self.max_non_matching_sim:
                        self.join_idx += 1
                        output_record, record_op_stats = self._process_join_candidate_with_sim(left_candidate, right_candidate, passed_operator=True)
                        output_records.append(output_record)
                        output_record_op_stats.append(record_op_stats)
                        print(f"{self.join_idx} JOINED (high sim: {embedding_sim:.4f} > {self.max_non_matching_sim:.4f})")

                    num_inputs_processed += 1

                # collect results as they complete
                for future in as_completed(futures):
                    self.join_idx += 1
                    join_output_record, join_output_record_op_stats, embedding_sim = future.result()
                    output_records.append(join_output_record)
                    output_record_op_stats.append(join_output_record_op_stats)
                    print(f"{self.join_idx} JOINED")

                    # update similarity thresholds
                    records_joined = join_output_record._passed_operator
                    if not records_joined and embedding_sim > self.max_non_matching_sim:
                        self.max_non_matching_sim = embedding_sim
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

        # return empty DataRecordSet if no output records were produced
        if len(output_records) == 0:
            self.residual_embedding_cost = total_embedding_cost
            return DataRecordSet([], []), num_inputs_processed

        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed

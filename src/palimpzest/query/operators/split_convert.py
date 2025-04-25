from __future__ import annotations

import math

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import Field, StringField
from palimpzest.query.generators.generators import generator_factory
from palimpzest.query.operators.convert import LLMConvert


class SplitConvert(LLMConvert):
    def __init__(self, num_chunks: int = 2, min_size_to_chunk: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_chunks = num_chunks
        self.min_size_to_chunk = min_size_to_chunk
        self.split_generator = generator_factory(self.model, PromptStrategy.SPLIT_PROPOSER, self.cardinality, self.verbose)
        self.split_merge_generator = generator_factory(self.model, PromptStrategy.SPLIT_MERGER, self.cardinality, self.verbose)

        # crude adjustment factor for naive estimation in no-sentinel setting
        self.naive_quality_adjustment = 0.6

    def __str__(self):
        op = super().__str__()
        op += f"    Chunk Size: {str(self.num_chunks)}\n"
        op += f"    Min Size to Chunk: {str(self.min_size_to_chunk)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"num_chunks": self.num_chunks, "min_size_to_chunk": self.min_size_to_chunk, **id_params}

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"num_chunks": self.num_chunks, "min_size_to_chunk": self.min_size_to_chunk, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Update the cost per record and quality estimates produced by LLMConvert's naive estimates.
        We adjust the cost per record to account for the reduced number of input tokens following
        the retrieval of relevant chunks, and we make a crude estimate of the quality degradation
        that results from using a downsized input (although this may in fact improve quality in
        some cases).
        """
        # get naive cost estimates from LLMConvert
        naive_op_cost_estimates = super().naive_cost_estimates(source_op_cost_estimates)

        # re-compute cost per record assuming we use fewer input tokens; naively assume a single input field
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # set refined estimate of cost per record
        naive_op_cost_estimates.cost_per_record = model_conversion_usd_per_record
        naive_op_cost_estimates.cost_per_record_lower_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.cost_per_record_upper_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * self.naive_quality_adjustment
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def is_image_conversion(self) -> bool:
        """SplitConvert is currently disallowed on image conversions, so this must be False."""
        return False

    def get_text_chunks(self, text: str, num_chunks: int) -> list[str]:
        """
        Given a text string, chunk it into num_chunks substrings of roughly equal size.
        """
        chunks = []

        idx, chunk_size = 0, math.ceil(len(text) / num_chunks)
        while idx + chunk_size < len(text):
            chunks.append(text[idx : idx + chunk_size])
            idx += chunk_size

        if idx < len(text):
            chunks.append(text[idx:])

        return chunks

    def get_chunked_candidate(self, candidate: DataRecord, input_fields: list[str]) -> list[DataRecord]:
        """
        For each text field, chunk the content. If a field is smaller than the chunk size,
        simply include the full field.
        """
        # compute mapping from each field to its chunked content
        field_name_to_chunked_content = {}
        for field_name in input_fields:
            field = candidate.get_field_type(field_name)
            content = candidate[field_name]

            # do not chunk this field if it is not a string or a list of strings
            is_string_field = isinstance(field, StringField)
            is_list_string_field = hasattr(field, "element_type") and isinstance(field.element_type, StringField)
            if not (is_string_field or is_list_string_field):
                field_name_to_chunked_content[field_name] = [content]
                continue

            # if this is a list of strings, join the strings
            if is_list_string_field:
                content = "[" + ", ".join(content) + "]"

            # skip this field if its length is less than the min size to chunk
            if len(content) < self.min_size_to_chunk:
                field_name_to_chunked_content[field_name] = [content]
                continue

            # chunk the content
            field_name_to_chunked_content[field_name] = self.get_text_chunks(content, self.num_chunks)

        # compute the true number of chunks (may be 1 if all fields are not chunked)
        num_chunks = max(len(chunks) for chunks in field_name_to_chunked_content.values())

        # create the chunked canidates
        candidates = []
        for chunk_idx in range(num_chunks):
            candidate_copy = candidate.copy()
            for field_name in input_fields:
                field_chunks = field_name_to_chunked_content[field_name]
                candidate_copy[field_name] = field_chunks[chunk_idx] if len(field_chunks) > 1 else field_chunks[0]

            candidates.append(candidate_copy)

        return candidates

    def convert(self, candidate: DataRecord, fields: dict[str, Field]) -> tuple[dict[str, list], GenerationStats]:
        # get the set of input fields to use for the convert operation
        input_fields = self.get_input_fields()

        # lookup most relevant chunks for each field using embedding search
        candidate_copy = candidate.copy()
        chunked_candidates = self.get_chunked_candidate(candidate_copy, input_fields)

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

        # generate outputs for each chunk separately
        chunk_outputs, chunk_generation_stats_lst = [], []
        for candidate in chunked_candidates:
            _, reasoning, chunk_generation_stats, _ = self.split_generator(candidate, fields, json_output=False, **gen_kwargs)
            chunk_outputs.append(reasoning)
            chunk_generation_stats_lst.append(chunk_generation_stats)

        # call the merger
        gen_kwargs = {
            "project_cols": input_fields,
            "output_schema": self.output_schema,
            "chunk_outputs": chunk_outputs,
        }
        field_answers, _, merger_gen_stats, _ = self.split_merge_generator(candidate, fields, **gen_kwargs)

        # compute the total generation stats
        generation_stats = sum(chunk_generation_stats_lst) + merger_gen_stats

        return field_answers, generation_stats

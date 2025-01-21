from __future__ import annotations

from typing import Any

from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
)
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import ListField, StringField
from palimpzest.query.operators.convert import FieldName, LLMConvert


class RAGConvert(LLMConvert):
    def __init__(self, num_chunks_per_field: int, chunk_size: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: in the future, we should abstract the embedding model to allow for different models
        self.client = OpenAI()
        self.embedding_model = "text-embedding-3-small"
        self.num_chunks_per_field = num_chunks_per_field
        self.chunk_size = chunk_size

        # crude adjustment factor for naive estimation in no-sentinel setting
        self.naive_quality_adjustment = 0.6

    def __str__(self):
        op = super().__str__()
        op += f"    Number of Chunks: {str(self.num_chunks_per_field)}\n"
        op += f"    Chunk Size: {str(self.chunk_size)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"num_chunks_per_field": self.num_chunks_per_field, "chunk_size": self.chunk_size, **id_params}

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"num_chunks_per_field": self.num_chunks_per_field, "chunk_size": self.chunk_size, **op_params}

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
        est_num_input_tokens = self.num_chunks_per_field * self.chunk_size
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # set refined estimate of cost per record and, for now,
        # assume quality multiplier is proportional to sqrt(sqrt(token_budget))
        naive_op_cost_estimates.cost_per_record = model_conversion_usd_per_record
        naive_op_cost_estimates.cost_per_record_lower_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.cost_per_record_upper_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * self.naive_quality_adjustment
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def is_image_conversion(self) -> bool:
        """RAGConvert is currently disallowed on image conversions, so this must be False."""
        return False

    def chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Given a text string, chunk it into substrings of length chunk_size.
        """
        chunks = []
        idx = 0
        while idx + chunk_size < len(text):
            chunks.append(text[idx : idx + chunk_size])
            idx += chunk_size
        
        if idx < len(text):
            chunks.append(text[idx:])

        return chunks

    def compute_embedding(self, text: str) -> list[float]:
        """
        Compute the embedding for a text string.
        """
        response = self.client.embeddings.create(input=text, model=self.embedding_model)

        return response.data[0].embedding

    def compute_similarity(self, query_embedding: list[float], chunk_embedding: list[float]) -> float:
        """
        Compute the similarity between the query and chunk embeddings.
        """
        return dot(query_embedding, chunk_embedding) / (norm(query_embedding) * norm(chunk_embedding))

    def get_chunked_candidate(self, candidate: DataRecord, input_fields: list[str], output_fields: list[str]) -> DataRecord:
        """
        For each text field, chunk the content and compute the chunk embeddings. Then select the top-k chunks
        for each field. If a field is smaller than the chunk size, simply include the full field.
        """
        # compute embedding for output fields
        output_fields_desc = ""
        field_desc_map = self.output_schema.field_desc_map()
        for field_name in output_fields:
            output_fields_desc += f"- {field_name}: {field_desc_map[field_name]}\n"
        query_embedding = self.compute_embedding(output_fields_desc)

        for field_name in input_fields:
            field = candidate.get_field_type(field_name)

            # skip this field if it is not a string or a list of strings
            is_string_field = isinstance(field, StringField)
            is_list_string_field = isinstance(field, ListField) and isinstance(field.element_type, StringField)
            if not (is_string_field or is_list_string_field):
                continue

            # if this is a list of strings, join the strings
            if is_list_string_field:
                candidate[field_name] = "[" + ", ".join(candidate[field_name]) + "]"

            # skip this field if it is a string field and its length is less than the chunk size
            if isinstance(field, str) and len(candidate[field_name]) < self.chunk_size:
                continue

            # chunk the content
            chunks = self.chunk_text(candidate[field_name], self.chunk_size)

            # compute embeddings for each chunk
            chunk_embeddings = [self.compute_embedding(chunk) for chunk in chunks]

            # select the top-k chunks
            sorted_chunks = sorted(
                zip(range(len(chunks)), chunks, chunk_embeddings),
                key=lambda tup: self.compute_similarity(query_embedding, tup[2]),
                reverse=True,
            )
            top_k_chunks = [(chunk_idx, chunk) for chunk_idx, chunk, _ in sorted_chunks[:self.num_chunks_per_field]]

            # sort the top-k chunks by their original index in the content, and join them with ellipses
            top_k_chunks = [chunk for _, chunk in sorted(top_k_chunks, key=lambda tup: tup[0])]
            candidate[field_name] = "...".join(top_k_chunks)

        return candidate

    def convert(self, candidate: DataRecord, fields: list[str]) -> tuple[dict[FieldName, list[Any]], GenerationStats]:
        # get the set of input fields to use for the convert operation
        input_fields = self.get_input_fields()

        # lookup most relevant chunks for each field using embedding search
        candidate_copy = candidate.copy()
        candidate_copy = self.get_chunked_candidate(candidate_copy, input_fields)

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

        # generate outputs for all fields in a single query
        field_answers, _, generation_stats = self.generator(candidate_copy, fields, **gen_kwargs)

        # if there was an error for any field, execute a conventional query on that field
        for field, answers in field_answers.items():
            if answers is None:
                single_field_answers, _, single_field_stats = self.generator(candidate_copy, [field], **gen_kwargs)
                field_answers.update(single_field_answers)
                generation_stats += single_field_stats

        return field_answers, generation_stats

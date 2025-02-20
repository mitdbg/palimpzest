from __future__ import annotations

import time

from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    Model,
)
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import Field, StringField
from palimpzest.query.operators.convert import LLMConvert


class RAGConvert(LLMConvert):
    def __init__(self, num_chunks_per_field: int, chunk_size: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: in the future, we should abstract the embedding model to allow for different models
        self.client = None
        self.embedding_model = Model.TEXT_EMBEDDING_3_SMALL
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

    def compute_embedding(self, text: str) -> tuple[list[float], GenerationStats]:
        """
        Compute the embedding for a text string. Return the embedding and the GenerationStats object
        that captures the cost of the operation.
        """
        # get the embedding model name
        model_name = self.embedding_model.value

        # compute the embedding
        start_time = time.time()
        response = self.client.embeddings.create(input=text, model=model_name)
        total_time = time.time() - start_time

        # extract the embedding
        embedding = response.data[0].embedding

        # compute the generation stats object
        model_card = MODEL_CARDS[model_name]
        total_input_tokens = response.usage.total_tokens
        total_input_cost = model_card["usd_per_input_token"] * total_input_tokens
        embed_stats = GenerationStats(
            model_name=model_name,  # NOTE: this should be overwritten by generation model in convert()
            total_input_tokens=total_input_tokens,
            total_output_tokens=0.0,
            total_input_cost=total_input_cost,
            total_output_cost=0.0,
            cost_per_record=total_input_cost,
            llm_call_duration_secs=total_time,
        )

        return embedding, embed_stats

    def compute_similarity(self, query_embedding: list[float], chunk_embedding: list[float]) -> float:
        """
        Compute the similarity between the query and chunk embeddings.
        """
        return dot(query_embedding, chunk_embedding) / (norm(query_embedding) * norm(chunk_embedding))

    def get_chunked_candidate(self, candidate: DataRecord, input_fields: list[str], output_fields: list[str]) -> tuple[DataRecord, GenerationStats]:
        """
        For each text field, chunk the content and compute the chunk embeddings. Then select the top-k chunks
        for each field. If a field is smaller than the chunk size, simply include the full field.
        """
        # initialize stats for embedding costs
        embed_stats = GenerationStats()

        # compute embedding for output fields
        output_fields_desc = ""
        field_desc_map = self.output_schema.field_desc_map()
        for field_name in output_fields:
            output_fields_desc += f"- {field_name}: {field_desc_map[field_name]}\n"
        query_embedding, query_embed_stats = self.compute_embedding(output_fields_desc)

        # add cost of embedding the query to embed_stats
        embed_stats += query_embed_stats

        # for each input field, chunk its content and compute the (per-chunk) embeddings
        for field_name in input_fields:
            field = candidate.get_field_type(field_name)

            # skip this field if it is not a string or a list of strings
            is_string_field = isinstance(field, StringField)
            is_list_string_field = hasattr(field, "element_type") and isinstance(field.element_type, StringField)
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
            chunk_embeddings, chunk_embed_stats_lst = zip(*[self.compute_embedding(chunk) for chunk in chunks])

            # add cost of embedding each chunk to embed_stats
            for chunk_embed_stats in chunk_embed_stats_lst:
                embed_stats += chunk_embed_stats

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

        return candidate, embed_stats

    def convert(self, candidate: DataRecord, fields: dict[str, Field]) -> tuple[dict[str, list], GenerationStats]:
        # set client
        self.client = OpenAI() if self.client is None else self.client

        # get the set of input fields to use for the convert operation
        input_fields = self.get_input_fields()
        output_fields = list(fields.keys())

        # lookup most relevant chunks for each field using embedding search
        candidate_copy = candidate.copy()
        candidate_copy, embed_stats = self.get_chunked_candidate(candidate_copy, input_fields, output_fields)

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

        # generate outputs for all fields in a single query
        field_answers, _, generation_stats, _ = self.generator(candidate_copy, fields, **gen_kwargs)

        # NOTE: summing embedding stats with generation stats is messy because it will lead to misleading
        #       measurements of total_input_tokens and total_output_tokens. We should fix this in the future.
        #       The good news: as long as we compute the cost_per_record of each GenerationStats object correctly,
        #       then the total cost of the operation will be correct (which will roll-up to correctly computing
        #       the total cost of the operator, plan, and execution).
        #
        # combine stats from embedding with stats for generation
        generation_stats += embed_stats

        # if there was an error for any field, execute a conventional query on that field
        for field_name, answers in field_answers.items():
            if answers is None:
                single_field_answers, _, single_field_stats, _ = self.generator(candidate_copy, {field_name: fields[field_name]}, **gen_kwargs)
                field_answers.update(single_field_answers)
                generation_stats += single_field_stats

        return field_answers, generation_stats

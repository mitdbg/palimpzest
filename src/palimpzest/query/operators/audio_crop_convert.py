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


class AudioCropConvert(LLMConvert):
    def __init__(self, crop_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: in the future, we should abstract the embedding model to allow for different models
        self.client = None
        self.crop_length=crop_length

        # crude adjustment factor for naive estimation in no-sentinel setting
        self.naive_quality_adjustment = 0.6

    def __str__(self):
        op = super().__str__()
        op += f"    Crop Length: {str(self.crop_length)}\n"
       
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"crop_length": self.crop_length, **id_params}

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"crop_length": self.crop_length, **op_params}

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

      
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * self.naive_quality_adjustment
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def is_image_conversion(self) -> bool:
        """audio_crop_convert is currently disallowed on image conversions, so this must be False."""
        return False

    

    def convert(self, candidate: DataRecord, fields: dict[str, Field]) -> tuple[dict[str, list], GenerationStats]:
        #COMMENT OUT 
        # set client
        #self.client = OpenAI() if self.client is None else self.client

        # get the set of input fields to use for the convert operation
        input_fields = self.get_input_fields()
        output_fields = list(fields.keys())

        #COMMENT OUT
        # lookup most relevant chunks for each field using embedding search
        #candidate_copy = candidate.copy()
        #candidate_copy, embed_stats = self.get_chunked_candidate(candidate_copy, input_fields, output_fields)

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

        # generate outputs for all fields in a single query
        field_answers, _, generation_stats, _ = self.generator(candidate, fields, crop_length=self.crop_length, **gen_kwargs)

        # NOTE: summing embedding stats with generation stats is messy because it will lead to misleading
        #       measurements of total_input_tokens and total_output_tokens. We should fix this in the future.
        #       The good news: as long as we compute the cost_per_record of each GenerationStats object correctly,
        #       then the total cost of the operation will be correct (which will roll-up to correctly computing
        #       the total cost of the operator, plan, and execution).
        #
        # combine stats from embedding with stats for generation
        

        # if there was an error for any field, execute a conventional query on that field
        for field_name, answers in field_answers.items():
            if answers is None:
                single_field_answers, _, single_field_stats, _ = self.generator(candidate, {field_name: fields[field_name]}, crop_length=self.crop_length, **gen_kwargs)
                field_answers.update(single_field_answers)
                generation_stats += single_field_stats

        return field_answers, generation_stats

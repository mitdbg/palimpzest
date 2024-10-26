from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    PromptStrategy,
)
from palimpzest.dataclasses import OperatorCostEstimates
from palimpzest.generators.generators import DSPyGenerator
from palimpzest.operators import LLMConvert, LLMConvertBonded, LLMConvertConventional
from palimpzest.utils.token_reduction_helpers import best_substring_match, find_best_range


class TokenReducedConvert(LLMConvert):
    # NOTE: moving these closer to the TokenReducedConvert class for now (in part to make
    #       them easier to mock); we can make these parameterized as well
    MAX_HEATMAP_UPDATES: int = 5
    TOKEN_REDUCTION_SAMPLE: int = 0
    TOKEN_REDUCTION_GRANULARITY: float = 0.001

    def __init__(self, token_budget: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_budget = token_budget
        self.heatmap_dict = {}
        self.resolution = self.TOKEN_REDUCTION_GRANULARITY
        self.first_execution = True

    def __str__(self):
        op = super().__str__()
        op += f"    Token Budget: {str(self.token_budget)}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"token_budget": self.token_budget, **copy_kwargs}

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"token_budget": self.token_budget, **op_params}

        return op_params

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.token_budget == other.token_budget
            and self.model == other.model
            and self.cardinality == other.cardinality
            and self.prompt_strategy == other.prompt_strategy
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Update the cost per record and quality estimates produced by LLMConvert's naive estimates.
        We adjust the cost per record to account for the reduced number of input tokens following
        token reduction, and we make a crude estimate of the quality degradation that results from
        using fewer tokens.
        """
        # get naive cost estimates from LLMConvert
        naive_op_cost_estimates = super().naiveCostEstimates(source_op_cost_estimates)

        # re-compute cost per record assuming we use fewer input tokens
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS * self.token_budget
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # set refined estimate of cost per record and, for now,
        # assume quality multiplier is proportional to sqrt(sqrt(token_budget))
        naive_op_cost_estimates.cost_per_record = model_conversion_usd_per_record
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * math.sqrt(math.sqrt(self.token_budget))

        return naive_op_cost_estimates

    def reduce_context(self, heatmap: List[int], full_context: str) -> str:
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            si, ei = find_best_range(
                heatmap,
                int(self.token_budget / self.TOKEN_REDUCTION_GRANULARITY),
                trim_zeros=False,
            )
            print("si:", si, "ei:", ei)
            sr, er = (
                si * self.TOKEN_REDUCTION_GRANULARITY,
                ei * self.TOKEN_REDUCTION_GRANULARITY,
            )
            test_len = len(full_context)
            start = int(sr * test_len)
            end = int(er * test_len)
            if self.verbose:
                print(f"start ratio: {sr} -- end ratio: {er}")
                print("character start:", start, "end:", end)
            sample = full_context[start:end]
            return sample

        else:
            raise NotImplementedError("Token reduction is only supported for DSPY_COT_QA prompts")

    def _dspy_generate_fields(
        self, prompt: str, content: str | List[bytes] | None = None, verbose: bool = False
    ) -> Tuple[List[Dict[str, List]] | Any]:
        full_context = content
        if self.first_execution or self.heatmap_dict["count"] < self.MAX_HEATMAP_UPDATES:
            print("Warming up heatmap")
            answer, query_stats = super()._dspy_generate_fields(prompt, full_context, verbose)
            self.first_execution = False
            # create the heatmap structure with default resolution of 0.001 and count of 0
            self.heatmap_dict = {
                "count": 0,
                "heatmap": [0] * int(1.0 / self.resolution),
            }
        else:
            doc_schema = str(self.outputSchema)
            doc_type = self.outputSchema.className()

            if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                generator = DSPyGenerator(self.model.value, self.prompt_strategy, doc_schema, doc_type, verbose)
            else:
                raise Exception(f"Token reduction not implemented for {self.prompt_strategy}")

            heatmap = self.heatmap_dict["heatmap"]
            count = self.heatmap_dict["count"]
            # only refer to the heatmap if the count is greater than a enough sample size
            # TODO: only trim the context if the attention is clustered in a small region
            if count >= self.TOKEN_REDUCTION_SAMPLE:
                context = self.reduce_context(heatmap, full_context)
                try:
                    answer, query_stats = generator.generate(context=context, question=prompt)
                except Exception as e:
                    print(f"DSPy generation error: {e}, falling back to unreduced generation")
                    answer, query_stats = super()._dspy_generate_fields(prompt, content, verbose)

        try:
            gsi, gei = best_substring_match(answer, full_context)
        except Exception as e:
            print("Error in substring match:", e)
            gsi, gei = 0, len(full_context)
        context_len = len(full_context)
        gsr, ger = gsi / context_len, gei / context_len
        norm_si, norm_ei = int(gsr / self.resolution), int(ger / self.resolution)
        if verbose:
            print(f"best_start: {gsi} -- best_end: {gei}")

        self.heatmap_dict["count"] += 1
        self.heatmap_dict["heatmap"][norm_si:norm_ei] = map(
            lambda x: x + 1, self.heatmap_dict["heatmap"][norm_si:norm_ei]
        )

        return answer, query_stats


class TokenReducedConvertConventional(TokenReducedConvert, LLMConvertConventional):
    pass


class TokenReducedConvertBonded(TokenReducedConvert, LLMConvertBonded):
    pass

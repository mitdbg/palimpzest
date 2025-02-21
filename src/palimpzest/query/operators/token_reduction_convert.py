from __future__ import annotations

import math
from typing import Any

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
)
from palimpzest.core.data.dataclasses import OperatorCostEstimates
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.utils.token_reduction_helpers import best_substring_match, find_best_range


# NOTE: this convert operation will not work with the new generation abstraction, and it needs to be worked on.
#       There are two minor issues with the operator as it exists:
#
#       1) The token reduction operation operated over the entire JSON string of the input DataRecord
#          - while this works in practice, it makes it difficult to use this operator with a generation framework
#            where each field may be placed in a specific place in the format string for a prompt
#          - we need to either (A) rewrite the reduction to take place on a field-by-field basis (or at least
#            make it possible to recover each field after a global reduction) or (B) add custom logic within
#            the Generator class(es) to handle this operator [I much prefer (A) over (B)]
#
#       2) The heatmap update logic does not translate well to the distributed setting, where this operator may
#          be copied and executed many times in parallel
#          - each copy of the operator will have its own heatmap and require MAX_HEATMAP_UPDATES just to enter the
#            phase where token reduction takes place
#          - this means that if we have 20-way parallelism and a MAX_HEATMAP_UPDATES = 5, it can take 100 inputs
#            before token reduction ever takes place
#          - this also creates difficulties in properly performing cost-estimation for this operator; e.g. if we use
#            n <= MAX_HEATMAP_UPDATES samples to cost this operator, then we will never actually measure its performance
#            in the token reduction phase -- which could have a serious degradation in quality that our optimizer doesn't see
class TokenReducedConvertBonded(LLMConvertBonded):
    # NOTE: moving these closer to the TokenReducedConvertBonded class for now (in part to make
    #       them easier to mock); we can make these parameterized as well
    MAX_HEATMAP_UPDATES: int = 5
    TOKEN_REDUCTION_SAMPLE: int = 0
    TOKEN_REDUCTION_GRANULARITY: float = 0.001

    def __init__(self, token_budget: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_budget = token_budget
        self.resolution = self.TOKEN_REDUCTION_GRANULARITY
        self.first_execution = True
        self.count = 0
        self.heatmap = [0] * int(1.0 / self.resolution)

    def __str__(self):
        op = super().__str__()
        op += f"    Token Budget: {str(self.token_budget)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"token_budget": self.token_budget, **id_params}

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"token_budget": self.token_budget, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Update the cost per record and quality estimates produced by LLMConvert's naive estimates.
        We adjust the cost per record to account for the reduced number of input tokens following
        token reduction, and we make a crude estimate of the quality degradation that results from
        using fewer tokens.
        """
        # get naive cost estimates from LLMConvert
        naive_op_cost_estimates = super().naive_cost_estimates(source_op_cost_estimates)

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
        naive_op_cost_estimates.cost_per_record_lower_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.cost_per_record_upper_bound = naive_op_cost_estimates.cost_per_record
        naive_op_cost_estimates.quality = (naive_op_cost_estimates.quality) * math.sqrt(math.sqrt(self.token_budget))
        naive_op_cost_estimates.quality_lower_bound = naive_op_cost_estimates.quality
        naive_op_cost_estimates.quality_upper_bound = naive_op_cost_estimates.quality

        return naive_op_cost_estimates

    def is_image_conversion(self) -> bool:
        """TokenReducedConvertBonded is currently disallowed on image conversions, so this must be False."""
        return False

    def reduce_context(self, full_context: str) -> str:
        range = find_best_range(
            self.heatmap,
            int(self.token_budget / self.TOKEN_REDUCTION_GRANULARITY),
            trim_zeros=False,
        )
        if not range:
            raise Exception("No range found in heatmap")
        si, ei = range
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

    def _dspy_generate_fields(self, prompt: str, content: str | list[str]) -> tuple[list[dict[str, list]] | Any]:
        raise Exception(
            "TokenReducedConvertBonded is executing despite being deprecated until implementation changes can be made."
        )
        answer, query_stats = None, None
        if self.first_execution or self.count < self.MAX_HEATMAP_UPDATES:
            if self.verbose:
                print("Warming up heatmap")
            answer, query_stats = super()._dspy_generate_fields(prompt, content)
            self.first_execution = False

        else:
            if self.verbose:
                print("Using heatmap")

            # only refer to the heatmap if the count is greater than a enough sample size
            # TODO: only trim the context if the attention is clustered in a small region
            if self.count >= self.TOKEN_REDUCTION_SAMPLE:
                context = self.reduce_context(content)
                try:
                    answer, _, query_stats = self.generator.generate(context=context, prompt=prompt)
                except Exception as e:
                    print(f"DSPy generation error: {e}, falling back to unreduced generation")
                    answer, query_stats = super()._dspy_generate_fields(prompt, content)

        # TODO: answer and query stats may be unbound if we hit the else block
        # and count < TOKEN_REDUCTION_SAMPLE, which makes the below pretty clunky
        # this throw asserts our view of the world and we should refactor this
        if answer is None or query_stats is None:
            raise Exception("answer or query_stats is None")
        try:
            match = best_substring_match(answer, content)
            if not match:
                gsi, gei = 0, len(content)
            else:
                gsi, gei = match
        except Exception as e:
            print("Error in substring match:", e)
            gsi, gei = 0, len(content)
        context_len = len(content)
        gsr, ger = gsi / context_len, gei / context_len
        norm_si, norm_ei = int(gsr / self.resolution), int(ger / self.resolution)
        if self.verbose:
            print(f"best_start: {gsi} -- best_end: {gei}")

        self.count += 1
        self.heatmap[norm_si:norm_ei] = map(lambda x: x + 1, self.heatmap[norm_si:norm_ei])

        return answer, query_stats

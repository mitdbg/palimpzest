from __future__ import annotations
import json 
import time
from typing import List
from palimpzest.constants import Model
from palimpzest.generators.dspy_utils import gen_filter_signature_class, gen_qa_signature_class
from .strategy import PhysicalOpStrategy
from palimpzest.profiler.attentive_trim import find_best_range, trim_context, update_heatmap_json, best_substring_match

from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, convert

class TokenReducedConvert(convert.LLMConvert):
    token_budget: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_dict = None

    def reduce_context(self, question:str, full_context: str) -> str:
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            heatmap = self.heatmap_dict["heatmap"]
            count = self.heatmap_dict["count"]
            # if self.verbose: # TODO not sure this exists
            print(f"count: {count}")

            # only refer to the heatmap if the count is greater than a enough sample size
            # TODO: only trim the context if the attention is clustered in a small region
            if count >= TOKEN_REDUCTION_SAMPLE:
                si, ei = find_best_range(
                    heatmap,
                    int(self.token_budget / TOKEN_REDUCTION_GRANULARITY),
                    trim_zeros=False,
                )
                sr, er = (
                    si * TOKEN_REDUCTION_GRANULARITY,
                    ei * TOKEN_REDUCTION_GRANULARITY,
                )
                self._print_verbose(f"start ratio: {sr} -- end ratio: {er}")
                return trim_context(full_context, sr, er)
    
        else:
            raise NotImplementedError("Token reduction is only supported for DSPY_COT_QA prompts")


class TokenReducedConventionalConvert(TokenReducedConvert):


    def convert(self, candidate_content, fields) -> None:

        doc_schema = str(self.outputSchema)
        doc_type = self.outputSchema.className()

        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            promptSignature = gen_filter_signature_class(doc_schema, doc_type)
        elif self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            promptSignature = gen_qa_signature_class(doc_schema, doc_type)

        reduction = False
        start_time = time.time()
        field_outputs, query_stats = {}, {}
        for field_name in fields:
            full_prompt = self._construct_query_prompt(fields_to_generate=[field_name])
            # The first time we see a prompt, we need to generate the heatmap
            if self.heatmap_dict is None:
                # create the heatmap structure with default resolution of 0.001 and count of 0
                buckets = int(1.0 / TOKEN_REDUCTION_GRANULARITY)
                hist = [0] * buckets
                heatmap_dict = {
                    "prompt_schema": f"{promptSignature}",
                    "question": field_name,
                    "resolution": TOKEN_REDUCTION_GRANULARITY,
                    "count": 0,
                    "heatmap": hist,
                }
                self.heatmap_dict = heatmap_dict
                prompt = full_prompt
            else:
                reduction = True
                prompt = self.reduce_context(question=field_name, full_context=candidate_content)
    

            json_objects, field_stats = self._dspy_generate_fields(fields_to_generate=[field_name], content=candidate_content, prompt=prompt)
            if reduction and field_stats['answer'] is None:
                json_objects, new_stats = self._dspy_generate_fields(fields_to_generate=[field_name], content=candidate_content, prompt=full_prompt)

                field_stats["llm_call_duration_secs"] += new_stats["llm_call_duration_secs"]
                for k, _ in new_stats['usage'].items():
                    field_stats['usage'][k] += new_stats['usage'][k]
                field_stats['cost_per_record'] += new_stats['cost_per_record']
                field_stats['finish_reason'] = new_stats['finish_reason']
                field_stats['answer_log_probs'] = new_stats['answer_log_probs']
                field_stats['answer'] = new_stats['answer']

            if self.prompt_strategy == PromptStrategy.DSPY_COT_QA and self.heatmap_dict["count"] < MAX_HEATMAP_UPDATES:
                if self.verbose: # TODO not sure this exists
                    print("Reduction enabled")
                    print(f"answer: {field_stats['answer']}")
                try:
                    gsi, gei = best_substring_match(field_stats['answer'], full_prompt)
                except Exception as e:
                    print("Error in substring match:", e)
                    gsi, gei = 0, len(full_prompt)

                context_len = len(full_prompt)
                gsr, ger = gsi / context_len, gei / context_len
                norm_si, norm_ei = int(gsr / TOKEN_REDUCTION_GRANULARITY), int(
                    ger / TOKEN_REDUCTION_GRANULARITY
                )
                if self.verbose:
                    print(f"best_start: {gsi} -- best_end: {gei}")

                self.heatmap_dict = update_heatmap_json(self.heatmap_dict, norm_si, norm_ei)

            for key, value in field_stats.items():
                # TODO maybe a better way to find which stats to aggregate?
                if type(value) == type(dict()):
                    for k2, v2 in value.items():
                        # Should we simply throw the usage away here?
                        query_stats[k2] = query_stats.get(k2,type(v2)()) + v2 
                else:
                    query_stats[key] = query_stats.get(key, type(value)()) + value
            field_outputs[field_name] = json_objects

        query_stats["total_time"] = time.time() - start_time
        return field_outputs, query_stats

class TokenReducedBondedQueryConvert(TokenReducedConvert):

    def convert(self, candidate_content, fields) -> None:
        prompt = self._construct_query_prompt(fields_to_generate=fields)
        # generate all fields in a single query
        final_json_objects, query_stats = self._dspy_generate_fields(fields_to_generate=fields, content=candidate_content, prompt=prompt)

        # if there was an error, execute a conventional query
        if all([v is None for v in final_json_objects[0].values()]):
            # generate each field one at a time
            field_outputs = {}
            for field_name in fields:
                prompt = self._construct_query_prompt(fields_to_generate=[field_name])
                json_objects, field_stats = self._dspy_generate_fields(fields_to_generate=[field_name], content = candidate_content, prompt=prompt)

                # update query_stats
                for key, value in field_stats.items():
                    if type(value) == type(dict()):
                        for k, v in value.items():
                            query_stats[key][k] = query_stats[key].get(k,0) + value[k]
                    else:
                        query_stats[key] += value

                # update field_outputs
                field_outputs[field_name] = json_objects


class TokenReductionStrategy(PhysicalOpStrategy):

    query_strategy_map = {
        QueryStrategy.BONDED: TokenReducedBondedQueryConvert,
        QueryStrategy.CONVENTIONAL: TokenReducedConventionalConvert,
        }

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                token_budgets: List[float],
                prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
                *args, **kwargs) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            for token_budget in token_budgets:
                if token_budget >= 1:
                    print("A token reduction strategy must specify a token_budget < 1!")
                    continue
                # TODO this or query strategy as a parameter? 
                for query_strategy in cls.query_strategy_map:
                    op_class = cls.query_strategy_map[query_strategy] 
                    # physical_op_type = type(cls.__name__+model.name,
                    physical_op_type = type(op_class.__name__,
                                            (op_class,),
                                            {'model': model,
                                            'prompt_strategy': prompt_strategy,
                                            'token_budget': token_budget})
                    return_operators.append(physical_op_type)

        return return_operators

class TokenReducedConvertStrategy(TokenReductionStrategy):
    """
    This strategy creates physical operator classes using a bonded query strategy.
    It ties together several records for the same fields, possibly defaulting to a conventional conversion strategy.
    """
    logical_op_class = logical.ConvertScan
    physical_op_class = TokenReducedConvert
# from __future__ import annotations

# from typing import List
# from palimpzest.constants import Model
# from palimpzest.generators.generators import DSPyGenerator
# from palimpzest.strategies import PhysicalOpStrategy
# from palimpzest.utils import getVisionModels

# from palimpzest.constants import *
# from palimpzest.elements import *
# from palimpzest.operators import logical, physical, convert
# from fuzzywuzzy import process, fuzz

# def find_best_range(values, budget, trim_zeros=False):
#     """
#     Finds the consecutive range with the biggest sum within a budget.

#     Args:
#         values: A list of non-negative numbers.
#         budget: The maximum number of consecutive elements to consider.

#     Returns:
#         A tuple containing the start and end indices (inclusive) of the best range,
#         or None if the array is empty.
#     """
#     if not values:
#         return None

#     n = len(values)
#     best_sum, best_start, current_sum, current_start = 0, 0, 0, 0

#     # Iterate through the array, keeping track of current and best ranges.
#     for i in range(n):
#         current_sum += values[i]

#         # If the current range exceeds the budget, remove elements from the beginning.
#         while current_start + budget - 1 < i and current_start + budget - 1 >= 0:
#             current_sum -= values[current_start]
#             current_start += 1

#         # Update best range if the current sum is bigger.
#         if current_sum > best_sum:
#             best_sum = current_sum
#             best_start = current_start

#     best_end = best_start + budget - 1
#     print("best_start:", best_start, "best_end:", best_end)
#     if trim_zeros:
#         # Trim leading/trailing zeros
#         while best_start >= 0 and values[best_start] == 0:
#             best_start += 1

#         while best_end < n and values[best_end] == 0:
#             best_end -= 1
#     else:
#         # balance the zero entries equally on both sides
#         leading_zeros = 0
#         trailing_zeros = 0
#         start_idx = best_start
#         end_idx = best_end
#         while start_idx >= 0 and values[start_idx] == 0:
#             leading_zeros += 1
#             start_idx += 1
#         while end_idx < n and values[end_idx] == 0:
#             trailing_zeros += 1
#             end_idx -= 1
#         half_zeros = int((leading_zeros + trailing_zeros) / 2)
#         print("leading_zeros:", leading_zeros, "trailing_zeros:", trailing_zeros, "half_zeros:", half_zeros)
#         best_start = best_start - half_zeros + leading_zeros
#         best_end = best_end - trailing_zeros + leading_zeros + trailing_zeros - half_zeros

#         if best_start < 0:
#             best_end = best_end - best_start
#             best_start = 0
#         if best_end >= n:
#             best_start = best_start - (best_end - n + 1)
#             best_end = n - 1

#     return best_start, best_end + 1


# def get_range_from_hist(file_path, range_budget, resolution=0.001, trim_zeros=True):
#     # Load data from csv file and extract he second column as values
#     values = []
#     with open(file_path, "r") as file:
#         for line in file:
#             line = line.strip()
#             values.append(int(float(line.split(",")[1])))
#     index_range = 1 / resolution
#     budget = int(range_budget * index_range)
#     # Find the best range
#     start, end = find_best_range(values, budget, trim_zeros=trim_zeros)
#     print("start:", start, "end:", end, "index_range:", index_range)
#     return start * 1.0 / index_range, end * 1.0 / index_range

# def best_substring_match(query, context):
#     # This will extract all substrings of length equal to the query from the string
#     candidates = [context[i:i + len(query)] for i in range(len(context) - len(query) + 1)]
#     print("grd:", query)
#     # Find the best match among the candidates
#     ret = process.extractOne(query, candidates, scorer=fuzz.ratio)
#     if ret is None:
#         return None

#     best_match, score = ret
#     positions = [can == best_match for can in candidates]
#     start = positions.index(True)
#     end = start + len(query)
#     # print("best match:", best_match, "score:", score, "start:", start, "end:", end)
#     # print("-------", string[start:end])
#     return start, end

# class TokenReducedConvert(convert.LLMConvert):
#     token_budget: float
#     # NOTE: moving these closer to the TokenReducedConvert class for now (in part to make
#     #       them easier to mock); we can make these parameterized as well
#     MAX_HEATMAP_UPDATES: int=5
#     TOKEN_REDUCTION_SAMPLE: int=0
#     TOKEN_REDUCTION_GRANULARITY: float=0.001

#     @classmethod
#     def materializes(self, logical_operator) -> bool:
#         if not isinstance(logical_operator, logical.ConvertScan):
#             return False

#         # token reduction is not well-defined for image conversions (yet)
#         if logical_operator.image_conversion or self.model in getVisionModels():
#             return False

#         return True

#     def __init__(self, verbose: bool=False, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.verbose = verbose
#         self.heatmap_dict = {}
#         self.resolution = self.TOKEN_REDUCTION_GRANULARITY
#         self.first_execution = True

#     def reduce_context(self, heatmap:List[int], full_context: str) -> str:
#         if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
#             si, ei = find_best_range(
#                 heatmap,
#                 int(self.token_budget / self.TOKEN_REDUCTION_GRANULARITY),
#                 trim_zeros=False,
#             )
#             print("si:", si, "ei:", ei)
#             sr, er = (
#                 si * self.TOKEN_REDUCTION_GRANULARITY,
#                 ei * self.TOKEN_REDUCTION_GRANULARITY,
#             )
#             test_len = len(full_context)
#             start = int(sr * test_len)
#             end = int(er * test_len)
#             if self.verbose:
#                 print(f"start ratio: {sr} -- end ratio: {er}")
#                 print("character start:", start, "end:", end)
#             sample = full_context[start:end]
#             return sample
    
#         else:
#             raise NotImplementedError("Token reduction is only supported for DSPY_COT_QA prompts")

#     def _dspy_generate_fields(self, prompt: str, content: str | List[bytes] | None = None, verbose: bool = False) -> physical.Tuple[List[logical.Dict[str, List]] | Any]:

#         full_context = content
#         if self.first_execution or self.heatmap_dict["count"] < self.MAX_HEATMAP_UPDATES:
#             print("Warming up heatmap")
#             answer, query_stats = super()._dspy_generate_fields(prompt, full_context, verbose)
#             self.first_execution = False
#             # create the heatmap structure with default resolution of 0.001 and count of 0
#             self.heatmap_dict = {
#                 "count": 0,
#                 "heatmap": [0] * int(1.0 / self.resolution),
#             }
#         else:
#             doc_schema = str(self.outputSchema)
#             doc_type = self.outputSchema.className()

#             if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
#                 generator = DSPyGenerator(
#                     self.model.value, self.prompt_strategy, doc_schema, doc_type, verbose
#                 )
#             else:
#                 raise Exception(f"Token reduction not implemented for {self.prompt_strategy}")

#             heatmap = self.heatmap_dict["heatmap"]
#             count = self.heatmap_dict["count"]
#             # only refer to the heatmap if the count is greater than a enough sample size
#             # TODO: only trim the context if the attention is clustered in a small region
#             if count >= self.TOKEN_REDUCTION_SAMPLE:
#                 context = self.reduce_context(heatmap, full_context)
#                 try:
#                     answer, query_stats = generator.generate(context=context, question=prompt)
#                 except Exception as e:
#                     print(f"DSPy generation error: {e}, falling back to unreduced generation")
#                     answer, query_stats = super()._dspy_generate_fields(prompt, content, verbose)

#         try:
#             gsi, gei = best_substring_match(answer, full_context)
#         except Exception as e:
#             print("Error in substring match:", e)
#             gsi, gei = 0, len(full_context)
#         context_len = len(full_context)
#         gsr, ger = gsi / context_len, gei / context_len
#         norm_si, norm_ei = int(gsr/self.resolution), int(ger/self.resolution)
#         if verbose:
#             print(f"best_start: {gsi} -- best_end: {gei}")

#         self.heatmap_dict["count"] += 1
#         self.heatmap_dict["heatmap"][norm_si:norm_ei] = map(lambda x: x+1, self.heatmap_dict["heatmap"][norm_si:norm_ei])
        
#         return answer, query_stats

# class TokenReducedConventionalConvert(TokenReducedConvert, convert.LLMConvertConventional):
#     pass

# class TokenReducedBondedConvert(TokenReducedConvert, convert.LLMConvertBonded):
#     pass


# class TokenReductionStrategy(PhysicalOpStrategy):

#     query_strategy_map = {
#         QueryStrategy.CONVENTIONAL: TokenReducedConventionalConvert,
#         QueryStrategy.BONDED: TokenReducedBondedConvert,
#         }
#     query_strategy = None

#     @staticmethod
#     def __new__(cls, 
#                 available_models: List[Model],
#                 token_budgets: List[float],
#                 prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
#                 *args, **kwargs) -> List[physical.PhysicalOperator]:

#         return_operators = []
#         for model in available_models:
#             if model.value not in MODEL_CARDS:
#                 raise ValueError(f"Model {model} not found in MODEL_CARDS")
#             for token_budget in token_budgets:
#                 if token_budget >= 1:
#                     print("A token reduction strategy must specify a token_budget < 1!")
#                     continue

#                 op_class = cls.query_strategy_map[cls.query_strategy]
#                 physical_op_type = type(op_class.__name__ + model.name + str(int(100*token_budget)),
#                                         (op_class,),
#                                         {'model': model,
#                                         'prompt_strategy': prompt_strategy,
#                                         'final': True,
#                                         'token_budget': token_budget,
#                                         })
#                 return_operators.append(physical_op_type)

#         return return_operators

# class TokenReducedConventionalConvertStrategy(TokenReductionStrategy):
#     """
#     This strategy creates physical operator classes using a conventional query strategy with token reduction.
#     """
#     logical_op_class = logical.ConvertScan
#     physical_op_class = TokenReducedConvert
#     query_strategy = QueryStrategy.CONVENTIONAL

# class TokenReducedBondedConvertStrategy(TokenReductionStrategy):
#     """
#     This strategy creates physical operator classes using a bonded query strategy with token reduction.
#     """
#     logical_op_class = logical.ConvertScan
#     physical_op_class = TokenReducedConvert
#     query_strategy = QueryStrategy.BONDED

from __future__ import annotations
import json 
import time
from typing import List
from palimpzest.constants import Model
from .strategy import PhysicalOpStrategy


from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, convert

class LLMBondedQueryConvert(convert.LLMConvert):

    def convert(self, 
                candidate_content,
                fields) -> None:

        prompt = self._construct_query_prompt(fields_to_generate=fields)

        # generate all fields in a single query
        json_answers, field_stats = self._dspy_generate_fields(fields_to_generate=fields, content=candidate_content, prompt=prompt)

        # if there was an error, execute a conventional query
        if all([v is None for v in json_answers.values()]):
            print("Falling back to conventional conversion")
            conventional_op = type('LLMFallback',
                                    (convert.LLMConvertConventional,),
                                    {'model': self.model,
                                     'prompt_strategy': self.prompt_strategy})
            
            return conventional_op(
                inputSchema = self.inputSchema,
                outputSchema = self.outputSchema,
                shouldProfile = self.shouldProfile,
                query_strategy = self.query_strategy,
            ).convert(candidate_content, fields)
        
        return json_answers, field_stats


class BondedQueryStrategy(PhysicalOpStrategy):

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
                *args, **kwargs) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type(cls.__name__+model.name,
            physical_op_type = type(cls.__name__,
                                    (cls.physical_op_class,),
                                    {'model': model,
                                     'prompt_strategy': prompt_strategy})
            return_operators.append(physical_op_type)

        return return_operators
class BondedQueryConvertStrategy(BondedQueryStrategy):
    """
    This strategy creates physical operator classes using a bonded query strategy.
    It ties together several records for the same fields, possibly defaulting to a conventional conversion strategy.
    """
    logical_op_class = logical.ConvertScan
    physical_op_class = LLMBondedQueryConvert
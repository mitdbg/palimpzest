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

    def convert(self, candidate_content,
                fields) -> None:

        # generate all fields in a single query
        final_json_objects, query_stats = self._dspy_generate_fields(fields, content=candidate_content)

        # if there was an error, execute a conventional query
        if all([v is None for v in final_json_objects[0].values()]):
            # generate each field one at a time
            field_outputs = {}
            for field_name in fields:
                json_objects, field_stats = self._dspy_generate_fields([field_name], content = candidate_content)

                # update query_stats
                for key, value in field_stats.items():
                    if type(value) == type(dict()):
                        for k, v in value.items():
                            query_stats[key][k] = query_stats[key].get(k,0) + value[k]
                    else:
                        query_stats[key] += value

                # update field_outputs
                field_outputs[field_name] = json_objects

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
from __future__ import annotations
from typing import List
from palimpzest.constants import Model
from palimpzest.utils.model_helpers import getVisionModels
from .strategy import PhysicalOpStrategy


from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, filter, convert


class LLMStrategy(PhysicalOpStrategy):

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_QA,
                enable_vision: bool = True,
                *args, **kwargs) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            if not enable_vision and model in getVisionModels():
                continue
            # physical_op_type = type(cls.physical_op_class.__name__+model.name,
            physical_op_type = type(cls.physical_op_class.__name__,
                                    (cls.physical_op_class,),
                                    {'model': model,
                                     'prompt_strategy': prompt_strategy,
                                     'final': True,
                                     })
            return_operators.append(physical_op_type)


        return return_operators


class LLMFilterStrategy(LLMStrategy):

    logical_op_class = logical.FilteredScan
    physical_op_class = filter.LLMFilter

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_BOOL,
                *args, **kwargs) -> List[physical.PhysicalOperator]:
        return super(cls, LLMFilterStrategy).__new__(cls,
                                                    available_models, 
                                                    prompt_strategy,
                                                    enable_vision=False) # TODO hardcode for now 


class LLMConventionalConvertStrategy(LLMStrategy):
    """
    This strategy creates physical operator classes for the Conventional strategy 
    """
    logical_op_class = logical.ConvertScan
    physical_op_class = convert.LLMConvertConventional


class LLMBondedConvertStrategy(LLMStrategy):
    """
    This strategy creates physical operator classes using a bonded query strategy.
    It ties together several records for the same fields, possibly defaulting to a conventional conversion strategy.
    """
    logical_op_class = logical.ConvertScan
    physical_op_class = convert.LLMConvertBonded

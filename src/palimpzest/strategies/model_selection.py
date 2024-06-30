from __future__ import annotations
from typing import List
from palimpzest.constants import Model
from .strategy import PhysicalOpStrategy


from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, filter, convert


class ModelSelectionStrategy(PhysicalOpStrategy):

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            print(f"MODEL: {model}")
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type(cls.physical_op_class.__name__+model.name,
            physical_op_type = type(cls.physical_op_class.__name__,
                                    (cls.physical_op_class,),
                                    {'model': model,
                                     'prompt_strategy': prompt_strategy})
            print(f"PHYSOPTYPE: {physical_op_type}")
            return_operators.append(physical_op_type)

        print(f"return ops: {return_operators}")

        return return_operators

class ModelSelectionFilterStrategy(ModelSelectionStrategy):

    logical_op_class = logical.FilteredScan
    physical_op_class = filter.LLMFilter

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy,
                *args, **kwargs) -> List[physical.PhysicalOperator]:
        return super(cls, ModelSelectionFilterStrategy).__new__(cls, available_models, prompt_strategy=PromptStrategy.DSPY_COT_BOOL) # TODO hardcode for now 
class ModelSelectionConvertStrategy(ModelSelectionStrategy):
    """
    This strategy creates physical operator classes for the Conventional strategy 
    """
    logical_op_class = logical.ConvertScan
    physical_op_class = convert.LLMConvertConventional
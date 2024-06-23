from __future__ import annotations
from typing import List
from palimpzest.constants import Model
from .strategy import PhysicalOpStrategy


from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical


class ModelSelectionFilterStrategy(PhysicalOpStrategy):

    logical_op_class = logical.FilteredScan
    physical_op_class = physical.LLMFilter

    @staticmethod
    def __new__(cls, 
                available_models: List[Model]) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type('LLMFilter'+model.name,
            physical_op_type = type('LLMFilter',
                                    (cls.physical_op_class,),
                                    {'model': model})
            return_operators.append(physical_op_type)

        return return_operators
    
class ModelSelectionConvertStrategy(PhysicalOpStrategy):
    """
    This strategy creates physical operator classes for the Conventional strategy 
    """


    logical_op_class = logical.ConvertScan
    physical_op_class = physical.LLMConvert

    @staticmethod
    def __new__(cls, 
                available_models: List[Model]) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type('LLMConvert'+model.name,
            physical_op_type = type('LLMConvert',
                                    (cls.physical_op_class,),
                                    {'model': model})
            return_operators.append(physical_op_type)

        return return_operators
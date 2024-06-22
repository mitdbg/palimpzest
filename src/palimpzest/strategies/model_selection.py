from __future__ import annotations
from typing import List, Tuple
from palimpzest.constants import Model
from palimpzest.operators.logical import FilteredScan
from palimpzest.operators.filter import FilterOp, LLMFilter
from palimpzest.operators.physical import PhysicalOperator
from .strategy import PhysicalOpStrategy

from palimpzest.generators.generators import DSPyGenerator

from palimpzest.constants import *
from palimpzest.corelib import Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import *
from palimpzest.operators import logical


class ModelSelectionFilterStrategy(PhysicalOpStrategy):

    logical_op_class = logical.FilteredScan
    physical_op_class = LLMConvert

    @staticmethod
    def __new__(cls, 
                available_models: List[Model]) -> List[PhysicalOperator]:

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

    logical_op_class = logical.ConvertScan
    physical_op_class = DSPyGenerator

    @staticmethod
    def __new__(cls, 
                available_models: List[Model]) -> List[PhysicalOperator]:

        return_operators = []
        for model in available_models:
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type('DSPyGenerator'+model.name,
            physical_op_type = type('DSPyGenerator',
                                    (cls.physical_op_class,),
                                    {'model': model})
            return_operators.append(physical_op_type)

        return return_operators
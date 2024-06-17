from palimpzest.constants import Model
from palimpzest.operators.logical import FilteredScan
from palimpzest.operators.filter import FilterOp
from .strategy import PhysicalOpStrategy
from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator

from palimpzest.constants import *
from palimpzest.corelib import Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import *
from palimpzest.operators import logical

class ModelSelectionFilter(PhysicalOpStrategy):

    logical_op_class = FilteredScan
    physical_op_class = FilterOp

    # I want the execution to have some available models.
    # I want to tell my strategy: with model X,Y,Z, give me the relevant physical operator.
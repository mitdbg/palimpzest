from palimpzest.constants import Model
from palimpzest.operators.logical import FilteredScan
from palimpzest.operators.filter import FilterOp
from .strategy import PhysicalOpStrategy

class ModelSelectionFilter(PhysicalOpStrategy):

    logical_op_class = FilteredScan
    physical_op_class = FilterOp

    def __new__(cls, model: Model) -> FilterOp:
        #WIP         
        return ModelSelectionFilterOperator
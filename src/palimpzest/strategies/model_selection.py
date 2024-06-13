from palimpzest.constants import Model
from palimpzest.operators.logical import FilteredScan
from palimpzest.operators.filter import FilterOp
from .strategy import PhysicalOpStrategy

class ModelSelectionFilter(PhysicalOpStrategy):

    logical_op_class = FilteredScan
    physical_op_class = FilterOp

    def __new__(cls, model: Model) -> FilterOp:
        
        class ModelSelectionFilterOperator(PhysicalOperator):
            def __init__(self, threshold: float):
                self.threshold = threshold
            def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
                return data[data['model_selection_score'] > self.threshold]

        return ModelSelectionFilterOperator
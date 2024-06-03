from typing import List
from palimpzest.sets import Set
from palimpzest.operators import LogicalOperator
from palimpzest.operators.physical import PhysicalOp


class Plan():

    operators = []

    def __init__(self):
        raise NotImplementedError
    

class LogicalPlan(Plan):
    
        def __init__(self, datasets: List[Set] = [], operators:List[Operators] = []):
            self.dataset = datasets
            self.operators = operators
        
        def addOperator(self, operator):
            self.operators.append(operator)
        
        def getOperators(self):
            return self.operators
        
        def __repr__(self):
            return f"LogicalPlan: {self.operators}"
        
class PhysicalPlan(Plan):
        
        def __init__(self, dataset):
            self.dataset = dataset
            self.operators = []
        
        def addOperator(self, operator):
            self.operators.append(operator)
        
        def getOperators(self):
            return self.operators
        
        def __repr__(self):
            return f"LogicalPlan: {self.operators}"
     
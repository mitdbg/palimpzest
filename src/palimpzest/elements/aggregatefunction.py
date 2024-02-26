from .elements import *

#############################
# An AggregateFunction that can be applied to a Set of elements
#############################
class AggregateFunction():
    """A function that can be applied to a Set of Elements"""

    def serialize(a):
        return {"aggFuncDesc": a.funcDesc}
    
    def deserialize(d):
        return AggregateFunction(d["aggFuncDesc"])

    def __init__(self, funcDesc: str):
        self.funcDesc = funcDesc

    def __str__(self):
        return "AggregateFunction(" + self.funcDesc + ")"

    def __hash__(self):
            # Custom hash function
        # For example, hash based on the value attribute
        return hash(self.funcDesc)

    def __eq__(self, other):
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, AggregateFunction) and self.funcDesc == other.funcDesc



from __future__ import annotations
from typing import Any, Dict

#############################
# An AggregateFunction that can be applied to a Set of DataRecords
#############################
class AggregateFunction():
    """A function that can be applied to a Set of DataRecords"""
    def __init__(self, funcDesc: str) -> None:
        self.funcDesc = funcDesc

    def serialize(a) -> Dict[str, Any]:
        return {"aggFuncDesc": a.funcDesc}
    
    def deserialize(d) -> AggregateFunction:
        return AggregateFunction(d["aggFuncDesc"])

    def __str__(self) -> str:
        return "AggregateFunction(" + self.funcDesc + ")"

    def __hash__(self) -> int:
        # custom hash function
        return hash(self.funcDesc)

    def __eq__(self, other: AggregateFunction) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, AggregateFunction) and self.funcDesc == other.funcDesc



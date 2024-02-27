from __future__ import annotations
from typing import Any, Dict

#############################
# Filters that can be applied against a particular Schema
#############################
class Filter():
    """A filter that can be applied to a Set"""
    def __init__(self, filterCondition: str) -> None:
        self.filterCondition = filterCondition

    def serialize(self) -> Dict[str: Any]:
        return {"filterCondition": self.filterCondition}

    def deserialize(d: Dict[str, Any]) -> Filter:
        return Filter(d["filterCondition"])

    def __str__(self) -> str:
        return "Filter(" + self.filterCondition + ")"

    def __hash__(self) -> int:
        # custom hash function
        return hash(self.filterCondition)

    def __eq__(self, other) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, Filter) and self.filterCondition == other.filterCondition
   
    def test(self, objToTest) -> bool:
        """Test whether the object matches the filter condition"""
        return self._compiledFilter(objToTest)

    def _compiledFilter(self, target) -> bool:
        """This is the compiled version of the filter condition. It will be implemented at compile time."""
        pass

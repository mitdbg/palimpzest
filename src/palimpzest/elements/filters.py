from __future__ import annotations

from typing import Any, Dict


#############################
# Filters that can be applied against a particular Schema
#############################
# TODO: think through a way to give filter functions fixed strings that could not be affected by a copy
#       potentially changing the address of a function; I don't think this happens today, but it's worth safeguarding against
class Filter:
    """A filter that can be applied to a Set"""

    def __init__(self, filterCondition: str | None = None, filterFn: callable | None = None) -> None:
        self.filterCondition = filterCondition
        self.filterFn = filterFn

    def serialize(self) -> Dict[str, Any]:
        return {"filterCondition": self.filterCondition, "filterFn": str(self.filterFn)}

    def __str__(self) -> str:
        return "Filter(" + self.getFilterStr() + ")"

    def getFilterStr(self) -> str:
        return self.filterCondition if self.filterCondition is not None else str(self.filterFn)

    def __hash__(self) -> int:
        # custom hash function
        return hash(self.filterCondition) if self.filterCondition is not None else hash(str(self.filterFn))

    def __eq__(self, other: Filter) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return (
            isinstance(other, Filter)
            and self.filterCondition == other.filterCondition
            and self.filterFn == other.filterFn
        )

from __future__ import annotations

from typing import Any, Callable


#############################
# Filters that can be applied against a particular Schema
#############################
# TODO: think through a way to give filter functions fixed strings that could not be affected by a copy
#       potentially changing the address of a function; I don't think this happens today, but it's worth safeguarding against
class Filter:
    """A filter that can be applied to a Set"""

    def __init__(self, filter_condition: str | None = None, filter_fn: Callable | None = None) -> None:
        self.filter_condition = filter_condition
        self.filter_fn = filter_fn

    def serialize(self) -> dict[str, Any]:
        return {"filter_condition": self.filter_condition, "filter_fn": str(self.filter_fn)}

    def get_filter_str(self) -> str:
        return self.filter_condition if self.filter_condition is not None else str(self.filter_fn)

    def __repr__(self) -> str:
        return "Filter(" + self.get_filter_str() + ")"

    def __hash__(self) -> int:
        # custom hash function
        return hash(self.filter_condition) if self.filter_condition is not None else hash(str(self.filter_fn))

    def __eq__(self, other) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return (
            isinstance(other, Filter)
            and self.filter_condition == other.filter_condition
            and self.filter_fn == other.filter_fn
        )
    def __str__(self) -> str:
        return self.get_filter_str()

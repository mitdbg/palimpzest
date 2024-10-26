from __future__ import annotations

from typing import Any, Dict, Type

from palimpzest.corelib.fields import Field
from palimpzest.corelib.schemas import OperatorDerivedSchema, Schema


# signature for a group by aggregate that applies
# group and aggregation to an input tuple
class GroupBySig:
    def __init__(self, gbyFields: list[str], aggFuncs: list[str], aggFields: list[str]):
        self.gbyFields = gbyFields
        self.aggFuncs = aggFuncs
        self.aggFields = aggFields

    def validateSchema(self, inputSchema: Type[Schema]) -> tuple[bool, str | None]:
        for f in self.gbyFields:
            if not hasattr(inputSchema, f):
                return (False, "Supplied schema has no field " + f)
        for f in self.aggFields:
            if not hasattr(inputSchema, f):
                return (False, "Supplied schema has no field " + f)
        return (True, None)

    def serialize(self) -> Dict[str, Any]:
        out = {
            "groupByFields": self.gbyFields,
            "aggFuncs": self.aggFuncs,
            "aggFields": self.aggFields,
        }
        return out

    def __str__(self) -> str:
        return "GroupBy(" + repr(self.serialize()) + ")"

    def __hash__(self) -> int:
        # custom hash function
        return hash(repr(self.serialize()))

    def __eq__(self, other) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, GroupBySig) and self.serialize() == other.serialize()

    def getAggFieldNames(self) -> list[str]:
        ops = []
        for i in range(0, len(self.aggFields)):
            ops.append(self.aggFuncs[i] + "(" + self.aggFields[i] + ")")
        return ops

    def outputSchema(self) -> Type[OperatorDerivedSchema]:
        # the output class varies depending on the group by, so here
        # we dynamically construct this output
        Schema = type("CustomGroupBy", (OperatorDerivedSchema,), {})

        for g in self.gbyFields:
            f = Field(desc=g, required=True)
            setattr(Schema, g, f)
        ops = self.getAggFieldNames()
        for op in ops:
            f = Field(desc=op, required=True)
            setattr(Schema, op, f)
        return Schema

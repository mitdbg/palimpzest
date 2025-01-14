from __future__ import annotations

from typing import Any

from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import OperatorDerivedSchema, Schema


# signature for a group by aggregate that applies
# group and aggregation to an input tuple
class GroupBySig:
    def __init__(self, group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]):
        self.group_by_fields = group_by_fields
        self.agg_funcs = agg_funcs
        self.agg_fields = agg_fields

    def validate_schema(self, input_schema: Schema) -> tuple[bool, str | None]:
        for f in self.group_by_fields:
            if not hasattr(input_schema, f):
                return (False, "Supplied schema has no field " + f)
        for f in self.agg_fields:
            if not hasattr(input_schema, f):
                return (False, "Supplied schema has no field " + f)
        return (True, None)

    def serialize(self) -> dict[str, Any]:
        out = {
            "group_by_fields": self.group_by_fields,
            "agg_funcs": self.agg_funcs,
            "agg_fields": self.agg_fields,
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

    def get_agg_field_names(self) -> list[str]:
        ops = []
        for i in range(0, len(self.agg_fields)):
            ops.append(self.agg_funcs[i] + "(" + self.agg_fields[i] + ")")
        return ops

    def output_schema(self) -> type[OperatorDerivedSchema]:
        # the output class varies depending on the group by, so here
        # we dynamically construct this output
        schema = type("CustomGroupBy", (OperatorDerivedSchema,), {})

        for g in self.group_by_fields:
            f = Field(desc=g)
            setattr(schema, g, f)
        ops = self.get_agg_field_names()
        for op in ops:
            f = Field(desc=op)
            setattr(schema, op, f)
        return schema

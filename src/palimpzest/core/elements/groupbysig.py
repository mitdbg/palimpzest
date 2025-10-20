from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from palimpzest.core.lib.schemas import create_schema_from_fields

# TODO:
# - move the arguments for group_by_fields, agg_funcs, and agg_fields into the Dataset.groupby() operator
# - construct the correct output schema using the input schema and the group by and aggregation fields
# - remove/update all other references to GroupBySig in the codebase

# TODO:
# - move the arguments for group_by_fields, agg_funcs, and agg_fields into the Dataset.groupby() operator
# - construct the correct output schema using the input schema and the group by and aggregation fields
# - remove/update all other references to GroupBySig in the codebase

# signature for a group by aggregate that applies
# group and aggregation to an input tuple
class GroupBySig:
    def __init__(self, group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]):
        self.group_by_fields = group_by_fields
        self.agg_funcs = agg_funcs
        self.agg_fields = agg_fields

    def validate_schema(self, input_schema: type[BaseModel]) -> tuple[bool, str | None]:
        for f in self.group_by_fields:
            if f not in input_schema.model_fields:
                return (False, "Supplied schema has no field " + f)
        for f in self.agg_fields:
            if f not in input_schema.model_fields:
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

    # TODO: output schema needs to account for input schema types and create new output schema types
    def output_schema(self) -> type[BaseModel]:
        # the output class varies depending on the group by, so here
        # we dynamically construct this output
        fields = []
        for g in self.group_by_fields:
            f = {"name": g, "type": Any, "desc": f"Group by field: {g}"}
            fields.append(f)

        ops = self.get_agg_field_names()
        for op in ops:
            f = {"name": op, "type": Any, "desc": f"Aggregate field: {op}"}
            fields.append(f)

        return create_schema_from_fields(fields)

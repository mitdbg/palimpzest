import click

from typing import Literal


class AutoIntParamType(click.ParamType):
    name = "integer"

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> int | Literal["max"] | None:
        if isinstance(value, int):
            return value

        if value == "max":
            return "max"

        self.fail("Invalid integer value: {value}")


INT_WITH_MAX = AutoIntParamType()

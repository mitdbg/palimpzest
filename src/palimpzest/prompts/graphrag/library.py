from __future__ import annotations

from functools import lru_cache
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined


@lru_cache(maxsize=1)
def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("palimpzest.prompts.graphrag", "templates"),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def render(template_name: str, **context: Any) -> str:
    """Render a GraphRAG prompt template from `prompts/graphrag/templates`."""

    tmpl = _env().get_template(template_name)
    return tmpl.render(**context)

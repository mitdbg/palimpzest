from __future__ import annotations

import pytest

from palimpzest.graphrag.deciders import (
    render_admittance_decision_prompt,
    render_synthesis_prompt,
    render_termination_decision_prompt,
    CMS_COMP_OPS_SYSTEM_PROMPT,
)
from palimpzest.prompts.graphrag.library import render


def test_graphrag_templates_render_smoke() -> None:
    txt = render("bootstrap_admittance.j2", query="What is X?")
    assert "Return plain text criteria" in txt

    txt2 = render("bootstrap_synthesis.j2", query="What is X?", system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT)
    assert "Return plain text" in txt2

    txt3 = render(
        "synthesis_answer.j2",
        query="What is X?",
        synthesis_criteria="Use only provided docs",
        context_block="[doc 1] node_id=n1\nhello",
        system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT,
    )
    assert "Use only" in txt3


def test_admittance_decision_prompt_renders() -> None:
    prompt = render_admittance_decision_prompt(
        query="What is X?",
        admittance_criteria="Admit nodes about X.",
        node_id="n1",
        depth=2,
        score=0.5,
        path_node_ids=["n0", "n1"],
        node_text="Some content",
    )
    assert "Return STRICT JSON" in prompt
    assert "\"decision\"" in prompt


def test_termination_decision_prompt_renders() -> None:
    prompt = render_termination_decision_prompt(
        query="What is X?",
        termination_criteria="Stop when enough evidence.",
        state={"steps": 10},
        node_text="Some content",
    )
    assert "Return STRICT JSON" in prompt


def test_synthesis_prompt_renders() -> None:
    prompt = render_synthesis_prompt(
        query="What is X?",
        synthesis_criteria="Use only provided docs",
        context_block="[doc 1] node_id=n1\nhello",
        system_prompt=CMS_COMP_OPS_SYSTEM_PROMPT,
    )
    assert "What is X?" in prompt
    assert "Use only provided docs" in prompt
    assert "[doc 1]" in prompt


def test_missing_template_variable_raises() -> None:
    with pytest.raises(Exception):
        render("admittance_decision.j2", query="q")

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import litellm

from palimpzest.prompts.graphrag.library import render


@dataclass(frozen=True)
class LLMBooleanDeciderConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float | None = 30.0


class LLMBooleanDecider:
    """Tiny utility for yes/no decisions via LiteLLM."""

    def __init__(self, *, config: LLMBooleanDeciderConfig) -> None:
        self.config = config

    def decide_prompt(self, *, prompt: str) -> tuple[bool, str | None, str | None]:
        decision, reason, raw_output, _meta = self.decide_prompt_with_meta(prompt=prompt)
        return decision, reason, raw_output

    def decide_prompt_with_meta(self, *, prompt: str) -> tuple[bool, str | None, str | None, dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            resp = litellm.completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_s,
            )
        except Exception as e:
            return False, f"litellm_error:{type(e).__name__}", None, {
                "latency_s": time.perf_counter() - t0,
                "input_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "cost_usd": None,
            }

        latency_s = time.perf_counter() - t0

        usage = None
        try:
            usage = getattr(resp, "usage", None)
        except Exception:
            usage = None

        # Best-effort token accounting across providers.
        input_tokens = None
        output_tokens = None
        cached_tokens = None
        try:
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                # Some providers include nested details.
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens")
            else:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                details = getattr(usage, "prompt_tokens_details", None)
                if isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens")
        except Exception:
            input_tokens = None
            output_tokens = None
            cached_tokens = None

        # Best-effort cost (USD) using LiteLLM helpers when available.
        cost_usd = None
        try:
            # litellm.completion_cost supports multiple response formats.
            cost_usd = float(litellm.completion_cost(resp))  # type: ignore[attr-defined]
        except Exception:
            cost_usd = None

        content = None
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = None

        if not isinstance(content, str) or not content.strip():
            return False, "empty_response", content if isinstance(content, str) else None, {
                "latency_s": latency_s,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "cost_usd": cost_usd,
            }

        try:
            obj = json.loads(content)
        except Exception:
            # best-effort recovery: find outermost JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(content[start : end + 1])
                except Exception:
                    return False, "non_json_response", content, {
                        "latency_s": latency_s,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": cached_tokens,
                        "cost_usd": cost_usd,
                    }
            else:
                return False, "non_json_response", content, {
                    "latency_s": latency_s,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cost_usd": cost_usd,
                }

        decision = bool(obj.get("decision"))
        reason = obj.get("reason")
        return decision, reason if isinstance(reason, str) else None, content, {
            "latency_s": latency_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
        }

    def decide(self, *, instruction: str, payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        prompt = (
            instruction
            + "\n\n"
            + "Return STRICT JSON only with keys: {\"decision\": true|false, \"reason\": string}.\n"
            + "Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        return self.decide_prompt(prompt=prompt)


def render_admittance_decision_prompt(
    *,
    query: str,
    admittance_criteria: str,
    node_id: str,
    depth: int,
    score: float,
    path_node_ids: list[str],
    node_text: str,
) -> str:
    return render(
        "admittance_decision.j2",
        query=query,
        admittance_criteria=admittance_criteria,
        node_id=node_id,
        depth=depth,
        score=score,
        path_node_ids=path_node_ids,
        node_text=node_text,
    )


def render_termination_decision_prompt(
    *,
    query: str,
    termination_criteria: str,
    state: dict[str, Any],
    node_text: str,
) -> str:
    return render(
        "termination_decision.j2",
        query=query,
        termination_criteria=termination_criteria,
        state_json=json.dumps(state, ensure_ascii=False),
        node_text=node_text,
    )


def _bootstrap_one(*, model: str, template: str, query: str) -> str:
    prompt = render(template, query=query)
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        timeout=60.0,
    )
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"bootstrap_empty:{template}")
    return content.strip()


def bootstrap_ranking_criteria(*, model: str, query: str) -> str:
    return _bootstrap_one(model=model, template="bootstrap_ranking.j2", query=query)


def bootstrap_admittance_criteria(*, model: str, query: str) -> str:
    return _bootstrap_one(model=model, template="bootstrap_admittance.j2", query=query)


def bootstrap_termination_criteria(*, model: str, query: str) -> str:
    return _bootstrap_one(model=model, template="bootstrap_termination.j2", query=query)


def bootstrap_meta_prompts(*, model: str, query: str) -> dict[str, str]:
    """Convenience wrapper that bootstraps all three criteria strings."""

    ranking = bootstrap_ranking_criteria(model=model, query=query)
    admittance = bootstrap_admittance_criteria(model=model, query=query)
    termination = bootstrap_termination_criteria(model=model, query=query)

    return {
        "ranking_prompt": ranking,
        "admittance_prompt": admittance or build_admittance_instruction(),
        "termination_prompt": termination or build_termination_instruction(),
    }


def build_admittance_instruction() -> str:
    return (
        "You are a gate for graph traversal. Decide if the current node is relevant to the query "
        "and worth exploring/emitting. Be conservative: only admit nodes that plausibly contain information "
        "needed to answer the query."  # noqa: E501
    )


def build_termination_instruction() -> str:
    return (
        "You are a termination controller for graph traversal. Decide if traversal should stop now. "
        "Terminate when the visited/admitted nodes are sufficient to answer the query, or if continuing is unlikely "
        "to add value."  # noqa: E501
    )

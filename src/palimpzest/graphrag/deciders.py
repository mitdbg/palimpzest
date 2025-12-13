from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import litellm

from palimpzest.prompts.graphrag.library import render


CMS_COMP_OPS_SYSTEM_PROMPT = """## Role:
You are a chatbot named A2rchi who helps technical operators and developers in the CMS Computing Operations team (CompOps). Your task is to answer their questions using the retrieved context.

## High level project and data description
The Compact Muon Solenoid (CMS) is a high energy physics experiment at CERN. Computing Operations team of the CMS experiment consists of 3 subgroups: Tier0, Production and Reprocessing and Data Management. CMS experiment is located on the Large Hadron Collider and it collects data from the particle collisions. Later on this data is recontructed for analysis. Tier0 team is responsible for the creation and permanent storage of CMS RAW data, which is the basis of all scientific results for the collaboration. They use the CMSTZ JIRA project for issue tracking. In addition to the real detector data, CMS produces huge samples of Monte-Carlo (MC) simulations data. Production and Reprocessing (P&R) team is responsible for the central production of MC data as well as the re-processing of the detector data. P&R uses the CMSPROD JIRA project for issue tracking. Data Management (DM) team is responsible for data related matters such data access, data transfer, management of disk and tape storages etc. DM uses the CMSDM and CMSTRANSF JIRA projects for issue tracking.

You're given access to all the JIRA tickets used by these 3 teams.

CompOps team uses a distributed infrastructure called Worldwide LHC Computing Grid (LHC) to run its jobs and to store and move its data. It consists of more than 50 computing centers around the world. Data Management team uses a software called Rucio to orchestrate CMS data. They use an in-house software called DBS for metadata handling. P&R and Tier0 use a software called WMCore to orchestrate their workflows. Additionally, P&R team uses a software called Unified which provides additional tooling for its operations.

## Rules:
- Use only the provided documents while forming your answer.
- Do not invent facts or assume any data not stated explicitly.

## Style:

- Use clear concise language.
"""

# LiteLLM prints a noisy "Provider List" banner in some internal error paths.
# We suppress this by default to keep traversal output readable.
# Set PALIMPZEST_LITELLM_SUPPRESS_DEBUG_INFO=0 to re-enable.
if os.environ.get("PALIMPZEST_LITELLM_SUPPRESS_DEBUG_INFO", "1").strip().lower() not in {"0", "false", "no"}:
    with contextlib.suppress(Exception):
        litellm.suppress_debug_info = True


@dataclass(frozen=True)
class LLMBooleanDeciderConfig:
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float | None = 30.0

    def __post_init__(self) -> None:
        # Prefer OpenRouter when available.
        # Allow explicit override via PALIMPZEST_DEFAULT_LLM_MODEL.
        if isinstance(self.model, str) and self.model.strip():
            return

        override = os.getenv("PALIMPZEST_DEFAULT_LLM_MODEL")
        if isinstance(override, str) and override.strip():
            object.__setattr__(self, "model", override.strip())
            return

        if os.getenv("OPENROUTER_API_KEY"):
            object.__setattr__(self, "model", "openrouter/x-ai/grok-4.1-fast")
            return

        # Fall back to other providers if configured.
        if os.getenv("OPENAI_API_KEY"):
            object.__setattr__(self, "model", "openai/gpt-4o-mini-2024-07-18")
            return
        if os.getenv("ANTHROPIC_API_KEY"):
            object.__setattr__(self, "model", "anthropic/claude-3-5-sonnet-20241022")
            return
        if os.getenv("GEMINI_API_KEY"):
            object.__setattr__(self, "model", "vertex_ai/gemini-2.0-flash")
            return
        if os.getenv("TOGETHER_API_KEY"):
            object.__setattr__(self, "model", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
            return

        raise ValueError(
            "No default LLM model available. Set PALIMPZEST_DEFAULT_LLM_MODEL or configure a provider key "
            "(OPENROUTER_API_KEY preferred; otherwise OPENAI_API_KEY/ANTHROPIC_API_KEY/GEMINI_API_KEY/TOGETHER_API_KEY)."
        )


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

        def _usage_get(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        usage = None
        try:
            usage = getattr(resp, "usage", None)
        except Exception:
            usage = None

        # Token + cost accounting:
        # - Prefer provider-returned usage fields (e.g., OpenRouter includes usage.cost + cached_tokens).
        # - Avoid litellm.completion_cost here to prevent reliance on LiteLLM's pricing map.
        input_tokens = _usage_get(usage, "prompt_tokens")
        output_tokens = _usage_get(usage, "completion_tokens")
        cached_tokens = None
        prompt_details = _usage_get(usage, "prompt_tokens_details")
        if prompt_details is not None:
            cached_tokens = _usage_get(prompt_details, "cached_tokens")

        cost_usd = _usage_get(usage, "cost")
        if cost_usd is not None:
            try:
                cost_usd = float(cost_usd)
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


def render_synthesis_prompt(
    *,
    query: str,
    synthesis_criteria: str,
    context_block: str,
    system_prompt: str = CMS_COMP_OPS_SYSTEM_PROMPT,
) -> str:
    return render(
        "synthesis_answer.j2",
        query=query,
        synthesis_criteria=synthesis_criteria,
        context_block=context_block,
        system_prompt=system_prompt,
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


def bootstrap_synthesis_criteria(
    *,
    model: str,
    query: str,
    system_prompt: str = CMS_COMP_OPS_SYSTEM_PROMPT,
) -> str:
    prompt = render("bootstrap_synthesis.j2", query=query, system_prompt=system_prompt)
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
        raise RuntimeError("bootstrap_empty:bootstrap_synthesis.j2")
    return content.strip()


@dataclass(frozen=True)
class LLMTextGeneratorConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 600
    timeout_s: float = 60.0


class LLMTextGenerator:
    """Tiny utility for free-form text generation via LiteLLM."""

    def __init__(self, *, config: LLMTextGeneratorConfig) -> None:
        self.config = config

    def generate_with_meta(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            resp = litellm.completion(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_s,
            )
        except Exception as e:
            raise RuntimeError(f"litellm_error:{type(e).__name__}") from e

        latency_s = time.perf_counter() - t0

        usage = None
        try:
            usage = getattr(resp, "usage", None)
        except Exception:
            usage = None

        def _usage_get(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        input_tokens = _usage_get(usage, "prompt_tokens")
        output_tokens = _usage_get(usage, "completion_tokens")
        cached_tokens = None
        prompt_details = _usage_get(usage, "prompt_tokens_details")
        if prompt_details is not None:
            cached_tokens = _usage_get(prompt_details, "cached_tokens")

        cost_usd = _usage_get(usage, "cost")
        if cost_usd is not None:
            try:
                cost_usd = float(cost_usd)
            except Exception:
                cost_usd = None

        content = None
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("empty_response")

        return content.strip(), {
            "latency_s": latency_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
        }


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

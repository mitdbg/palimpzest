"""Minimal LiteLLM smoke test for OpenRouter models.

Usage (recommended):
  doppler run -- /Users/jason/projects/mit/palimpzest/.venv/bin/python scratchpad/scripts/test_litellm_openrouter.py \
    --model openrouter/x-ai/grok-4.1-fast

This script is intentionally tiny so failures are easy to interpret.
It prints:
- whether the call succeeded
- any exception type/message
- token usage (if present)
- cost estimate via litellm.completion_cost (if supported)

Notes:
- Requires OPENROUTER_API_KEY in env for OpenRouter models.
- Keeps prompts short to avoid rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional


def _safe_cost_usd(resp: Any) -> Optional[float]:
    try:
        import litellm  # type: ignore

        # litellm.completion_cost supports multiple response formats.
        cost = litellm.completion_cost(resp)  # type: ignore[attr-defined]
        if cost is None:
            return None
        return float(cost)
    except Exception:
        return None


def _extract_usage(resp: Any) -> Dict[str, Any]:
    usage = {}
    try:
        # LiteLLM returns an OpenAI-like object; it can be dict-like.
        if isinstance(resp, dict):
            usage = resp.get("usage") or {}
        else:
            usage = getattr(resp, "usage", None) or {}
    except Exception:
        usage = {}

    # Normalize to JSON-safe plain dict.
    def _to_jsonable(v: Any) -> Any:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, dict):
            return {str(k): _to_jsonable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        # Handle pydantic / wrappers / dataclasses
        for attr in ("model_dump", "dict"):
            fn = getattr(v, attr, None)
            if callable(fn):
                try:
                    return _to_jsonable(fn())
                except Exception:
                    pass
        return str(v)

    if not isinstance(usage, dict):
        try:
            usage = dict(usage)
        except Exception:
            usage = {}
    return _to_jsonable(usage)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/x-ai/grok-4.1-fast")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Basic env sanity: avoid printing keys.
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    print(json.dumps({"has_OPENROUTER_API_KEY": has_openrouter, "model": args.model}))

    import litellm  # type: ignore

    if args.debug:
        # LiteLLM itself recommends this for actionable errors.
        try:
            litellm._turn_on_debug()  # type: ignore[attr-defined]
        except Exception:
            pass

    messages = [
        {
            "role": "user",
            "content": "Return STRICT JSON only: {\"ok\": true, \"reason\": \"smoke\"}",
        }
    ]

    # OpenRouter often benefits from explicit api_base + headers; LiteLLM can infer,
    # but we set it here to reduce ambiguity.
    extra_headers = {
        # These are optional, but recommended by OpenRouter.
        "HTTP-Referer": "https://local.palimpzest",
        "X-Title": "palimpzest-smoke-test",
    }

    t0 = time.time()
    try:
        resp = litellm.completion(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            api_base="https://openrouter.ai/api/v1",
            extra_headers=extra_headers,
        )
    except Exception as e:
        dt = time.time() - t0
        print(json.dumps({"ok": False, "elapsed_s": round(dt, 3), "exc_type": type(e).__name__, "exc": str(e)}))
        return 2

    dt = time.time() - t0

    # Extract text content conservatively.
    content = None
    try:
        if isinstance(resp, dict):
            content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
        else:
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None) if msg is not None else None
    except Exception:
        content = None

    usage = _extract_usage(resp)
    cost_usd = _safe_cost_usd(resp)

    print(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": round(dt, 3),
                "usage": usage,
                "cost_usd": cost_usd,
                "content_preview": (content or "")[:200],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

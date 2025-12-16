from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient

from palimpzest.server.graphrag_app import DEFAULT_SNAPSHOT_PATH, RunRequest, create_app


DEFAULT_QUERY_TEXT = (
    "What are the most common operational challenges across all three groups "
    "(Data management, Production and Reprocessing, Tier0)?"
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() else default


def _env_opt_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    # Distinguish between "unset" (env missing) and "explicitly disabled".
    if not raw or raw.lower() in {"none", "null", "off", "disabled"}:
        return ""
    return raw


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _compact_trace_payload(payload: dict, *, max_text_chars: int = 240) -> dict:
    """Return a compacted version of a trace payload suitable for stdout."""

    def _truncate(s: str) -> str:
        s = s.replace("\n", "\\n")
        if len(s) <= max_text_chars:
            return s
        return s[: max_text_chars - 12] + "…(truncated)"

    out = dict(payload)
    node = out.get("node")
    if isinstance(node, dict):
        node2 = dict(node)
        txt = node2.get("text")
        if isinstance(txt, str):
            node2["text"] = _truncate(txt)
        out["node"] = node2

    neighbors = out.get("neighbors")
    if isinstance(neighbors, list) and len(neighbors) > 20:
        out["neighbors"] = neighbors[:20] + [{"_note": f"{len(neighbors) - 20} more omitted"}]

    return out


def main() -> None:
    snapshot_path = Path(os.getenv("PZ_GRAPH_SNAPSHOT_PATH", str(DEFAULT_SNAPSHOT_PATH)))
    if not snapshot_path.is_absolute():
        snapshot_path = Path.cwd() / snapshot_path

    app = create_app(snapshot_path=snapshot_path)
    client = TestClient(app)

    query_text = _env_str("PZ_TEST_QUERY", DEFAULT_QUERY_TEXT)
    edge_type = _env_str("PZ_TEST_EDGE_TYPE", "all")
    entry_points = _env_int("PZ_TEST_ENTRY_POINTS", 5)
    max_steps = _env_int("PZ_TEST_MAX_STEPS", 200)
    ranking_model = _env_opt_str("PZ_TEST_RANKING_MODEL")
    admittance_model = _env_opt_str("PZ_TEST_ADMITTANCE_MODEL")
    termination_model = _env_opt_str("PZ_TEST_TERMINATION_MODEL")
    debug_trace = _env_bool("PZ_TEST_DEBUG_TRACE", default=False)
    print_trace_json = _env_bool("PZ_TEST_PRINT_TRACE_JSON", default=False)

    payload = RunRequest(
        index="cms_standard",
        query=query_text,
        entry_points=entry_points,
        max_steps=max_steps,
        edge_type=edge_type,
        ranking_model=ranking_model,
        admittance_model=admittance_model,
        termination_model=termination_model,
        debug_trace=debug_trace,
    ).model_dump(mode="json")

    res = client.post("/api/run", json=payload)
    res.raise_for_status()
    run_id = res.json()["run_id"]

    events: list[dict] = []
    with client.websocket_connect(f"/ws/{run_id}") as ws:
        while True:
            try:
                events.append(json.loads(ws.receive_text()))
            except Exception:
                break

    export_dir = Path("CURRENT_WORKSTREAM/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"graphrag_trace_{run_id}.jsonl"
    export_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")

    # Print a quick summary, and the final `result` payload if present.
    event_types = [e.get("event_type") for e in events]
    print(f"run_id={run_id}")
    print(f"events_export={export_path}")
    counts = Counter(t for t in event_types if t)
    print(f"events={len(events)} event_type_counts={dict(counts)}")

    trace_events = [e for e in events if e.get("event_type") == "traverse_trace"]
    if debug_trace:
        print(f"traverse_trace_events={len(trace_events)}")
        for e in trace_events:
            data = e.get("data")
            if isinstance(data, dict):
                et = data.get("event_type")
                step = data.get("step")
                node_id = data.get("node_id") or (data.get("popped") or {}).get("node_id")
                # Print one line per trace event; full payload is in the export.
                print(f"TRACE et={et} step={step} node_id={node_id}")

        if print_trace_json:
            for e in trace_events:
                data = e.get("data")
                if isinstance(data, dict):
                    print(json.dumps(_compact_trace_payload(data), ensure_ascii=False))

    result = next((e for e in events if e.get("event_type") == "result"), None)
    if result:
        data = result.get("data") or {}
        answer = data.get("answer")
        path = data.get("path")
        print("result.answer=", answer)
        print("result.path_len=", len(path) if isinstance(path, list) else None)

        visited = None
        if isinstance(answer, str):
            m = re.search(r"Visited\s+(\d+)\s+nodes\.", answer)
            if m:
                visited = int(m.group(1))

        # Treat missing traversal as a failed smoke test.
        if visited == 0 or counts.get("search_step", 0) == 0:
            raise SystemExit(1)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

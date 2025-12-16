"""Generate a per-step markdown audit from a GraphRAG trace JSONL.

Usage:
  .venv/bin/python scratchpad/scripts/audit_graphrag_trace.py \
    --trace CURRENT_WORKSTREAM/exports/graphrag_trace_DEEPDIVE.jsonl \
    --out   CURRENT_WORKSTREAM/notes/trace_deepdive_audit.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _clip(s: str | None, max_chars: int) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _md_inline(obj: Any) -> str:
    return "`" + str(obj).replace("`", "\\`") + "`"


def _md_quote(s: str) -> str:
    s = s.rstrip("\n")
    lines = s.splitlines() or [""]
    return "\n".join("> " + line for line in lines)


@dataclass
class TraceContext:
    run_id: str | None = None
    query_text: str | None = None
    traverse_init: dict[str, Any] | None = None
    seed_nodes: list[dict[str, Any]] | None = None
    result: dict[str, Any] | None = None
    event_counts: Counter[str] | None = None


def _load_trace(trace_path: Path) -> tuple[TraceContext, dict[int, list[dict[str, Any]]]]:
    ctx = TraceContext(seed_nodes=[])
    step_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_counts: Counter[str] = Counter()

    with trace_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            et = ev.get("event_type")
            if isinstance(et, str):
                event_counts[et] += 1

            if et == "trace_init":
                ctx.run_id = (ev.get("data") or {}).get("run_id")
            elif et == "query_start":
                ctx.query_text = ev.get("query_text")
            elif et == "result":
                ctx.result = ev.get("data")
            elif et == "traverse_trace":
                d = ev.get("data") or {}
                inner_et = d.get("event_type")
                if inner_et == "traverse_init":
                    ctx.traverse_init = d
                elif inner_et == "seed_node":
                    (ctx.seed_nodes or []).append(d)
                else:
                    step = d.get("step")
                    if isinstance(step, int):
                        step_events[step].append(d)

    ctx.event_counts = event_counts
    return ctx, dict(step_events)


def _write_audit(
    *,
    ctx: TraceContext,
    step_events: dict[int, list[dict[str, Any]]],
    trace_path: Path,
    out_path: Path,
    max_prompt_chars: int,
    max_output_chars: int,
    max_node_text_chars: int,
) -> None:
    lines: list[str] = []

    lines.append("# Trace Deep Dive Audit")
    lines.append("")
    lines.append(f"Trace file: {_md_inline(trace_path.as_posix())}")
    if ctx.run_id:
        lines.append(f"Run id: {_md_inline(ctx.run_id)}")
    if ctx.query_text:
        lines.append(f"Query: {_md_inline(ctx.query_text)}")

    if ctx.event_counts is not None:
        lines.append("")
        lines.append("## Event Counts")
        lines.append(f"- total_lines: {trace_path.stat().st_size} bytes")
        for k, v in ctx.event_counts.most_common():
            lines.append(f"- {k}: {v}")

    if ctx.traverse_init:
        lines.append("")
        lines.append("## Traverse Init")
        for k in (
            "graph_id",
            "graph_revision",
            "start_node_ids",
            "edge_type",
            "include_overlay",
            "beam_width",
            "max_steps",
            "allow_revisit",
            "ranker_id",
            "visit_filter_id",
            "admittance_id",
            "termination_id",
            "node_program_id",
            "tracer_id",
        ):
            if k in ctx.traverse_init:
                lines.append(f"- {k}: {_md_inline(ctx.traverse_init.get(k))}")

    if ctx.seed_nodes:
        lines.append("")
        lines.append("## Seed Nodes")
        for sn in ctx.seed_nodes:
            node = sn.get("node") or {}
            label = node.get("label")
            ntype = node.get("type")
            lines.append(
                "- "
                + " ".join(
                    [
                        f"node_id={_md_inline(sn.get('node_id'))}",
                        f"depth={sn.get('depth')}",
                        f"score={sn.get('score')}",
                        f"type={_md_inline(ntype)}",
                        f"label={_md_inline(label)}" if label else "label=None",
                    ]
                )
            )

    for step in sorted(step_events.keys()):
        evs = step_events[step]
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in evs:
            by_type[str(d.get("event_type"))].append(d)

        lines.append("")
        lines.append(f"## Step {step}")

        begin = (by_type.get("step_begin") or [None])[0]
        if begin:
            lines.append(f"- popped: {begin.get('popped')}")
            lines.append(
                f"- frontier_size_before: {begin.get('frontier_size_before')} visited_count: {begin.get('visited_count')} admitted_nodes: {begin.get('admitted_nodes')}"
            )

        loaded = (by_type.get("step_node_loaded") or [None])[0]
        if loaded:
            node = loaded.get("node") or {}
            text_prefix = _clip(node.get("text"), max_node_text_chars)
            lines.append(
                "- node: "
                + " ".join(
                    [
                        f"node_id={_md_inline(loaded.get('node_id'))}",
                        f"type={_md_inline(node.get('type'))}",
                        f"label={_md_inline(node.get('label'))}" if node.get("label") else "label=None",
                    ]
                )
            )
            if text_prefix:
                lines.append(f"- node_text (prefix): {_md_inline(text_prefix)}")

        vf = (by_type.get("step_gate_visit_filter") or [None])[0]
        if vf:
            lines.append(
                f"- visit_filter: passed={vf.get('passed')} visit_filter_id={_md_inline(vf.get('visit_filter_id'))}"
            )

        adm = (by_type.get("step_gate_admittance") or [None])[0]
        if adm:
            lines.append(
                f"- admittance: admitted={adm.get('admitted')} model={_md_inline(adm.get('model'))} admittance_id={_md_inline(adm.get('admittance_id'))}"
            )
            if adm.get("reason"):
                lines.append(f"- admittance.reason: {_md_inline(adm.get('reason'))}")
            if adm.get("raw_output"):
                lines.append(
                    f"- admittance.raw_output (clipped): {_md_inline(_clip(adm.get('raw_output'), max_output_chars))}"
                )
            if adm.get("prompt"):
                lines.append(
                    f"- admittance.prompt (clipped):\n{_md_quote(_clip(adm.get('prompt'), max_prompt_chars) or '')}"
                )

        npd = (by_type.get("step_node_program_done") or [None])[0]
        if npd:
            lines.append(
                f"- node_program: produced={npd.get('produced')} node_program_id={_md_inline(npd.get('node_program_id'))}"
            )

        if by_type.get("step_emit_traversal_record"):
            lines.append("- emitted: traversal_record=True")

        term = (by_type.get("step_termination") or [None])[0]
        if term:
            lines.append(
                f"- termination: terminated={term.get('terminated')} termination_id={_md_inline(term.get('termination_id'))}"
            )
            if term.get("reason"):
                lines.append(f"- termination.reason: {_md_inline(term.get('reason'))}")
            if term.get("raw_output"):
                lines.append(
                    f"- termination.raw_output (clipped): {_md_inline(_clip(term.get('raw_output'), max_output_chars))}"
                )
            if term.get("prompt"):
                lines.append(
                    f"- termination.prompt (clipped):\n{_md_quote(_clip(term.get('prompt'), max_prompt_chars) or '')}"
                )

        exp = (by_type.get("step_expand") or [None])[0]
        if exp:
            neighbors = exp.get("neighbors") or []
            lines.append(
                f"- expand: expanded_edges={exp.get('expanded_edges')} neighbors={len(neighbors)} frontier_size_after={exp.get('frontier_size_after')} visited_count={exp.get('visited_count')}"
            )
            for i, n in enumerate(neighbors):
                parts = [
                    f"neighbor_id={_md_inline(n.get('neighbor_id'))}",
                    f"edge_type={_md_inline(n.get('edge_type'))}",
                    f"direction={_md_inline(n.get('direction'))}",
                    f"enqueued={n.get('enqueued')}",
                ]
                if n.get("skip_reason") is not None:
                    parts.append(f"skip_reason={_md_inline(n.get('skip_reason'))}")
                if n.get("score") is not None:
                    parts.append(f"score={n.get('score')}")
                lines.append(f"- neighbor[{i}]: " + " ".join(parts))

        end = (by_type.get("step_end") or [None])[0]
        if end:
            lines.append(
                f"- step_end: skipped={end.get('skipped')} skip_reason={_md_inline(end.get('skip_reason'))} frontier_size_after={end.get('frontier_size_after')} visited_count={end.get('visited_count')}"
            )

    if ctx.result:
        lines.append("")
        lines.append("## Result")
        if "answer" in ctx.result:
            lines.append(f"- answer: {_md_inline(ctx.result.get('answer'))}")
        path = ctx.result.get("path")
        if isinstance(path, list):
            lines.append(f"- path_len: {len(path)}")
            lines.append(f"- path: {_md_inline(path)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-prompt-chars", type=int, default=800)
    p.add_argument("--max-output-chars", type=int, default=800)
    p.add_argument("--max-node-text-chars", type=int, default=240)
    args = p.parse_args()

    ctx, step_events = _load_trace(args.trace)
    _write_audit(
        ctx=ctx,
        step_events=step_events,
        trace_path=args.trace,
        out_path=args.out,
        max_prompt_chars=args.max_prompt_chars,
        max_output_chars=args.max_output_chars,
        max_node_text_chars=args.max_node_text_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

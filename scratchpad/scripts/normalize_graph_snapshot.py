#!/usr/bin/env python3
"""Normalize Palimpzest graph snapshot JSON to match the `GraphSnapshot` schema.

Why this exists:
- Some external graph exporters emit `version` as a string (e.g. "1.1")
- Some omit required `revision`
- Some use `links` instead of `edges`

This script makes the minimal, non-destructive fixes so the UI/server can load the snapshot.

Usage:
  python scratchpad/scripts/normalize_graph_snapshot.py data/my_graph.json --inplace
  python scratchpad/scripts/normalize_graph_snapshot.py data/my_graph.json --output data/my_graph.fixed.json

Notes:
- `GraphSnapshot.version` is an int in this repo. We coerce common string formats.
- `GraphSnapshot.revision` is required. We default to 0 if missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        # Common formats: "1", "1.0", "1.1"
        try:
            if "." in s:
                return int(float(s))
            return int(s)
        except ValueError:
            return default
    return default


def normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    out = dict(snapshot)

    # keys
    if "links" in out and "edges" not in out:
        out["edges"] = out.pop("links")

    if "nodes" not in out:
        out["nodes"] = []
    if "edges" not in out:
        out["edges"] = []

    # required fields
    out["revision"] = _coerce_int(out.get("revision"), default=0)
    out["version"] = _coerce_int(out.get("version"), default=1)

    # induction_log default
    if "induction_log" not in out or out["induction_log"] is None:
        out["induction_log"] = {"entries": []}

    # minimal sanity: keep graph_id if present, otherwise derive from filename elsewhere
    if "graph_id" not in out or not isinstance(out["graph_id"], str) or not out["graph_id"].strip():
        out["graph_id"] = "unknown"

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize graph snapshot JSON to match GraphSnapshot schema")
    parser.add_argument("input", type=Path)
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    parser.add_argument("--output", type=Path, default=None, help="Write to a new path instead of overwriting")
    args = parser.parse_args()

    if args.inplace and args.output is not None:
        raise SystemExit("Use either --inplace or --output, not both")

    raw = json.loads(args.input.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("Top-level JSON must be an object")

    fixed = normalize_snapshot(raw)

    out_path = args.input if args.inplace or args.output is None else args.output
    out_path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print(f"graph_id={fixed.get('graph_id')} version={fixed.get('version')} revision={fixed.get('revision')}")
    print(f"nodes={len(fixed.get('nodes', []))} edges={len(fixed.get('edges', []))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

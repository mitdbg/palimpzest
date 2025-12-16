import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

# ISO with millis and tz offset, appears as first line for most tickets
ISO_TZ_MS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})")
# ISO without millis (just in case)
ISO_TZ_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})")
# Naive datetime
NAIVE_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def main() -> None:
    g = GraphDataset.load(GRAPH_FILE)
    tickets = [n for n in g._nodes_by_id.values() if n.type == "jira_tickets"]

    counts = Counter()
    example = {"iso_tz_ms": None, "iso_tz": None, "naive": None}

    for n in tickets:
        txt = n.text or ""
        m = ISO_TZ_MS_RE.search(txt)
        if m:
            counts["iso_tz_ms"] += 1
            example["iso_tz_ms"] = example["iso_tz_ms"] or m.group("ts")
        m = ISO_TZ_RE.search(txt)
        if m:
            counts["iso_tz"] += 1
            example["iso_tz"] = example["iso_tz"] or m.group("ts")
        m = NAIVE_RE.search(txt)
        if m:
            counts["naive"] += 1
            example["naive"] = example["naive"] or m.group("ts")

    print(f"tickets: {len(tickets)}")
    print(dict(counts))
    print("examples:")
    for k, v in example.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

import re
from pathlib import Path
from collections import Counter

from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

# Typical first-line timestamp in your sample:
# 2021-06-04T07:54:46.000+0200
CREATED_AT_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})\s*$")

# Sometimes embedded in rucio snippets inside the ticket text:
# Created at:                 2021-06-04 19:53:20
# Updated at:                 2021-06-10 18:23:59
INLINE_CREATED_AT_RE = re.compile(r"^Created at:\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)
INLINE_UPDATED_AT_RE = re.compile(r"^Updated at:\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)


def main() -> None:
    g = GraphDataset.load(GRAPH_FILE)
    tickets = [n for n in g._nodes_by_id.values() if n.type == "jira_tickets"]

    first_line_hits = 0
    inline_created_hits = 0
    inline_updated_hits = 0
    first_line_formats = Counter()

    for n in tickets:
        txt = (n.text or "").lstrip("\ufeff")
        if not txt:
            continue
        first = txt.splitlines()[0].strip()
        if CREATED_AT_RE.match(first):
            first_line_hits += 1
            first_line_formats["iso_ms_tz"] += 1

        if INLINE_CREATED_AT_RE.search(txt):
            inline_created_hits += 1
        if INLINE_UPDATED_AT_RE.search(txt):
            inline_updated_hits += 1

    print(f"tickets: {len(tickets)}")
    print(f"first-line created_at matches: {first_line_hits}")
    print(f"inline 'Created at:' matches: {inline_created_hits}")
    print(f"inline 'Updated at:' matches: {inline_updated_hits}")
    print(f"first-line formats: {dict(first_line_formats)}")


if __name__ == "__main__":
    main()

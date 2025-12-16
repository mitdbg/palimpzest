import re
from datetime import UTC, datetime
from pathlib import Path

from collections import defaultdict

from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

# Most leaf tickets have this as the first line.
FIRSTLINE_CREATED_AT_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})\s*$")

# Additional timestamps may appear inside the ticket body (e.g. embedded command output).
# We no longer use these to populate updated_at (unreliable), but we still allow
# a fallback for created_at when the first-line timestamp is missing.
ISO_TZ_MS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})")
NAIVE_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

# Sometimes embedded as labeled fields.
INLINE_CREATED_AT_RE = re.compile(r"^Created at:\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)


def _parse_firstline_created_at(txt: str) -> datetime | None:
    first = txt.splitlines()[0].strip() if txt else ""
    m = FIRSTLINE_CREATED_AT_RE.match(first)
    if not m:
        return None
    return datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(UTC)


def _parse_all_timestamps(txt: str) -> list[datetime]:
    out: list[datetime] = []
    if not txt:
        return out

    for m in ISO_TZ_MS_RE.finditer(txt):
        try:
            out.append(datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(UTC))
        except Exception:
            pass

    # Treat naive timestamps as UTC (best-effort; source system timezone is unknown).
    for m in NAIVE_RE.finditer(txt):
        try:
            out.append(datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC))
        except Exception:
            pass

    for m in INLINE_CREATED_AT_RE.finditer(txt):
        try:
            out.append(datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC))
        except Exception:
            pass

    return out


def _dt_min(vals: list[datetime]) -> datetime | None:
    return min(vals) if vals else None


def _dt_max(vals: list[datetime]) -> datetime | None:
    return max(vals) if vals else None


def main() -> None:
    g = GraphDataset.load(GRAPH_FILE)

    # Recompute `source` and `level` if they were lost in prior saves.
    # - source: derived from node.type
    # - level: derived from hierarchy edges (leaf=0, parent=1+max(child))
    for n in g._nodes_by_id.values():
        if n.type == "jira_tickets":
            n.source = "jira"
        elif n.type and n.type.startswith("git_"):
            n.source = "git"

    children_by_parent_all: dict[str, list[str]] = defaultdict(list)
    for e in g._edges_by_id.values():
        if e.type != "hierarchy:child":
            continue
        children_by_parent_all[e.src].append(e.dst)

    level_cache: dict[str, int] = {}
    visiting: set[str] = set()

    def compute_level(node_id: str) -> int:
        if node_id in level_cache:
            return level_cache[node_id]
        if node_id in visiting:
            # Should not happen (hierarchy should be acyclic); fall back.
            return 0
        visiting.add(node_id)
        child_ids = children_by_parent_all.get(node_id, [])
        if not child_ids:
            lvl = 0
        else:
            lvl = 1 + max(compute_level(c) for c in child_ids)
        visiting.remove(node_id)
        level_cache[node_id] = lvl
        return lvl

    for node_id, n in g._nodes_by_id.items():
        # Only set if missing.
        if n.level is None:
            n.level = compute_level(node_id)

    # Keep null for git_docs; also clear updated_at everywhere (unreliable).
    for n in g._nodes_by_id.values():
        if n.type == "git_docs":
            n.created_at = None
            n.updated_at = None
        else:
            n.updated_at = None

    jira_nodes = {n.id: n for n in g._nodes_by_id.values() if n.type == "jira_tickets"}

    # Build parent->children from hierarchy edges among jira nodes.
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for e in g._edges_by_id.values():
        if e.type != "hierarchy:child":
            continue
        if e.src in jira_nodes and e.dst in jira_nodes:
            children_by_parent[e.src].append(e.dst)

    # 1) Leaf tickets: parse from text.
    leaf_updated_created = 0
    leaf_count = 0

    for n in jira_nodes.values():
        if n.level != 0:
            continue
        leaf_count += 1
        txt = (n.text or "").lstrip("\ufeff")
        if not txt:
            continue

        firstline_created = _parse_firstline_created_at(txt)
        all_ts = _parse_all_timestamps(txt)

        created = firstline_created or _dt_min(all_ts)

        if created is not None:
            n.created_at = created
            leaf_updated_created += 1

    # 2) Summary nodes: derive from children bottom-up by level.
    max_level = max((n.level for n in jira_nodes.values() if n.level is not None), default=0)
    summary_updated = 0

    for level in range(1, max_level + 1):
        for n in jira_nodes.values():
            if n.level != level:
                continue

            child_ids = children_by_parent.get(n.id, [])
            if not child_ids:
                continue

            child_created = [jira_nodes[c].created_at for c in child_ids if jira_nodes[c].created_at is not None]
            created = _dt_min(child_created)

            if created is not None:
                n.created_at = created
            if created is not None:
                summary_updated += 1

    g.save(GRAPH_FILE)
    print(f"Leaf jira_tickets (level=0): {leaf_count}")
    print(f"Backfilled created_at for {leaf_updated_created} leaf tickets")
    print("Cleared updated_at for all nodes (unreliable)")
    print(f"Derived created_at for {summary_updated} summary jira_tickets nodes")
    print(f"Saved {GRAPH_FILE}")


if __name__ == "__main__":
    main()

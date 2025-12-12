from __future__ import annotations

from pydantic import BaseModel, Field

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode
from palimpzest.core.data.iter_dataset import MemoryDataset


def test_traverse_operator_beam_search_and_filter() -> None:
    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c", "d"]:
        g.add_node(GraphNode(id=node_id))

    # a -> b, a -> c, b -> d, c -> d
    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_ac", src="a", dst="c", type="rel"))
    g.add_edge(GraphEdge(id="e_bd", src="b", dst="d", type="rel"))
    g.add_edge(GraphEdge(id="e_cd", src="c", dst="d", type="rel"))

    # Ranker prefers d > c > b > a
    scores = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.9}

    def ranker(node_id, node, edge, from_node_id, path_node_ids, path_edge_ids):
        return scores[node_id]

    # Filter out node 'b' at visit-time
    def visit_filter(node_id, node, depth, score, path_node_ids, path_edge_ids):
        return node_id != "b"

    ds = g.traverse(
        start_node_ids=["a"],
        max_steps=10,
        edge_type="rel",
        ranker=ranker,
        ranker_id="toy",
        visit_filter=visit_filter,
        visit_filter_id="no-b",
    )

    out = ds.run()
    node_ids = [r.node_id for r in out]

    # 'a' should appear, 'b' should not
    assert "a" in node_ids
    assert "b" not in node_ids

    # 'c' should be reached, and then 'd'
    assert "c" in node_ids
    assert "d" in node_ids


def test_traverse_operator_respects_edge_type_filter() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))
    g.add_node(GraphNode(id="c"))

    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e2", src="a", dst="c", type="other"))

    ds = g.traverse(start_node_ids=["a"], edge_type="rel", max_steps=10)
    out = ds.run()

    node_ids = [r.node_id for r in out]
    assert "b" in node_ids
    assert "c" not in node_ids


def test_traverse_operator_runs_node_program_per_visit() -> None:
    class NodeProgramOut(BaseModel):
        label: str = Field(description="Per-node label")

    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c"]:
        g.add_node(GraphNode(id=node_id))

    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_ac", src="a", dst="c", type="rel"))

    scores = {"a": 0.1, "b": 0.9, "c": 0.2}

    def ranker(node_id, node, edge, from_node_id, path_node_ids, path_edge_ids):
        return scores[node_id]

    def node_program(*, node_id, node, graph, depth, score, path_node_ids, path_edge_ids):
        # Simple non-LLM Palimpzest plan; returns exactly one record.
        return MemoryDataset(
            id=f"node-program-{node_id}",
            vals=[{"label": f"L-{node_id}"}],
            schema=NodeProgramOut,
        )

    ds = g.traverse(
        start_node_ids=["a"],
        edge_type="rel",
        max_steps=10,
        ranker=ranker,
        ranker_id="toy",
        node_program=node_program,
        node_program_id="labeler",
        node_program_output_schema=NodeProgramOut,
    )

    out = ds.run()
    rows = [(r.node_id, r.label) for r in out]

    # a is visited, plus (b,c) from frontier. Each visit emits one subprogram row.
    assert ("a", "L-a") in rows
    assert ("b", "L-b") in rows
    assert ("c", "L-c") in rows


def test_traverse_operator_admittance_gates_outputs() -> None:
    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c"]:
        g.add_node(GraphNode(id=node_id))
    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_ac", src="a", dst="c", type="rel"))

    def admittance(node_id, node, depth, score, path_node_ids, path_edge_ids):
        return node_id != "c"

    ds = g.traverse(
        start_node_ids=["a"],
        edge_type="rel",
        max_steps=10,
        admittance=admittance,
        admittance_id="no-c",
    )
    out = ds.run()
    node_ids = [r.node_id for r in out]
    assert "a" in node_ids
    assert "b" in node_ids
    assert "c" not in node_ids


def test_traverse_operator_termination_stops_early() -> None:
    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c", "d"]:
        g.add_node(GraphNode(id=node_id))
    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_bc", src="b", dst="c", type="rel"))
    g.add_edge(GraphEdge(id="e_cd", src="c", dst="d", type="rel"))

    def termination(state: dict) -> bool:
        # stop after admitting 2 nodes
        return state["admitted_nodes"] >= 2

    ds = g.traverse(
        start_node_ids=["a"],
        edge_type="rel",
        max_steps=10,
        termination=termination,
        termination_id="two-nodes",
    )
    out = ds.run()
    node_ids = [r.node_id for r in out]
    assert "a" in node_ids
    assert "b" in node_ids
    assert "d" not in node_ids


def test_traverse_operator_tracer_emits_step_events() -> None:
    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c"]:
        g.add_node(GraphNode(id=node_id, text=f"text-{node_id}"))
    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_ac", src="a", dst="c", type="rel"))

    trace: list[dict] = []

    def tracer(ev: dict) -> None:
        trace.append(ev)

    ds = g.traverse(
        start_node_ids=["a"],
        edge_type="rel",
        max_steps=5,
        tracer=tracer,
        tracer_id="test",
    )
    out = ds.run()
    assert len(out) > 0

    # Basic lifecycle events
    assert any(e.get("event_type") == "traverse_init" for e in trace)
    assert any(e.get("event_type") == "traverse_summary" for e in trace)
    assert any(e.get("event_type") == "traverse_done" for e in trace)

    # Per-step events include step number and node id
    step_begins = [e for e in trace if e.get("event_type") == "step_begin"]
    assert step_begins
    assert all("step" in e and "popped" in e for e in step_begins)

    expands = [e for e in trace if e.get("event_type") == "step_expand"]
    assert expands
    assert isinstance(expands[0].get("neighbors"), list)

    summary = next(e for e in trace if e.get("event_type") == "traverse_summary")
    assert "top_nodes" in summary
    assert "top_edges" in summary
    assert isinstance(summary.get("skip_counts"), dict)
    assert isinstance(summary.get("ts_ms"), int)
    assert isinstance(summary.get("trace_seq"), int)


def test_traverse_operator_tracer_includes_neighbor_scores_when_ranker_present() -> None:
    g = GraphDataset(graph_id="g")
    for node_id in ["a", "b", "c"]:
        g.add_node(GraphNode(id=node_id, text=f"text-{node_id}"))
    g.add_edge(GraphEdge(id="e_ab", src="a", dst="b", type="rel"))
    g.add_edge(GraphEdge(id="e_ac", src="a", dst="c", type="rel"))

    def ranker(node_id, node, edge, from_node_id, path_node_ids, path_edge_ids):
        return {"a": 0.0, "b": 2.0, "c": 1.0}[node_id]

    trace: list[dict] = []

    def tracer(ev: dict) -> None:
        trace.append(ev)

    ds = g.traverse(
        start_node_ids=["a"],
        edge_type="rel",
        max_steps=2,
        ranker=ranker,
        ranker_id="toy",
        tracer=tracer,
        tracer_id="test",
    )
    out = ds.run()
    assert len(out) > 0

    expands = [e for e in trace if e.get("event_type") == "step_expand"]
    assert expands
    neighbors = expands[0].get("neighbors")
    assert isinstance(neighbors, list)
    enqueued = [n for n in neighbors if n.get("enqueued") is True]
    assert enqueued
    assert all("score" in n for n in enqueued)

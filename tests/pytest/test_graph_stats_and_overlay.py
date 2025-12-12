from __future__ import annotations

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode
from palimpzest.core.models import ExecutionStats, OperatorStats, PlanStats, RecordOpStats, compute_graph_stats
from palimpzest.query.processor.graph_overlay import add_shortcut_overlay_edges_from_traversals


def test_compute_graph_stats_from_record_op_stats() -> None:
    es = ExecutionStats(execution_id="x")
    ps = PlanStats(plan_id="p", plan_str="p")

    traverse_stats = OperatorStats(full_op_id="f1", op_name="TraverseOp")
    traverse_stats.record_op_stats_lst.append(
        RecordOpStats(
            record_id="r1",
            record_parent_ids=None,
            record_source_indices=[],
            record_state={"node_id": "b", "path_edge_ids": ["e1"]},
            full_op_id="f1",
            logical_op_id="l1",
            op_name="TraverseOp",
            time_per_record=0.0,
            cost_per_record=0.0,
            op_details={"graph_id": "g"},
        )
    )

    induce_stats = OperatorStats(full_op_id="f2", op_name="InduceEdgesOp")
    induce_stats.record_op_stats_lst.append(
        RecordOpStats(
            record_id="r2",
            record_parent_ids=None,
            record_source_indices=[],
            record_state={"edge_type": "rel", "created": True, "existed": False},
            full_op_id="f2",
            logical_op_id="l2",
            op_name="InduceEdgesOp",
            time_per_record=0.0,
            cost_per_record=0.0,
            op_details={"graph_id": "g"},
        )
    )

    ps.operator_stats = {"l1": {"f1": traverse_stats}, "l2": {"f2": induce_stats}}
    es.plan_stats = {"p": ps}

    stats = compute_graph_stats(es)
    assert stats["g"]["traverse"]["node_visits"]["b"] == 1
    assert stats["g"]["traverse"]["edge_traversals"]["e1"] == 1
    assert stats["g"]["induce"]["edges_created"] == 1
    assert stats["g"]["induce"]["by_edge_type"]["rel"]["created"] == 1


def test_overlay_shortcut_builder_adds_edges() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))
    g.add_node(GraphNode(id="c"))
    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))
    g.add_edge(GraphEdge(id="e2", src="b", dst="c", type="links_to"))

    traversal_states = [
        {"path_node_ids": ["a", "b", "c"], "path_edge_ids": ["e1", "e2"]},
        {"path_node_ids": ["a", "b", "c"], "path_edge_ids": ["e1", "e2"]},
    ]
    added = add_shortcut_overlay_edges_from_traversals(
        graph=g,
        traversal_record_states=traversal_states,
        edge_type="overlay:shortcut",
        min_hops=2,
        min_count=2,
    )
    assert added == 1
    assert any(e.type == "overlay:shortcut" and e.src == "a" and e.dst == "c" for e in g.to_snapshot().edges)

from __future__ import annotations

from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode


def test_induce_operator_adds_edge_when_predicate_true() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))

    def pred(src_id, src, dst_id, dst, graph):
        return True

    rev0 = g.revision
    ds = g.induce_edges(
        candidate_pairs=[("a", "b")],
        edge_type="rel",
        predicate=pred,
        predicate_id="always-true",
    )
    out = ds.run()

    assert len(out) == 1
    r = list(out)[0]
    assert r.created is True
    assert r.edge_type == "rel"
    assert r.edge_id is not None
    assert g.has_edge(r.edge_id)
    assert g.revision == rev0 + 1


def test_induce_operator_does_not_add_edge_when_predicate_false() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))

    def pred(src_id, src, dst_id, dst, graph):
        return False

    rev0 = g.revision
    ds = g.induce_edges(
        candidate_pairs=[("a", "b")],
        edge_type="rel",
        predicate=pred,
        predicate_id="always-false",
    )
    out = ds.run()

    assert len(out) == 1
    r = list(out)[0]
    assert r.created is False
    assert r.existed is False
    assert g.revision == rev0


def test_induce_operator_existing_edge_no_overwrite_does_not_mutate() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))

    # First run creates the edge.
    out1 = g.induce_edges(candidate_pairs=[("a", "b")], edge_type="rel").run()
    assert len(out1) == 1
    edge_id = list(out1)[0].edge_id
    assert edge_id is not None
    assert g.has_edge(edge_id)

    rev1 = g.revision

    # Second run should notice it exists and not mutate.
    out2 = g.induce_edges(candidate_pairs=[("a", "b")], edge_type="rel", overwrite=False).run()
    assert len(out2) == 1
    r2 = list(out2)[0]
    assert r2.existed is True
    assert r2.created is False
    assert g.revision == rev1


def test_induce_operator_float_score_threshold() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))

    def pred(src_id, src, dst_id, dst, graph):
        return 0.49

    out = g.induce_edges(
        candidate_pairs=[("a", "b")],
        edge_type="rel",
        predicate=pred,
        predicate_id="score",
        threshold=0.5,
    ).run()

    assert len(out) == 1
    r = list(out)[0]
    assert r.created is False
    assert r.score == 0.49

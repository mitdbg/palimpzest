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


def test_knn_similarity_induction_full_and_incremental() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a", embedding=[1.0, 0.0]))
    g.add_node(GraphNode(id="b", embedding=[0.9, 0.1]))
    g.add_node(GraphNode(id="c", embedding=[0.0, 1.0]))

    out_full = g.run_knn_similarity_induction(edge_type="sim:knn", k=1, threshold=None, mode="full")
    assert len(out_full) > 0
    edges0 = [e for e in g.to_snapshot().edges if e.type == "sim:knn"]
    assert all(e.src != e.dst for e in edges0)

    # Re-run incremental with no new nodes: should be a no-op.
    rev0 = g.revision
    out_noop = g.reapply_inductions_incremental()
    assert len(out_noop) == 1
    assert g.revision == rev0

    # Add a new node; incremental should add new edges involving it.
    g.add_node(GraphNode(id="d", embedding=[0.95, 0.05]))
    rev1 = g.revision
    out_inc = g.reapply_inductions_incremental()
    assert len(out_inc) == 1
    assert g.revision > rev1

    edges1 = [e for e in g.to_snapshot().edges if e.type == "sim:knn"]
    assert any(e.src == "d" or e.dst == "d" for e in edges1)


def test_knn_similarity_induction_requires_topk_xor_threshold() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a", embedding=[1.0, 0.0]))
    g.add_node(GraphNode(id="b", embedding=[0.9, 0.1]))

    try:
        g.run_knn_similarity_induction(k=None, threshold=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    try:
        g.run_knn_similarity_induction(k=1, threshold=0.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    out = g.run_knn_similarity_induction(k=None, threshold=0.8, mode="full")
    assert len(out) >= 0

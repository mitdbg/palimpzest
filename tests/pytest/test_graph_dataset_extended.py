from __future__ import annotations

import pytest
from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode, GraphEdge

@pytest.fixture
def graph():
    return GraphDataset(graph_id="test_graph_extended")

def test_upsert_node_triggers_induction(graph):
    g = graph
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    
    # Register induction: connect if val=1
    # Wait, attr_equals checks if src.val == dst.val.
    # It does NOT check if val == 1.
    # The test comment says "connect if val=1".
    # But the predicate is `attr_equals`.
    # So it connects a->a because a.val == a.val.
    # Why is it failing?
    # Maybe allow_self_edges is not being passed correctly?
    
    spec_id = g.add_predicate_induction(
        edge_type="val_eq",
        generator_kind="all_pairs",
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}],
        symmetric=False,
        allow_self_edges=True
    )
    
    # Initial run
    edges = g.run_induction(spec_id, mode="full")
    assert len(edges) == 1 # a->a
    
    # Update node "a" to val=2
    # upsert_node should bump revision
    g.upsert_node(GraphNode(id="a", attrs={"val": 2}))
    
    # Incremental run
    # Should re-process "a".
    # Since val=2, it should NOT generate edge (unless we had another node with val=2)
    # But wait, the old edge a->a (val=1) exists.
    # Does induction remove old edges? No.
    # Does it add new ones? Yes.
    # If we change val to 2, and run induction, it checks a->a. val=2 == val=2.
    # So it generates a->a again.
    # InduceEdges checks if existed. If existed and overwrite=False, it does nothing.
    # So we expect 1 edge (the existing one).
    
    edges = g.run_induction(spec_id, mode="incremental")
    # It returns edges that were "processed".
    # InduceEdges returns a record with created=False, existed=True.
    # run_induction returns the dataset output.
    # So we should see the edge in the output.
    assert len(edges) == 1
    assert edges.data_records[0].existed is True
    assert edges.data_records[0].created is False

def test_inplace_modification_does_not_trigger_induction(graph):
    g = graph
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    
    spec_id = g.add_predicate_induction(
        edge_type="val_eq",
        generator_kind="all_pairs",
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}],
        symmetric=False,
        allow_self_edges=True
    )
    
    g.run_induction(spec_id, mode="full")
    
    # Modify in place
    node = g.get_node("a")
    node.attrs["val"] = 2
    
    # Incremental run
    # Should NOT see "a" as impacted because revision didn't bump.
    edges = g.run_induction(spec_id, mode="incremental")
    assert len(edges) == 0

def test_duplicate_edge_handling(graph):
    g = graph
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    g.add_node(GraphNode(id="b", attrs={"val": 1}))
    
    spec_id = g.add_predicate_induction(
        edge_type="link",
        generator_kind="all_pairs",
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}],
        symmetric=False
    )
    
    # Run once
    edges1 = g.run_induction(spec_id, mode="full")
    assert len(edges1) == 2 # a->b, b->a
    assert all(e.created for e in edges1)
    
    # Run again (full)
    edges2 = g.run_induction(spec_id, mode="full")
    assert len(edges2) == 2
    assert all(e.existed for e in edges2)
    assert all(not e.created for e in edges2)

def test_persistence_of_revisions(tmp_path):
    g = GraphDataset(graph_id="persist")
    g.add_node(GraphNode(id="a"))
    rev_a = g.get_node("a").revision
    
    spec_id = g.add_knn_similarity_topk(k=1, edge_type="sim")
    # Need embedding for KNN
    g.get_node("a").embedding = [1.0]
    g.run_induction(spec_id, mode="full")
    
    last_rev = g.induction_log().get(spec_id).last_revision
    assert last_rev is not None
    
    # Save
    path = tmp_path / "graph.json"
    g.save(path)
    
    # Load
    g2 = GraphDataset.load(path)
    assert g2.revision == g.revision
    assert g2.get_node("a").revision == rev_a
    assert g2.induction_log().get(spec_id).last_revision == last_rev

def test_incremental_mode_text_anchor(graph):
    g = graph
    g.add_node(GraphNode(id="a", text="hello world", attrs={"name": "A"}))
    
    # Inducer: text contains name
    spec_id = g.add_predicate_induction(
        edge_type="mentions",
        generator_kind="text_anchor",
        generator_params={
            "source_text_field": "text",
            "target_fields": ["attrs.name"],
            "min_anchor_len": 1 # Allow short anchors for test
        },
        predicates=[{"kind": "text_contains", "params": {"source_field": "text", "target_fields": ["attrs.name"], "boundaries": False}}],
        symmetric=False,
        incremental_mode="bidirectional"
    )
    
    g.run_induction(spec_id, mode="full")
    
    # Add node B with name "world" (A contains "world")
    g.add_node(GraphNode(id="b", attrs={"name": "world"}))
    
    # Incremental run
    # Should find A -> B (Old -> New)
    # text_anchor generator should find "world" in A's text and link to B.
    edges = g.run_induction(spec_id, mode="incremental")
    
    pairs = sorted([(e.src_node_id, e.dst_node_id) for e in edges])
    assert ("a", "b") in pairs

def test_reapply_inductions_multiple(graph):
    g = graph
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    
    # Inducer 1: val=1
    spec1 = g.add_predicate_induction(
        edge_type="t1",
        generator_kind="all_pairs",
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}],
        allow_self_edges=True
    )
    
    # Inducer 2: val=1 (different type)
    spec2 = g.add_predicate_induction(
        edge_type="t2",
        generator_kind="all_pairs",
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}],
        allow_self_edges=True
    )
    
    # Initial run
    g.reapply_inductions_incremental()
    
    # Add node b
    g.add_node(GraphNode(id="b", attrs={"val": 1}))
    
    # Reapply
    results = g.reapply_inductions_incremental()
    assert len(results) == 2
    
    # Check edges
    # Should have b->b for both types (and a->b, b->a if symmetric/bidirectional logic applied, but here symmetric=False default)
    # Wait, add_predicate_induction defaults symmetric=False.
    # incremental_mode defaults to "source".
    # So New->All.
    # b->a, b->b.
    
    edges1 = results[0]
    edges2 = results[1]
    
    pairs1 = sorted([(e.src_node_id, e.dst_node_id) for e in edges1])
    pairs2 = sorted([(e.src_node_id, e.dst_node_id) for e in edges2])
    
    assert ("b", "a") in pairs1
    assert ("b", "b") in pairs1
    assert ("b", "a") in pairs2
    assert ("b", "b") in pairs2

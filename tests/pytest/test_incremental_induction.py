from __future__ import annotations

import pytest
from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode

def test_incremental_induction_with_revision() -> None:
    g = GraphDataset(graph_id="g1")
    
    # Add initial nodes
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    g.add_node(GraphNode(id="b", attrs={"val": 2}))
    
    assert g.revision == 2
    assert g.get_node("a").revision == 1
    assert g.get_node("b").revision == 2
    
    # Register induction (e.g. attr_bucket for simplicity, or just check impacted logic)
    # We can test _impacted_node_ids directly to avoid setting up full induction machinery
    
    # Mock induction spec
    from palimpzest.core.data.induction import InductionSpec
    spec = InductionSpec(
        edge_type="test",
        generator={"kind": "all_pairs", "params": {}},
        decider={"kind": "predicate", "params": {"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}}
    )
    spec_id = g.add_induction(spec)
    
    # Initially, all nodes should be impacted
    impacted = g._impacted_node_ids(spec_id=spec_id, mode="incremental")
    assert impacted == {"a", "b"}
    
    # Run induction (simulated)
    # We manually update the log to simulate a run
    from palimpzest.core.data.induction import InductionLogEntry
    g._induction_log.upsert(InductionLogEntry(
        spec=spec,
        processed_node_ids=["a", "b"],
        last_revision=g.revision
    ))
    
    # Now, no nodes should be impacted
    impacted = g._impacted_node_ids(spec_id=spec_id, mode="incremental")
    assert impacted == set()
    
    # Add a new node
    g.add_node(GraphNode(id="c", attrs={"val": 3}))
    assert g.revision == 3
    assert g.get_node("c").revision == 3
    
    # Now "c" should be impacted
    impacted = g._impacted_node_ids(spec_id=spec_id, mode="incremental")
    assert impacted == {"c"}
    
    # Update log again
    g._induction_log.upsert(InductionLogEntry(
        spec=spec,
        processed_node_ids=["a", "b", "c"],
        last_revision=g.revision
    ))
    
    # No nodes impacted
    impacted = g._impacted_node_ids(spec_id=spec_id, mode="incremental")
    assert impacted == set()

def test_incremental_induction_fallback() -> None:
    """Test fallback to processed_node_ids if last_revision is None (legacy case)."""
    g = GraphDataset(graph_id="g1")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))
    
    from palimpzest.core.data.induction import InductionSpec, InductionLogEntry
    spec = InductionSpec(
        edge_type="test",
        generator={"kind": "all_pairs", "params": {}},
        decider={"kind": "predicate", "params": {"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}}
    )
    spec_id = g.add_induction(spec)
    
    # Simulate legacy log entry (no last_revision)
    g._induction_log.upsert(InductionLogEntry(
        spec=spec,
        processed_node_ids=["a"],
        last_revision=None
    ))
    
    # "b" should be impacted because it's not in processed_node_ids
    impacted = g._impacted_node_ids(spec_id=spec_id, mode="incremental")
    assert impacted == {"b"}

def test_incremental_induction_bidirectional() -> None:
    """Test that bidirectional incremental mode checks Old -> New."""
    g = GraphDataset(graph_id="g1")
    
    # Add initial node "a"
    g.add_node(GraphNode(id="a", attrs={"val": 1}))
    
    # Register bidirectional induction
    # We use a predicate that checks if src.val == dst.val
    from palimpzest.core.data.induction import InductionSpec
    spec = InductionSpec(
        edge_type="test",
        symmetric=False,
        incremental_mode="bidirectional",
        generator={"kind": "all_pairs", "params": {}},
        decider={"kind": "predicate", "params": {"kind": "attr_equals", "params": {"src_attr": "attrs.val"}}}
    )
    spec_id = g.add_induction(spec)
    
    # Run initial induction (a -> a)
    # Since allow_self_edges is False by default, no edges.
    g.run_induction(spec_id, mode="incremental")
    
    # Add new node "b" with same val
    g.add_node(GraphNode(id="b", attrs={"val": 1}))
    
    # Run incremental induction
    # Should find a -> b (Old -> New) and b -> a (New -> Old)
    # Because it's bidirectional and symmetric=False (so we get directed edges)
    edges = g.run_induction(spec_id, mode="incremental")
    
    edge_pairs = sorted([(e.src_node_id, e.dst_node_id) for e in edges])
    assert edge_pairs == [("a", "b"), ("b", "a")]
    
    # Verify log updated
    entry = g._induction_log.get(spec_id)
    assert entry.last_revision == g.revision
    
    # Add node "c" with different val
    g.add_node(GraphNode(id="c", attrs={"val": 2}))
    edges = g.run_induction(spec_id, mode="incremental")
    assert len(edges) == 0


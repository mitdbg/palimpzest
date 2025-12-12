from __future__ import annotations

import json

import pytest

from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode


def test_graphdataset_add_get_and_neighbors() -> None:
    g = GraphDataset(graph_id="g1")

    g.add_node(GraphNode(id="a", type="entity"))
    g.add_node(GraphNode(id="b", type="entity"))

    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))

    assert g.has_node("a")
    assert g.has_edge("e1")

    assert g.get_node("a").id == "a"
    assert g.get_edge("e1").src == "a"

    out_edges = list(g.iter_out_edges("a"))
    assert [e.id for e in out_edges] == ["e1"]

    neighbors = list(g.iter_neighbors("a"))
    assert len(neighbors) == 1
    edge, node = neighbors[0]
    assert edge.id == "e1"
    assert node.id == "b"


def test_graphdataset_invariants() -> None:
    g = GraphDataset(graph_id="g1")
    g.add_node(GraphNode(id="a"))

    with pytest.raises(ValueError, match="does not exist"):
        g.add_edge(GraphEdge(id="e1", src="a", dst="missing", type="links_to"))

    g.add_node(GraphNode(id="b"))
    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))

    with pytest.raises(ValueError, match="already exists"):
        g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))


def test_graphdataset_revision_is_monotonic() -> None:
    g = GraphDataset(graph_id="g1")
    assert g.revision == 0

    g.add_node(GraphNode(id="a"))
    assert g.revision == 1

    g.add_node(GraphNode(id="b"))
    assert g.revision == 2

    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))
    assert g.revision == 3

    g.remove_edge("e1")
    assert g.revision == 4


def test_graphdataset_remove_node_cascade_bumps_once() -> None:
    g = GraphDataset(graph_id="g1")
    g.add_node(GraphNode(id="a"))
    g.add_node(GraphNode(id="b"))
    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to"))

    before = g.revision
    g.remove_node("a", cascade=True)
    assert g.revision == before + 1

    assert not g.has_node("a")
    assert not g.has_edge("e1")


def test_graphdataset_save_load_roundtrip(tmp_path) -> None:
    g = GraphDataset(graph_id="g1", name="test")
    g.add_node(GraphNode(id="a", attrs={"k": "v"}))
    g.add_node(GraphNode(id="b"))
    g.add_edge(GraphEdge(id="e1", src="a", dst="b", type="links_to", attrs={"w": 0.1}))

    path = tmp_path / "graph.json"
    g.save(path)

    # file is valid json
    payload = json.loads(path.read_text())
    assert payload["graph_id"] == "g1"
    assert "induction_log" in payload

    g2 = GraphDataset.load(path)
    assert g2.graph_id == "g1"
    assert set(n.id for n in g2.to_snapshot().nodes) == {"a", "b"}
    assert set(e.id for e in g2.to_snapshot().edges) == {"e1"}

    # adjacency rebuilt
    assert [e.id for e in g2.iter_out_edges("a")] == ["e1"]

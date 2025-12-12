from __future__ import annotations

from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode


def test_predicate_text_contains_with_text_anchor_generator() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="t1", label="CMSTZ-895"))
    g.add_node(GraphNode(id="t2", label="CMSTRANSF-260"))
    g.add_node(GraphNode(id="src", label="SRC", text="See CMSTZ-895 and also CMSTRANSF-260. Duplicate CMSTZ-895."))

    spec_id = g.add_predicate_induction(
        edge_type="ref:predicate",
        generator_kind="text_anchor",
        generator_params={"source_text_field": "text", "target_fields": ["label"], "min_anchor_len": 5},
        predicates=[
            {"kind": "text_contains", "params": {"source_field": "text", "target_fields": ["label"], "boundaries": True}}
        ],
        predicate_mode="all",
        symmetric=False,
    )
    out = g.run_induction(spec_id, mode="full")

    assert any(r.src_node_id == "src" and r.dst_node_id == "t1" and r.created is True for r in out)
    assert any(r.src_node_id == "src" and r.dst_node_id == "t2" and r.created is True for r in out)


def test_predicate_regex_match() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="t1", label="CMSTZ-895"))
    g.add_node(GraphNode(id="src", label="SRC", text="Reference: CMSTZ-895 (see above)"))

    spec_id = g.add_predicate_induction(
        edge_type="ref:regex",
        generator_kind="all_pairs",
        generator_params={},
        predicates=[
            {
                "kind": "regex_match",
                "params": {
                    "source_field": "text",
                    "patterns": [r"\b[A-Z][A-Z0-9]{1,9}-\d+\b"],
                    "target_fields": ["label"],
                },
            }
        ],
        predicate_mode="all",
        symmetric=False,
    )
    out = g.run_induction(spec_id, mode="full")
    assert any(r.src_node_id == "src" and r.dst_node_id == "t1" and r.created is True for r in out)


def test_predicate_attr_equals() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="a", label="A", attrs={"team": "x"}))
    g.add_node(GraphNode(id="b", label="B", attrs={"team": "x"}))
    g.add_node(GraphNode(id="c", label="C", attrs={"team": "y"}))

    spec_id = g.add_predicate_induction(
        edge_type="rel:same_team",
        generator_kind="attr_bucket",
        generator_params={"attr_path": "attrs.team"},
        predicates=[{"kind": "attr_equals", "params": {"src_attr": "attrs.team", "dst_attr": "attrs.team"}}],
        predicate_mode="all",
        symmetric=False,
    )
    out = g.run_induction(spec_id, mode="full")
    assert any(r.src_node_id == "a" and r.dst_node_id == "b" and r.created is True for r in out)
    assert not any(r.src_node_id == "a" and r.dst_node_id == "c" and r.created is True for r in out)


def test_predicate_compound_all() -> None:
    g = GraphDataset(graph_id="g")
    g.add_node(GraphNode(id="t1", label="CMSTZ-895", attrs={"team": "x"}))
    g.add_node(GraphNode(id="t2", label="CMSTZ-999", attrs={"team": "y"}))
    g.add_node(GraphNode(id="src", label="SRC", attrs={"team": "x"}, text="See CMSTZ-895 and CMSTZ-999"))

    spec_id = g.add_predicate_induction(
        edge_type="ref:filtered",
        generator_kind="text_anchor",
        generator_params={"source_text_field": "text", "target_fields": ["label"], "min_anchor_len": 5},
        predicates=[
            {"kind": "text_contains", "params": {"source_field": "text", "target_fields": ["label"], "boundaries": True}},
            {"kind": "attr_equals", "params": {"src_attr": "attrs.team", "dst_attr": "attrs.team"}},
        ],
        predicate_mode="all",
        symmetric=False,
    )
    out = g.run_induction(spec_id, mode="full")
    assert any(r.src_node_id == "src" and r.dst_node_id == "t1" and r.created is True for r in out)
    assert not any(r.src_node_id == "src" and r.dst_node_id == "t2" and r.created is True for r in out)

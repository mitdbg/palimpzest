import json
from pathlib import Path

from fastapi.testclient import TestClient

from palimpzest.core.data.graph_dataset import GraphEdge, GraphNode, GraphSnapshot
from palimpzest.server.graphrag_app import create_app


def _write_snapshot(path: Path) -> None:
    snap = GraphSnapshot(
        graph_id="g",
        revision=1,
        nodes=[
            GraphNode(id="A", label="Alpha", attrs={"level": 3}),
            GraphNode(id="B", label="Beta"),
            GraphNode(id="C", label="Gamma"),
        ],
        edges=[
            GraphEdge(id="e1", src="A", dst="B", type="hierarchy:child"),
            GraphEdge(id="e2", src="B", dst="C", type="hierarchy:child"),
        ],
    )
    path.write_text(json.dumps(snap.model_dump(mode="json")))


def test_api_graph_and_run_emits_frontier_updates(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snap.json"
    _write_snapshot(snapshot_path)

    app = create_app(snapshot_path=snapshot_path)
    client = TestClient(app)

    g = client.get("/api/graph")
    assert g.status_code == 200
    payload = g.json()
    assert {n["id"] for n in payload["nodes"]} == {"A", "B", "C"}
    assert len(payload["links"]) == 2

    r = client.post(
        "/api/run",
        json={
            "index": "cms_standard",
            "query": "A",
            "entry_points": 1,
            "max_steps": 5,
                "edge_type": "all",
        },
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    events = []
    with client.websocket_connect(f"/ws/{run_id}") as ws:
        while True:
            try:
                events.append(json.loads(ws.receive_text()))
            except Exception:
                break

    event_types = [e.get("event_type") for e in events]
    assert "trace_init" in event_types
    assert "query_start" in event_types
    assert "search_step" in event_types
    assert "frontier_update" in event_types
    assert "result" in event_types
    assert "query_end" in event_types


def test_run_with_admittance_model_and_missing_litellm_fails_open(tmp_path: Path, monkeypatch) -> None:
    """Regression: previously an admittance_model + missing LiteLLM/provider config
    would cause every admit decision to error -> False, pruning traversal after 1 node.
    """

    # Force the LLM decider to hit a LiteLLM error path deterministically.
    import palimpzest.graphrag.deciders as deciders

    monkeypatch.setattr(deciders, "litellm", None)

    snapshot_path = tmp_path / "snap.json"
    _write_snapshot(snapshot_path)

    app = create_app(snapshot_path=snapshot_path)
    client = TestClient(app)

    r = client.post(
        "/api/run",
        json={
            "index": "cms_standard",
            "query": "A",
            "entry_points": 1,
            "max_steps": 5,
            "edge_type": "all",
            # Turn off reranking so the only gating comes from admittance_model.
            "ranking_model": "off",
            # Explicitly enable admittance even though LiteLLM is missing.
            "admittance_model": "openrouter/x-ai/grok-4.1-fast",
            # Disable termination to keep traversal behavior simple.
            "termination_model": "off",
        },
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    events: list[dict] = []
    with client.websocket_connect(f"/ws/{run_id}") as ws:
        while True:
            try:
                events.append(json.loads(ws.receive_text()))
            except Exception:
                break

    # Ensure we got more than one search step (A plus at least one neighbor).
    search_steps = [e for e in events if e.get("event_type") == "search_step"]
    assert len(search_steps) >= 2

    # And that at least one step shows a fail-open reason.
    assert any((s.get("data") or {}).get("reason", "").startswith("litellm_error") for s in search_steps)


def test_api_resources_and_load_graph_allow_data_and_exports(tmp_path: Path, monkeypatch) -> None:
    # Isolate filesystem assumptions by running the app from a temporary cwd.
    monkeypatch.chdir(tmp_path)

    exports_dir = tmp_path / "CURRENT_WORKSTREAM/exports"
    data_dir = tmp_path / "data"
    exports_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    export_snap = exports_dir / "cms_standard_graph_snapshot.json"
    data_snap = data_dir / "cms_v1_graph.json"
    _write_snapshot(export_snap)
    _write_snapshot(data_snap)

    app = create_app(snapshot_path=export_snap)
    client = TestClient(app)

    r = client.get("/api/resources")
    assert r.status_code == 200
    payload = r.json()
    assert "CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json" in payload["indices"]
    assert "data/cms_v1_graph.json" in payload["indices"]

    # Can load a graph under data/
    lg = client.post("/api/load_graph", json={"index": "data/cms_v1_graph.json"})
    assert lg.status_code == 200
    assert lg.json()["ok"] is True

    # Directory traversal is rejected.
    bad = client.post("/api/load_graph", json={"index": "../secrets.json"})
    assert bad.status_code == 400

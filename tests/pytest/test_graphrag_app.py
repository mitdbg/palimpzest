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

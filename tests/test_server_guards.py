import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scout.warehouse import server
from scout.warehouse.models import get_spec


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def test_open_when_no_token_configured(client, monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    assert client.get("/").status_code == 200
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_token_required_for_page_and_socket(client, monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "s3cret")
    assert client.get("/").status_code == 401
    assert client.get("/?token=wrong").status_code == 401
    assert client.get("/?token=s3cret").status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?token=wrong"):
            pass
    with client.websocket_connect("/ws?token=s3cret") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_health_needs_no_token(client, monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "s3cret")
    assert client.get("/health").json() == {"ok": True}


def test_run_budget_is_clamped_to_server_ceiling(monkeypatch):
    monkeypatch.setenv("MAX_RUN_BUDGET_USD", "0.25")
    assert server.run_budget(5.0) == 0.25      # the UI cannot raise the ceiling
    assert server.run_budget(0.10) == 0.10
    assert server.run_budget(None) == 0.25
    assert server.run_budget(0) == 0.25
    assert server.run_budget(-3) == 0.01


def test_total_spend_cap_blocks_paid_runs_but_not_the_baseline(client, monkeypatch):
    monkeypatch.setenv("MAX_TOTAL_SPEND_USD", "1.00")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")  # make the model "available" without a real key
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    server.hub.spent = 1.00
    try:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "start", "model": "gpt-5.4-mini", "task_id": "L1_pick_place"})
            msgs = []
            while True:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] in ("error", "result"):
                    break
            assert msgs[-1]["type"] == "error" and "spend cap" in msgs[-1]["message"]
    finally:
        server.hub.spent = 0.0

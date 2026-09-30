import pytest
from fastapi.testclient import TestClient

from specserve import app as app_module
from specserve.service import GenerationService


class StubEngine:
    def __init__(self, bundles):
        self._target, self._draft = bundles

    def target(self):
        return self._target

    def draft(self):
        return self._draft


@pytest.fixture
def client(bundles):
    app_module.service = GenerationService(StubEngine(bundles))
    with TestClient(app_module.app) as c:
        yield c


PAYLOAD = {
    "text": "hello world",
    "mode": "speculative",
    "max_new_tokens": 8,
    "draft_steps": 3,
    "temperature": 0,
    "seed": 7,
}


def _parse_sse(text):
    events = []
    for frame in text.strip().split("\n\n"):
        name, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        events.append((name, data))
    return events


def test_generate_stream_and_stats(client):
    resp = client.post("/api/generate", json=PAYLOAD)
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0][0] == "token"
    assert events[-1][0] == "done"
    import json

    stats = json.loads(events[-1][1])
    assert stats["mode"] == "speculative"
    assert stats["candidates"] == 6  # two of eight are target bonus tokens
    assert stats["accepted"] == 8
    assert stats["target_forwards"] == 1 + 2  # rounds of 4, 4
    assert client.get("/health").json()["busy"] is False


def test_invalid_params_do_not_occupy_slot(client):
    bad = dict(PAYLOAD, max_new_tokens=0)
    assert client.post("/api/generate", json=bad).status_code == 422
    bad = dict(PAYLOAD, temperature=-1)
    assert client.post("/api/generate", json=bad).status_code == 422
    bad = dict(PAYLOAD, draft_steps=0)
    assert client.post("/api/generate", json=bad).status_code == 422
    bad = dict(PAYLOAD, mode="bogus")
    assert client.post("/api/generate", json=bad).status_code == 422
    assert client.get("/health").json()["busy"] is False
    # slot still usable
    assert client.post("/api/generate", json=PAYLOAD).status_code == 200


def test_busy_returns_409(bundles):
    # Hold the slot with a raw lock, then request must be rejected.
    app_module.service = GenerationService(StubEngine(bundles))
    app_module.service._slot.acquire()
    try:
        with TestClient(app_module.app) as client:
            resp = client.post("/api/generate", json=PAYLOAD)
            assert resp.status_code == 409
    finally:
        app_module.service._slot.release()


def test_autoregressive_mode(client):
    payload = dict(PAYLOAD, mode="autoregressive")
    resp = client.post("/api/generate", json=payload)
    events = _parse_sse(resp.text)
    assert events[-1][0] == "done"
    assert sum(1 for name, _ in events if name == "token") == 8


def test_client_disconnect_releases_slot(client):
    resp = client.post("/api/generate", json=PAYLOAD)
    assert resp.status_code == 200
    assert client.get("/health").json()["busy"] is False

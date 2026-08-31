"""Gateway API tests with a scripted provider."""

import pytest
from fastapi.testclient import TestClient

from gateway.providers import ChatResult, ProviderError


@pytest.fixture
def client(monkeypatch, tmp_path):
    import gateway.api as api
    from gateway.metrics import UsageStore

    calls = {"tiers": []}

    def fake_chat(self, tier, messages, temperature=None, max_tokens=None):
        calls["tiers"].append(tier)
        if getattr(fake_chat, "fail_tier", None) == tier:
            raise ProviderError(f"{tier} is down")
        return ChatResult(f"answer from {tier}", f"model-{tier}", False, 42, 100, 20, 0.0005)

    monkeypatch.setattr(type(api._provider), "chat", fake_chat)
    api._usage = UsageStore(db_path=tmp_path / "usage.db")
    fake_chat.fail_tier = None
    return TestClient(api.app), calls, fake_chat


def test_auto_routing_returns_openai_shape(client):
    c, calls, _ = client
    response = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "answer from small"
    gw = body["gateway"]
    assert gw["decision"] == "small" and gw["served_by"] == "small"
    assert gw["fallback_used"] is False
    assert "cost_usd" in gw and "reason" in gw


def test_hard_prompt_routes_large(client):
    c, calls, _ = client
    body = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content":
                      "Implement a python function step by step and explain the algorithm"}],
    }).json()
    assert body["gateway"]["decision"] == "large"


def test_explicit_model_bypasses_router(client):
    c, calls, _ = client
    body = c.post("/v1/chat/completions", json={
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "hi"}],
    }).json()
    assert body["gateway"]["router"] == "explicit"
    assert body["gateway"]["served_by"] == "large"


def test_fallback_when_chosen_tier_fails(client):
    c, calls, fake_chat = client
    fake_chat.fail_tier = "small"
    body = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }).json()
    assert body["gateway"]["decision"] == "small"
    assert body["gateway"]["served_by"] == "large"
    assert body["gateway"]["fallback_used"] is True
    assert "small" in body["gateway"]["fallback_errors"]


def test_validation_and_unknown_model(client):
    c, _, _ = client
    assert c.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "system", "content": "x"}],
    }).status_code == 422
    assert c.post("/v1/chat/completions", json={
        "model": "gpt-99", "messages": [{"role": "user", "content": "x"}],
    }).status_code == 404


def test_admin_endpoints(client):
    c, _, _ = client
    c.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hello there"}],
    })
    metrics = c.get("/admin/metrics").json()
    assert metrics["usage"]["requests"] >= 1
    assert "hit_rate" in metrics["cache"]

    routers = c.get("/admin/routers").json()
    assert "v1_score" in routers["available"]
    assert c.post("/admin/routers/v0_rules").json()["active"] == "v0_rules"
    assert c.post("/admin/routers/nope").status_code == 404
    c.post("/admin/routers/v1_score")


def test_landing_and_health(client):
    c, _, _ = client
    assert "LLM Gateway" in c.get("/").text
    health = c.get("/healthz").json()
    assert health["status"] == "ok" and "routers_loaded" in health

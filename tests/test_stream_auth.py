"""Streaming and auth/budget tests with a scripted provider."""

import json

import pytest
from fastapi.testclient import TestClient

from gateway.config import AuthConfig, KeySpec
from gateway.providers import ProviderError


@pytest.fixture
def client(monkeypatch, tmp_path):
    import gateway.api as api
    from gateway.metrics import UsageStore

    def fake_stream_chat(self, tier, messages, temperature=None, max_tokens=None):
        from gateway.config import get_config
        from gateway.providers import ResponseCache

        if getattr(fake_stream_chat, "fail_tier", None) == tier:
            raise ProviderError(f"{tier} is down")

        def events():
            yield {"delta": "Hello "}
            yield {"delta": f"from {tier}"}
            yield {"usage": {"input": 50, "output": 10}}

        spec = get_config().models[tier]
        temp = get_config().llm.temperature if temperature is None else temperature
        cache_key = ResponseCache.key_for(spec.name, messages, temp)
        return events(), f"model-{tier}", cache_key

    from gateway.providers import ResponseCache

    monkeypatch.setattr(type(api._provider), "stream_chat", fake_stream_chat)
    api._usage = UsageStore(db_path=tmp_path / "usage.db")
    api._provider.cache = ResponseCache(max_entries=100, ttl_seconds=60)  # fresh per test
    fake_stream_chat.fail_tier = None
    return TestClient(api.app), api, fake_stream_chat


def _read_sse(response):
    chunks = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    assert response.text.rstrip().endswith("data: [DONE]")
    return chunks


def test_streaming_returns_sse_chunks_and_records_cost(client):
    c, api, _ = client
    response = c.post("/v1/chat/completions", json={
        "model": "auto", "stream": True,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = _read_sse(response)
    text = "".join(
        ch["choices"][0]["delta"].get("content", "") for ch in chunks
    )
    assert text == "Hello from small"
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["gateway"]["served_by"] == "small"
    assert final["usage"]["total_tokens"] == 60
    assert api._usage.summary()["requests"] == 1


def test_streaming_fallback_before_first_byte(client):
    c, _, fake = client
    fake.fail_tier = "small"
    response = c.post("/v1/chat/completions", json={
        "model": "auto", "stream": True,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    })
    chunks = _read_sse(response)
    assert chunks[-1]["gateway"]["served_by"] == "large"
    assert chunks[-1]["gateway"]["fallback_used"] is True


def test_streamed_response_populates_cache_for_repeat(client):
    c, api, _ = client
    payload = {"model": "auto", "stream": True,
               "messages": [{"role": "user", "content": "What is the capital of Peru?"}]}
    c.post("/v1/chat/completions", json=payload)
    response = c.post("/v1/chat/completions", json=payload)
    chunks = _read_sse(response)
    assert chunks[-1]["gateway"]["cached"] is True
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello from small"


@pytest.fixture
def authed(monkeypatch, client):
    c, api, fake = client
    auth = AuthConfig(enabled=True, keys={
        "sk-team-a": KeySpec(name="team-a", daily_budget_usd=1.0),
        "sk-broke": KeySpec(name="broke", daily_budget_usd=0.0),
    })
    monkeypatch.setattr(get_cfg_target := api.get_config(), "auth", auth)
    return c, api


def test_auth_rejects_missing_and_unknown_keys(authed):
    c, _ = authed
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi there"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    assert c.post("/v1/chat/completions", json=body,
                  headers={"Authorization": "Bearer nope"}).status_code == 401


def test_auth_accepts_key_and_tracks_spend(authed, monkeypatch):
    c, api = authed

    def fake_chat(self, tier, messages, temperature=None, max_tokens=None):
        from gateway.providers import ChatResult
        return ChatResult("hi", f"model-{tier}", False, 5, 100, 20, 0.01)

    monkeypatch.setattr(type(api._provider), "chat", fake_chat)
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi there"}]}
    ok = c.post("/v1/chat/completions", json=body,
                headers={"Authorization": "Bearer sk-team-a"})
    assert ok.status_code == 200
    assert api._usage.summary()["by_key"]["sk-team-a"]["requests"] == 1


def test_budget_exhaustion_returns_429(authed):
    c, _ = authed
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi there"}]}
    response = c.post("/v1/chat/completions", json=body,
                      headers={"Authorization": "Bearer sk-broke"})
    assert response.status_code == 429
    assert "budget" in response.json()["detail"]

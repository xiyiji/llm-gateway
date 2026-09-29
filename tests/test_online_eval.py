"""Online evaluation: sampler, pairwise judge, quality guard, shadow scoring,
promotion rule, trackers, and the loop wired through the gateway API."""

import json

import pytest
from fastapi.testclient import TestClient

from gateway.config import OnlineEvalConfig
from gateway.online_eval import (PAIRWISE_PROMPT, GuardedRouter, Judgment, JudgmentStore,
                                 OnlineEvaluator, PairwiseJudge, QualityGuard, Sample,
                                 Sampler, category_of, shadow_compare)
from gateway.providers import ChatResult
from gateway.routers import AlwaysLarge, AlwaysSmall, RouteDecision
from gateway.tracking import JsonlTracker
from scripts.retrain_from_live import decide_promotion


class ScriptedProvider:
    """Answers say which tier wrote them; the judge scores 'good' answers 5
    and everything else 2. `small_good` decides whether small answers are good."""

    cache_enabled = False

    def __init__(self, small_good=lambda prompt: True):
        self.small_good = small_good
        self.calls = []

    def chat(self, tier, messages, temperature=None, max_tokens=None):
        if messages[0]["role"] == "system" and messages[0]["content"] == PAIRWISE_PROMPT:
            p = json.loads(messages[1]["content"])
            sa = 5 if "good" in p["answer_a"] else 2
            sb = 5 if "good" in p["answer_b"] else 2
            return ChatResult(json.dumps({"score_a": sa, "score_b": sb}), "judge",
                              False, 1, 10, 10, 0.001)
        prompt = [m["content"] for m in messages if m["role"] == "user"][-1]
        self.calls.append(tier)
        good = tier == "large" or self.small_good(prompt)
        return ChatResult(f"{tier} {'good' if good else 'poor'}", f"m-{tier}", False, 1,
                          10, 10, 0.0001 if tier == "small" else 0.001)

    def available(self):
        return True, None


def _cfg(**kw):
    base = dict(enabled=True, sample_rate=1.0, warmup_per_category=0, guard_threshold=0.8,
                guard_window=10, guard_min_samples=5, guard_recover_margin=0.05,
                judge_budget_usd=100.0, log_every=1000)
    base.update(kw)
    return OnlineEvalConfig(**base)


def test_category_of():
    assert category_of("Implement a python function to parse dates") == "code"
    assert category_of("Calculate 12 + 30") == "math"
    assert category_of("Explain why the sky is blue") == "reasoning"
    assert category_of("What is the capital of Peru?") == "lookup"
    assert category_of("Tell me a story about a fox") == "general"


def test_sampler_warmup_then_rate():
    s = Sampler(rate=0.0, warmup_per_category=3)
    assert [s.take("lookup") for _ in range(5)] == [True, True, True, False, False]
    assert s.take("code")  # a new category gets its own warmup


def test_pairwise_judge_maps_back_through_random_order():
    judge = PairwiseJudge(ScriptedProvider(), seed=0)
    for _ in range(20):  # both presentation orders get exercised
        v = judge.judge("q", "small poor", "large good")
        assert (v["score_small"], v["score_large"]) == (2, 5)
        v = judge.judge("q", "small good", "large poor")
        assert (v["score_small"], v["score_large"]) == (5, 2)


def test_guard_trips_reroutes_and_releases_with_hysteresis():
    events = []
    g = QualityGuard(threshold=0.8, window=10, min_samples=5, recover_margin=0.05,
                     on_event=events.append)
    for _ in range(4):
        g.observe("lookup", False)
    assert not g.is_overridden("lookup")  # below min_samples: no decision yet
    g.observe("lookup", False)
    assert g.is_overridden("lookup") and events[-1]["event"] == "guard_on"

    small = RouteDecision("small", "looks easy", 0.9, "v1_score")
    guarded = g.apply(small, "lookup")
    assert guarded.action == "large" and guarded.router == "v1_score+guard"
    assert guarded.router_action == "small"
    assert g.apply(small, "math").action == "small"  # other categories untouched

    for _ in range(8):  # window now 8 good of 10 -> 0.8: recovered but not past margin
        g.observe("lookup", True)
    assert g.is_overridden("lookup")
    g.observe("lookup", True)  # 9/10 = 0.9 >= 0.85
    assert not g.is_overridden("lookup") and events[-1]["event"] == "guard_off"


def test_guard_reset_keeps_override_until_fresh_evidence():
    g = QualityGuard(0.8, 10, 5, 0.05)
    for _ in range(5):
        g.observe("lookup", False)
    g.reset_history()
    assert g.is_overridden("lookup")
    for _ in range(4):
        g.observe("lookup", True)
    assert g.is_overridden("lookup")  # 4 < min_samples
    g.observe("lookup", True)
    assert not g.is_overridden("lookup")


def _j(prompt, s_small, s_large, c_small=0.1, c_large=1.0):
    return Judgment("id" + prompt, 0.0, category_of(prompt), "r", "small", "small",
                    s_small >= s_large, s_small, s_large, c_small, c_large, 0.0, prompt)


def test_shadow_compare_uses_both_counterfactual_outcomes():
    js = [_j("What is the capital of Peru?", 5, 5), _j("Tell me a story", 1, 5)]
    out = shadow_compare(js, {"small": AlwaysSmall(), "large": AlwaysLarge()})
    assert out["small"]["mean_quality"] == pytest.approx(0.5)
    assert out["large"]["mean_quality"] == pytest.approx(1.0)
    assert out["small"]["mean_cost_usd"] == pytest.approx(0.1)
    assert out["_oracle_quality"] == pytest.approx(1.0)
    guarded = GuardedRouter(AlwaysSmall(), {"general"})
    g = shadow_compare(js, {"g": guarded})["g"]
    assert g["mean_quality"] == pytest.approx(1.0) and g["small_share"] == 0.5


def test_promotion_never_trades_quality_for_cost():
    active = {"mean_quality": 0.90, "mean_cost_usd": 1.0}
    assert decide_promotion({"mean_quality": 0.90, "mean_cost_usd": 0.8}, active)[0]
    assert decide_promotion({"mean_quality": 0.93, "mean_cost_usd": 1.05}, active)[0]
    assert not decide_promotion({"mean_quality": 0.89, "mean_cost_usd": 0.2}, active)[0]
    assert not decide_promotion({"mean_quality": 0.95, "mean_cost_usd": 2.0}, active)[0]


def test_evaluator_labels_guard_budget_and_queue(tmp_path):
    provider = ScriptedProvider(small_good=lambda p: "hard" not in p)
    ev = OnlineEvaluator(provider, _cfg(), store=JudgmentStore(tmp_path / "j.db"),
                         tracker=JsonlTracker(tmp_path / "e.jsonl"))
    for i in range(6):
        prompt = f"What is the hard case {i}?"
        assert ev.offer(Sample(prompt, [{"role": "user", "content": prompt}], "lookup",
                               "v1_score", "small", "small", "small poor", 0.0001))
    assert ev.drain() == 6
    labels = ev.export_labels()
    assert len(labels) == 6 and all(l["small_ok"] == 0 for l in labels)
    assert provider.calls.count("large") == 6  # counterfactual ran on the other tier
    assert ev.guard.is_overridden("lookup")
    assert (tmp_path / "e.jsonl").read_text().count("guard_on") == 1

    # requests the router sent to large still become labels, but don't move the guard
    prompt = "Explain step by step"
    ev.offer(Sample(prompt, [{"role": "user", "content": prompt}], "reasoning",
                    "v1_score", "large", "large", "large good", 0.001))
    ev.drain()
    assert "reasoning" not in ev.guard.state()

    ev.judge_spend_usd = ev.cfg.judge_budget_usd  # budget exhausted: stop spending
    ev.offer(Sample("q", [{"role": "user", "content": "q"}], "general", "r", "small",
                    "small", "small good", 0.0))
    assert ev.drain() == 1 and len(ev.export_labels()) == 7


@pytest.fixture
def online_client(monkeypatch, tmp_path):
    import gateway.api as api
    from gateway.metrics import UsageStore
    from gateway.routers import build_registry

    provider = ScriptedProvider(small_good=lambda p: "capital" not in p)
    monkeypatch.setattr(api, "_provider", provider)
    monkeypatch.setattr(api, "_usage", UsageStore(db_path=tmp_path / "usage.db"))
    monkeypatch.setattr(api, "_registry", build_registry())
    monkeypatch.setitem(api._state, "active_router", "v1_score")
    ev = OnlineEvaluator(provider, _cfg(), store=JudgmentStore(tmp_path / "j.db"),
                         tracker=JsonlTracker(tmp_path / "e.jsonl"))
    monkeypatch.setattr(api, "_evaluator", ev)
    return TestClient(api.app), ev


def _ask(client, text, model="auto"):
    return client.post("/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": text}]}).json()


def test_api_loop_detects_protects_and_reports(online_client):
    client, ev = online_client
    for i in range(5):  # the router sends these to small; the small model is wrong
        body = _ask(client, f"What is the capital of country {i}?")
        assert body["gateway"]["served_by"] == "small"
    client.post("/admin/online-eval/drain")
    body = _ask(client, "What is the capital of country 99?")
    assert body["gateway"]["served_by"] == "large"
    assert "quality guard" in body["gateway"]["reason"]
    assert _ask(client, "Tell me a story about a fox")["gateway"]["served_by"] == "small"

    summary = client.get("/admin/online-eval").json()
    assert summary["enabled"] and summary["categories"]["lookup"]["overridden"]
    assert summary["guard_events"][0]["event"] == "guard_on"
    assert "v1_score" in summary["shadow"]

    # explicit model requests bypass routing and are not sampled
    before = ev._q.qsize()
    _ask(client, "What is the capital of X?", model="deepseek-chat")
    assert ev._q.qsize() == before


def test_router_switch_resets_guard_evidence(online_client):
    client, ev = online_client
    for i in range(5):
        _ask(client, f"What is the capital of country {i}?")
    client.post("/admin/online-eval/drain")
    assert ev.guard.state()["lookup"]["n"] == 5
    client.post("/admin/routers/v0_rules")
    assert ev.guard.state() == {} and ev.guard.is_overridden("lookup")


def test_online_eval_disabled_by_default():
    import gateway.api as api

    if api._evaluator is None:
        assert TestClient(api.app).get("/admin/online-eval").json() == {"enabled": False}


def test_mlflow_tracker_records_metrics_events_and_registry(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from gateway.tracking import MlflowTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    t = MlflowTracker(uri, "test-exp")
    t.log_metrics({"small_win_rate.lookup": 0.5}, step=1)
    t.log_event({"event": "guard_on", "category": "lookup"})
    model = tmp_path / "m.pkl"
    model.write_bytes(b"x")
    t.log_router_run("v2_live", {"a": 1}, {"q": 0.9}, model, register=True)
    c = MlflowClient(uri)
    runs = c.search_runs([c.get_experiment_by_name("test-exp").experiment_id])
    kinds = {r.data.tags["kind"] for r in runs}
    assert kinds == {"online_eval", "router_training"}
    assert str(c.get_model_version_by_alias("gateway-router", "candidate").version) == "1"

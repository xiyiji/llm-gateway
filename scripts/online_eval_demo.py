"""The online-eval loop, end to end, on a traffic shift the offline benchmark
cannot see. Runs through the real gateway API with a simulated provider, so
it needs no API key and is fully reproducible.

The scenario:
  1. baseline  - traffic looks like the benchmark (lookups, arithmetic,
                 explanations, code). The active router (v1_score) sends the
                 easy half to the small model and quality holds.
  2. shift     - a new kind of request appears: short "What is the tax
                 treatment of ...?" policy questions. They look like simple
                 lookups to the router (short, "what is"), so they go to the
                 small model, which gets most of them wrong. The benchmark has
                 no such prompts, so offline numbers do not move.
  3. protect   - the sampled live judgments show the small model's win rate
                 on routed-small 'lookup' requests collapsing; the quality
                 guard reroutes that category to the large model.
  4. learn     - the judged live requests become labels; a retrained router
                 is scored against the active one on newer live requests it
                 never saw, promoted only because it wins, and the guard
                 releases once the new router's small calls hold up.

The simulated provider has a hidden ground truth per request (was the small
model's answer good?), and the simulated judge reads it. Real deployments
replace both with the actual models and judge; the gateway code is identical.

Usage: python scripts/online_eval_demo.py [--seed 7] [--json var/online_eval_demo.json]
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.config import OnlineEvalConfig  # noqa: E402
from gateway.online_eval import PAIRWISE_PROMPT, JudgmentStore, OnlineEvaluator  # noqa: E402
from gateway.providers import ChatResult  # noqa: E402
from gateway.tracking import JsonlTracker  # noqa: E402

COST = {"small": 0.00020, "large": 0.00110}
P_SMALL_GOOD = {"capital": 0.97, "arith": 0.95, "explain": 0.55, "code": 0.40, "policy": 0.15}
P_LARGE_GOOD = {"capital": 0.99, "arith": 0.99, "explain": 0.93, "code": 0.90, "policy": 0.92}

CITIES = ["France", "Japan", "Kenya", "Peru", "Norway", "Chile", "Egypt", "Canada",
          "Vietnam", "Poland", "Ghana", "Nepal", "Spain", "Brazil", "Iceland"]
POLICY = [
    "What is the tax treatment of exercising incentive stock options and selling "
    "the shares in the same calendar year in {st}, including AMT?",
    "What is the rule for deducting a home office used partly for a W-2 job and "
    "partly for freelance work in {st} this year?",
    "What is the correct way to report a wash sale that spans two brokerage "
    "accounts, one of them an IRA, for a {st} resident?",
    "What is the depreciation schedule for a rental property bought mid-year in "
    "{st} if part of it is used personally for a few weeks?",
]
STATES = ["Washington", "California", "Texas", "New York", "Oregon", "Florida"]


def make_request(rng: random.Random, topic: str, n: int):
    if topic == "capital":
        return f"What is the capital of {rng.choice(CITIES)}? (#{n})"
    if topic == "arith":
        return f"Calculate {rng.randint(10, 99)} + {rng.randint(10, 99)}. Just the number. (#{n})"
    if topic == "explain":
        return (f"Explain step by step why a hash map lookup is O(1) on average and "
                f"compare it with a balanced tree. (#{n})")
    if topic == "code":
        return (f"Implement a python function that parses an ISO date and returns the "
                f"weekday; include unit tests. (#{n})")
    return rng.choice(POLICY).format(st=rng.choice(STATES)) + f" (#{n})"


class SimProvider:
    """Deterministic stand-in for the model API. Each request carries a hidden
    outcome per tier; the judge call reads the outcomes back from the answers."""

    cache_enabled = False

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.truth = {}  # prompt -> {"topic", "small", "large"}

    def register(self, prompt: str, topic: str):
        self.truth[prompt] = {"topic": topic,
                              "small": self.rng.random() < P_SMALL_GOOD[topic],
                              "large": self.rng.random() < P_LARGE_GOOD[topic]}

    def chat(self, tier, messages, temperature=None, max_tokens=None):
        if messages and messages[0]["role"] == "system" and messages[0]["content"] == PAIRWISE_PROMPT:
            payload = json.loads(messages[1]["content"])
            sa = 5 if "good" in payload["answer_a"] else 2
            sb = 5 if "good" in payload["answer_b"] else 2
            return ChatResult(json.dumps({"score_a": sa, "score_b": sb, "rationale": "sim"}),
                              "sim-judge", False, 5, 200, 30, COST["large"])
        prompt = [m["content"] for m in messages if m["role"] == "user"][-1]
        good = self.truth[prompt][tier]
        return ChatResult(f"[{tier}] {'good' if good else 'poor'} answer", f"sim-{tier}",
                          False, 5, 60, 40, COST[tier])

    def available(self):
        return True, None


def run(seed: int = 7, work_dir: Path = None, use_mlflow: bool = False) -> dict:
    import gateway.api as api
    from fastapi.testclient import TestClient
    from gateway.metrics import UsageStore
    from gateway.routers import build_registry
    from scripts.retrain_from_live import retrain

    work_dir = Path(work_dir or tempfile.mkdtemp(prefix="online-eval-demo-"))
    rng = random.Random(seed)
    sim = SimProvider(seed)
    cfg = OnlineEvalConfig(enabled=True, sample_rate=0.15, warmup_per_category=20,
                           guard_threshold=0.80, guard_window=40, guard_min_samples=20,
                           guard_recover_margin=0.05, judge_budget_usd=100.0, log_every=10)
    if use_mlflow:
        from gateway.tracking import MlflowTracker

        tracker = MlflowTracker(f"sqlite:///{work_dir / 'mlflow.db'}", "online-eval-demo")
    else:
        tracker = JsonlTracker(work_dir / "events.jsonl")
    store = JudgmentStore(work_dir / "online_eval.db")

    api._provider = sim
    api._usage = UsageStore(db_path=work_dir / "usage.db")
    api._registry = build_registry()
    api._state["active_router"] = "v1_score"
    api._evaluator = OnlineEvaluator(sim, cfg, store=store, tracker=tracker, seed=seed)
    client = TestClient(api.app)

    counter = {"n": 0}
    seen_events = []

    def send(topic_weights: dict, n_requests: int) -> dict:
        topics, weights = zip(*topic_weights.items())
        good = cost = small = policy_small = policy = 0
        unguarded_good = 0
        for _ in range(n_requests):
            counter["n"] += 1
            topic = rng.choices(topics, weights)[0]
            prompt = make_request(rng, topic, counter["n"])
            sim.register(prompt, topic)
            body = client.post("/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": prompt}]}).json()
            served = body["gateway"]["served_by"]
            good += sim.truth[prompt][served]
            router_tier = api._registry[api._state["active_router"]].route(prompt).action
            unguarded_good += sim.truth[prompt][router_tier]  # same router, no guard
            cost += body["gateway"]["cost_usd"]
            small += served == "small"
            if topic == "policy":
                policy += 1
                policy_small += served == "small"
            api._evaluator.drain()  # judge in step; in production this is the worker thread
            for event in api._evaluator.guard.events[len(seen_events):]:
                seen_events.append({**event, "at_request": counter["n"]})
        return {"requests": n_requests, "served_quality": round(good / n_requests, 3),
                "quality_without_guard": round(unguarded_good / n_requests, 3),
                "cost_per_1k_usd": round(cost / n_requests * 1000, 4),
                "small_share": round(small / n_requests, 3),
                **({"policy_sent_small": round(policy_small / policy, 3)} if policy else {}),
                "guard": {k: v for k, v in api._evaluator.guard.state().items()}}

    baseline_mix = {"capital": 40, "arith": 30, "explain": 15, "code": 15}
    shifted_mix = {"capital": 26, "arith": 20, "explain": 10, "code": 10, "policy": 34}

    phases = {}
    phases["1_baseline"] = send(baseline_mix, 400)
    phases["2_shift_first_100"] = send(shifted_mix, 100)
    phases["3_shift_guarded"] = send(shifted_mix, 800)

    result = retrain(store, "v1_score", holdout=0.3, out_path=work_dir / "v2_live.pkl",
                     tracker=tracker, include_offline=False,
                     guarded_categories=api._evaluator.guard.overridden_categories())
    if result["promote"]:
        from gateway.routers import V2Learned

        live = V2Learned(Path(result["model_path"]))
        live.name = "v2_live"
        api._registry["v2_live"] = live
        client.post("/admin/routers/v2_live")
    phases["4_retrained_guard_still_on"] = send(shifted_mix, 400)
    phases["5_retrained_guard_released"] = send(shifted_mix, 400)

    summary = client.get("/admin/online-eval").json()
    return {"seed": seed, "phases": phases,
            "retrain": {k: result[k] for k in ("promote", "reason", "scores")},
            "guard_events": seen_events,
            "shift_started_at_request": 401,
            "judged": summary["judged"], "judge_spend_usd": summary["judge_spend_usd"],
            "work_dir": str(work_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--mlflow", action="store_true",
                        help="record metrics, guard events and router runs in MLflow")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()
    out = run(args.seed, args.work_dir, use_mlflow=args.mlflow)
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.parent.mkdir(exist_ok=True, parents=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Online evaluation: keep the routing decision honest after launch.

The offline benchmark is a launch gate. It says the router is safe on the
tasks we chose. Production asks a different question every day: is the small
model still good enough on what people actually send, with whatever model
version the provider is serving this week?

This module answers that on live traffic, off the request path:

    request ──▶ router ──▶ QualityGuard ──▶ provider ──▶ response to client
                                                  │
                                    Sampler (e.g. 5%, plus the first N of
                                    every new request category)
                                                  ▼
                               queue ──▶ worker: run the *other* tier on the
                               same messages (counterfactual), then a
                               pairwise judge: was the small answer as good?
                                                  ▼
                   JudgmentStore (SQLite) ──▶ QualityGuard (per-category win
                   rate, reroutes a failing category to the large model)
                                         ──▶ shadow comparison of routers on
                                             the same live samples
                                         ──▶ labels for the next router
                                         ──▶ Tracker (MLflow when installed)

Why a counterfactual instead of a reference answer: live traffic has no
reference. Running both tiers on a small sample turns every judged request
into the exact label the router is trained on ("was the small model
sufficient here?"), and it lets any candidate router be scored on the same
requests without guessing: both outcomes are known.
"""

import json
import queue
import random
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from gateway.config import VAR_DIR, OnlineEvalConfig
from gateway.features import featurize
from gateway.routers import RouteDecision

CATEGORIES = ("code", "math", "reasoning", "lookup", "general")


def category_of(prompt: str) -> str:
    """Coarse request type from the same named features the routers use.
    Quality is tracked per category because a global average hides the one
    kind of request that is failing."""
    _, f = featurize(prompt)
    if f["code_kw"] > 0 or f["has_code_block"] > 0:
        return "code"
    if f["math_kw"] > 0 or f["math_op_count"] >= 0.4:
        return "math"
    if f["reasoning_kw"] > 0 or f["multistep_kw"] >= 0.5:
        return "reasoning"
    if f["simple_kw"] > 0:
        return "lookup"
    return "general"


@dataclass
class Sample:
    prompt: str
    messages: List[Dict]
    category: str
    router: str
    decision: str          # what the router chose ("small" | "large")
    served_tier: str       # what actually answered (after guard / fallback)
    response_text: str
    cost_usd: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    ts: float = field(default_factory=time.time)


@dataclass
class Judgment:
    sample_id: str
    ts: float
    category: str
    router: str
    decision: str
    served_tier: str
    small_ok: bool         # label: small answer at least as good as large
    score_small: int       # judge scores 1-5
    score_large: int
    cost_small: float
    cost_large: float
    judge_cost: float
    prompt: str


class Sampler:
    """Uniform rate, plus the first `warmup_per_category` requests of every
    category so a brand-new kind of traffic is measured immediately instead
    of after thousands of requests."""

    def __init__(self, rate: float, warmup_per_category: int = 20, seed: int = 0):
        self.rate = rate
        self.warmup = warmup_per_category
        self._seen: Dict[str, int] = {}
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def take(self, category: str) -> bool:
        with self._lock:
            n = self._seen.get(category, 0)
            self._seen[category] = n + 1
            return n < self.warmup or self._rng.random() < self.rate


PAIRWISE_PROMPT = """You compare two answers to the same user request.
Grade each answer 1-5 for correctness and completeness (5 = fully correct and
complete, 3 = partly correct or missing a key point, 1 = wrong or off-topic).
Judge the substance, not the length or style.
Respond with JSON only: {"score_a": <1-5>, "score_b": <1-5>, "rationale": "<one sentence>"}"""


class PairwiseJudge:
    """Reference-free pairwise judge. The two answers are shown in random
    order and mapped back, so the judge's position bias cannot favour one
    tier systematically."""

    def __init__(self, provider, judge_tier: str = "large", seed: int = 0):
        self.provider = provider
        self.judge_tier = judge_tier
        self._rng = random.Random(seed)

    def judge(self, prompt: str, small_text: str, large_text: str) -> Dict:
        small_first = self._rng.random() < 0.5
        a, b = (small_text, large_text) if small_first else (large_text, small_text)
        payload = json.dumps({"request": prompt, "answer_a": a, "answer_b": b})
        result = self.provider.chat(
            self.judge_tier,
            [{"role": "system", "content": PAIRWISE_PROMPT},
             {"role": "user", "content": payload}],
        )
        sa = _score(result.text, "score_a")
        sb = _score(result.text, "score_b")
        s_small, s_large = (sa, sb) if small_first else (sb, sa)
        return {"score_small": s_small, "score_large": s_large,
                "judge_cost": result.cost_usd}


def _score(text: str, key: str) -> int:
    match = re.search(rf'"{key}"\s*:\s*(\d)', text)
    return max(1, min(5, int(match.group(1)))) if match else 1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    sample_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    category TEXT NOT NULL,
    router TEXT NOT NULL,
    decision TEXT NOT NULL,
    served_tier TEXT NOT NULL,
    small_ok INTEGER NOT NULL,
    score_small INTEGER NOT NULL,
    score_large INTEGER NOT NULL,
    cost_small REAL NOT NULL,
    cost_large REAL NOT NULL,
    judge_cost REAL NOT NULL,
    prompt TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judgments_cat_ts ON judgments(category, ts);
"""


class JudgmentStore:
    def __init__(self, db_path: Optional[Path] = None):
        VAR_DIR.mkdir(exist_ok=True)
        self.db_path = db_path or (VAR_DIR / "online_eval.db")
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def add(self, j: Judgment) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (j.sample_id, j.ts, j.category, j.router, j.decision, j.served_tier,
                 int(j.small_ok), j.score_small, j.score_large, j.cost_small,
                 j.cost_large, j.judge_cost, j.prompt),
            )

    def all(self, since_ts: float = 0.0) -> List[Judgment]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM judgments WHERE ts >= ? ORDER BY ts", (since_ts,)
            ).fetchall()
        return [Judgment(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6]), r[7], r[8],
                         r[9], r[10], r[11], r[12]) for r in rows]


class QualityGuard:
    """Per-category safety valve on the routing decision.

    Tracks the small model's live win rate over the last `window` judged
    requests of each category *that the router chose to send to the small
    model* — the question is whether that routing decision is still safe. When it falls below `threshold` (with at least
    `min_samples` judgments), requests in that category that the router would
    send to the small model go to the large one instead. The override lifts
    only once the rate recovers to `threshold + recover_margin` (hysteresis,
    so the route doesn't flap). The small model keeps being measured while
    overridden, because sampled requests still get a counterfactual small
    answer; that is what lets the guard release on its own."""

    def __init__(self, threshold: float, window: int, min_samples: int,
                 recover_margin: float, on_event: Optional[Callable[[Dict], None]] = None):
        self.threshold = threshold
        self.window = window
        self.min_samples = min_samples
        self.recover_margin = recover_margin
        self._hist: Dict[str, Deque[bool]] = {}
        self._overridden: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._on_event = on_event or (lambda e: None)
        self.events: List[Dict] = []

    def rate(self, category: str) -> Optional[float]:
        hist = self._hist.get(category)
        if not hist:
            return None
        return sum(hist) / len(hist)

    def observe(self, category: str, small_ok: bool) -> None:
        with self._lock:
            hist = self._hist.setdefault(category, deque(maxlen=self.window))
            hist.append(small_ok)
            n, rate = len(hist), sum(hist) / len(hist)
            event = None
            if category not in self._overridden:
                if n >= self.min_samples and rate < self.threshold:
                    self._overridden[category] = time.time()
                    event = {"event": "guard_on", "category": category,
                             "small_win_rate": round(rate, 3), "n": n}
            elif n >= self.min_samples and rate >= self.threshold + self.recover_margin:
                del self._overridden[category]
                event = {"event": "guard_off", "category": category,
                         "small_win_rate": round(rate, 3), "n": n}
            if event:
                event["ts"] = time.time()
                self.events.append(event)
        if event:
            self._on_event(event)

    def reset_history(self) -> None:
        """Called when the active router changes: the old router's win rate
        says nothing about the new one. Overrides stay on until the new
        router earns `min_samples` of fresh evidence."""
        with self._lock:
            self._hist.clear()

    def overridden_categories(self):
        return set(self._overridden)

    def is_overridden(self, category: str) -> bool:
        return category in self._overridden

    def apply(self, decision: RouteDecision, category: str) -> RouteDecision:
        if decision.action != "small" or not self.is_overridden(category):
            return decision
        rate = self.rate(category) or 0.0
        guarded = RouteDecision(
            "large",
            f"quality guard: small-model win rate on live '{category}' traffic is "
            f"{rate:.0%}, below {self.threshold:.0%} (router said: {decision.reason})",
            decision.confidence, f"{decision.router}+guard",
        )
        guarded.router_action = decision.action  # what the router itself chose
        return guarded

    def state(self) -> Dict:
        with self._lock:
            return {
                cat: {"small_win_rate": round(sum(h) / len(h), 3), "n": len(h),
                      "overridden": cat in self._overridden}
                for cat, h in sorted(self._hist.items())
            }


class GuardedRouter:
    """What production actually serves: a router plus the guard's current
    overrides. Candidates must beat this, not the bare router."""

    def __init__(self, router, overridden_categories):
        self.router = router
        self.overridden = set(overridden_categories)
        self.name = f"{getattr(router, 'name', 'router')}+guard"

    def route(self, text: str) -> RouteDecision:
        decision = self.router.route(text)
        if decision.action == "small" and category_of(text) in self.overridden:
            return RouteDecision("large", "guarded category", decision.confidence, self.name)
        return decision

    def overridden_categories(self):
        return set(self.overridden)


def _quality(score: int) -> float:
    return (score - 1) / 4.0


def shadow_compare(judgments: List[Judgment], routers: Dict[str, object]) -> Dict:
    """Score every router on the same judged live requests.

    Each judgment holds the outcome of *both* tiers, so a router's value on a
    request is simply the outcome of the tier it would have picked: no
    importance weighting, no guessing. Quality = judge score mapped to 0-1."""
    out = {}
    for name, router in routers.items():
        q = c = n_small = 0.0
        for j in judgments:
            action = router.route(j.prompt).action
            if action == "small":
                q += _quality(j.score_small); c += j.cost_small; n_small += 1
            else:
                q += _quality(j.score_large); c += j.cost_large
        n = max(len(judgments), 1)
        out[name] = {"mean_quality": round(q / n, 4), "mean_cost_usd": round(c / n, 8),
                     "small_share": round(n_small / n, 4), "n": len(judgments)}
    if judgments:
        oracle_q = sum(max(_quality(j.score_small), _quality(j.score_large))
                       for j in judgments) / len(judgments)
        out["_large_only_quality"] = round(
            sum(_quality(j.score_large) for j in judgments) / len(judgments), 4)
        out["_oracle_quality"] = round(oracle_q, 4)
    return out


class OnlineEvaluator:
    """Owns the sample queue, the worker, and everything downstream of a
    judgment. `offer` is the only call on the request path and never blocks."""

    def __init__(self, provider, cfg: OnlineEvalConfig, store: Optional[JudgmentStore] = None,
                 tracker=None, seed: int = 0):
        from gateway.tracking import build_tracker

        self.provider = provider
        self.cfg = cfg
        self.store = store or JudgmentStore()
        self.tracker = tracker or build_tracker(cfg)
        self.sampler = Sampler(cfg.sample_rate, cfg.warmup_per_category, seed)
        self.judge = PairwiseJudge(provider, cfg.judge_tier, seed)
        self.guard = QualityGuard(cfg.guard_threshold, cfg.guard_window,
                                  cfg.guard_min_samples, cfg.guard_recover_margin,
                                  on_event=self.tracker.log_event)
        self._q: "queue.Queue[Sample]" = queue.Queue(maxsize=cfg.queue_max)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.dropped = 0
        self.judged = 0
        self.judge_spend_usd = 0.0

    # ---- request path ------------------------------------------------------
    def offer(self, sample: Sample) -> bool:
        if not self.sampler.take(sample.category):
            return False
        try:
            self._q.put_nowait(sample)
            return True
        except queue.Full:  # never slow the request path; count what we shed
            self.dropped += 1
            return False

    # ---- worker ------------------------------------------------------------
    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.evaluate(sample)
            except Exception as exc:  # a judge failure must never kill the loop
                self.tracker.log_event({"event": "judge_error", "error": str(exc)[:200]})

    def drain(self) -> int:
        """Judge everything queued, synchronously (tests, demos, admin)."""
        n = 0
        while True:
            try:
                sample = self._q.get_nowait()
            except queue.Empty:
                return n
            self.evaluate(sample)
            n += 1

    def evaluate(self, sample: Sample) -> Optional[Judgment]:
        if self.judge_spend_usd >= self.cfg.judge_budget_usd:
            self.dropped += 1
            return None
        other = "large" if sample.served_tier == "small" else "small"
        counterfactual = self.provider.chat(other, sample.messages)
        if sample.served_tier == "small":
            small_text, large_text = sample.response_text, counterfactual.text
            cost_small, cost_large = sample.cost_usd, counterfactual.cost_usd
        else:
            small_text, large_text = counterfactual.text, sample.response_text
            cost_small, cost_large = counterfactual.cost_usd, sample.cost_usd
        verdict = self.judge.judge(sample.prompt, small_text, large_text)
        j = Judgment(
            sample_id=sample.id, ts=sample.ts, category=sample.category,
            router=sample.router, decision=sample.decision,
            served_tier=sample.served_tier,
            small_ok=verdict["score_small"] >= verdict["score_large"],
            score_small=verdict["score_small"], score_large=verdict["score_large"],
            cost_small=cost_small, cost_large=cost_large,
            judge_cost=verdict["judge_cost"] + counterfactual.cost_usd,
            prompt=sample.prompt,
        )
        self.store.add(j)
        if sample.decision == "small":  # guard measures the router's small calls
            self.guard.observe(j.category, j.small_ok)
        self.judged += 1
        self.judge_spend_usd += j.judge_cost
        if self.judged % self.cfg.log_every == 0:
            self.tracker.log_metrics(self.window_metrics(), step=self.judged)
        return j

    # ---- reporting ---------------------------------------------------------
    def window_metrics(self) -> Dict[str, float]:
        metrics = {"judged_total": float(self.judged), "dropped_total": float(self.dropped),
                   "judge_spend_usd": round(self.judge_spend_usd, 6)}
        for cat, s in self.guard.state().items():
            metrics[f"small_win_rate.{cat}"] = s["small_win_rate"]
            metrics[f"guard_on.{cat}"] = 1.0 if s["overridden"] else 0.0
        return metrics

    def summary(self, routers: Optional[Dict[str, object]] = None) -> Dict:
        return {
            "sample_rate": self.cfg.sample_rate,
            "judged": self.judged, "queued": self._q.qsize(), "dropped": self.dropped,
            "judge_spend_usd": round(self.judge_spend_usd, 6),
            "categories": self.guard.state(),
            "guard_events": self.guard.events[-20:],
            **({"shadow": shadow_compare(self.store.all(), routers)} if routers else {}),
        }

    def export_labels(self) -> List[Dict]:
        """Judged live requests as router training rows."""
        return [{"prompt": j.prompt, "category": j.category, "small_ok": int(j.small_ok),
                 "score_small": j.score_small, "score_large": j.score_large,
                 "cost_small": j.cost_small, "cost_large": j.cost_large, "ts": j.ts}
                for j in self.store.all()]

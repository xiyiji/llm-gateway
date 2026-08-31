"""Four generations of routing policy behind one interface.

v0_rules    keyword heuristics, the baseline everyone starts with
v1_score    weighted difficulty score with a threshold
v2_learned  logistic regression trained on measured outcomes
v3_ppo      linear policy optimized with PPO on quality-minus-cost reward

Every router answers the same question: should this prompt go to the small
model or the large one, and why.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from gateway.config import MODELS_DIR
from gateway.features import FEATURE_NAMES, featurize


class RouteDecision:
    def __init__(self, action: str, reason: str, confidence: float, router: str):
        self.action = action  # "small" | "large"
        self.reason = reason
        self.confidence = confidence
        self.router = router

    def as_dict(self) -> Dict:
        return {
            "router": self.router,
            "decision": self.action,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


class V0Rules:
    name = "v0_rules"

    def route(self, text: str) -> RouteDecision:
        _, f = featurize(text)
        signals = []
        if f["code_kw"] > 0:
            signals.append("code keywords")
        if f["reasoning_kw"] > 0:
            signals.append("reasoning keywords")
        if f["n_chars"] >= 1.2:  # ~1200 chars
            signals.append("long prompt")
        if f["multistep_kw"] >= 0.5:
            signals.append("multi-step markers")
        if signals:
            return RouteDecision("large", "matched: " + ", ".join(signals), 0.6, self.name)
        return RouteDecision("small", "no difficulty keywords matched", 0.6, self.name)


class V1Score:
    name = "v1_score"
    WEIGHTS = {
        "code_kw": 1.4, "reasoning_kw": 1.2, "multistep_kw": 1.0,
        "math_op_count": 0.7, "n_chars": 0.5, "number_count": 0.5,
        "has_code_block": 1.0, "simple_kw": -1.2, "question_marks": -0.2,
    }
    THRESHOLD = 1.0

    def route(self, text: str) -> RouteDecision:
        _, f = featurize(text)
        score = sum(w * f[k] for k, w in self.WEIGHTS.items())
        top = sorted(
            ((k, w * f[k]) for k, w in self.WEIGHTS.items() if abs(w * f[k]) > 0.05),
            key=lambda kv: -abs(kv[1]),
        )[:3]
        reason = f"difficulty score {score:.2f} vs threshold {self.THRESHOLD}; " + \
                 "drivers: " + (", ".join(f"{k}={v:+.2f}" for k, v in top) or "none")
        if score >= self.THRESHOLD:
            return RouteDecision("large", reason, min(0.5 + score / 5, 0.95), self.name)
        return RouteDecision("small", reason, min(0.5 + (self.THRESHOLD - score) / 5, 0.95), self.name)


class V2Learned:
    """Logistic regression predicting P(small model is sufficient)."""

    name = "v2_learned"

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "v2_logreg.pkl")
        self._clf = None

    def _load(self):
        if self._clf is None:
            with open(self.model_path, "rb") as fh:
                self._clf = pickle.load(fh)
        return self._clf

    def loaded(self) -> bool:
        return self.model_path.exists()

    def route(self, text: str) -> RouteDecision:
        clf = self._load()
        vec, _ = featurize(text)
        p_small_ok = float(clf.predict_proba(np.array([vec]))[0][1])
        if p_small_ok >= 0.5:
            return RouteDecision(
                "small", f"model predicts small is sufficient (p={p_small_ok:.2f})",
                p_small_ok, self.name,
            )
        return RouteDecision(
            "large", f"model predicts small would fall short (p={p_small_ok:.2f})",
            1 - p_small_ok, self.name,
        )


class V3PPO:
    """Linear softmax policy trained with PPO on reward = quality - lambda * cost."""

    name = "v3_ppo"

    def __init__(self, policy_path: Optional[Path] = None):
        self.policy_path = policy_path or (MODELS_DIR / "v3_policy.npz")
        self._params = None

    def _load(self):
        if self._params is None:
            data = np.load(self.policy_path)
            self._params = (data["W"], data["b"])
        return self._params

    def loaded(self) -> bool:
        return self.policy_path.exists()

    def route(self, text: str) -> RouteDecision:
        W, b = self._load()
        vec, _ = featurize(text)
        logits = W @ np.array(vec) + b
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()  # index 0 = small, 1 = large
        action = "small" if probs[0] >= probs[1] else "large"
        conf = float(probs.max())
        return RouteDecision(
            action, f"PPO policy π(small)={probs[0]:.2f}, π(large)={probs[1]:.2f}",
            conf, self.name,
        )


class AlwaysSmall:
    name = "always_small"

    def route(self, text: str) -> RouteDecision:
        return RouteDecision("small", "fixed baseline", 1.0, self.name)


class AlwaysLarge:
    name = "always_large"

    def route(self, text: str) -> RouteDecision:
        return RouteDecision("large", "fixed baseline", 1.0, self.name)


def build_registry() -> Dict[str, object]:
    registry: Dict[str, object] = {
        "always_small": AlwaysSmall(),
        "always_large": AlwaysLarge(),
        "v0_rules": V0Rules(),
        "v1_score": V1Score(),
    }
    v2, v3 = V2Learned(), V3PPO()
    if v2.loaded():
        registry["v2_learned"] = v2
    if v3.loaded():
        registry["v3_ppo"] = v3
    return registry

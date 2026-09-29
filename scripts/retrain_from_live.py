"""Retrain the learned router on judged live traffic, and only recommend it
if it beats the active router on newer live requests it never saw.

Labels: every judged live request says whether the small model's answer was
as good as the large model's on that exact request. Older judgments (plus the
offline benchmark labels, when var/results.json exists) train a candidate;
the newest `--holdout` fraction is kept back and every router is scored on it
with both tiers' outcomes known. Every run is logged to the tracker (MLflow
when installed) with the candidate registered as `gateway-router@candidate`.

Usage: python scripts/retrain_from_live.py [--holdout 0.3] [--promote]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.config import MODELS_DIR, VAR_DIR, get_config  # noqa: E402
from gateway.features import featurize  # noqa: E402
from gateway.online_eval import GuardedRouter, JudgmentStore, shadow_compare  # noqa: E402
from gateway.routers import AlwaysLarge, V2Learned, build_registry  # noqa: E402
from gateway.tracking import build_tracker  # noqa: E402


def offline_rows():
    path = VAR_DIR / "results.json"
    if not path.exists():
        return []
    rows = []
    for slot in json.loads(path.read_text(encoding="utf-8")).values():
        if "small" in slot and "large" in slot:
            ok = slot["small"]["quality"] >= slot["large"]["quality"] - 0.05
            rows.append((slot["prompt"], int(ok)))
    return rows


def pick_threshold(X, y, target_precision: float) -> float:
    """Lowest P(small is sufficient) cut-off whose out-of-fold precision
    reaches `target_precision`: send a request to the small model only when
    the router is as sure as the quality bar the guard enforces. Quality
    first, then as much small-model traffic as that bar allows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = max(2, min(5, int(min(y.sum(), len(y) - y.sum()))))
    probs = cross_val_predict(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced")),
        X, y, cv=folds, method="predict_proba")[:, 1]
    for t in np.round(np.arange(0.5, 0.96, 0.05), 2):
        picked = probs >= t
        if picked.sum() >= 5 and y[picked].mean() >= target_precision:
            return float(t)
    return 0.95


def decide_promotion(cand: dict, active: dict, min_gain: float = 0.02) -> tuple:
    """Promote when the candidate is at least as good for less money, or
    clearly better at similar money. Quality is never traded for cost."""
    if cand["mean_quality"] >= active["mean_quality"] and \
            cand["mean_cost_usd"] < active["mean_cost_usd"]:
        return True, "same or better quality at lower cost"
    if cand["mean_quality"] >= active["mean_quality"] + min_gain and \
            cand["mean_cost_usd"] <= active["mean_cost_usd"] * 1.10:
        return True, "better quality at similar cost"
    return False, "does not beat the active router on held-out live traffic"


def retrain(store: JudgmentStore, active_name: str, holdout: float = 0.3,
            out_path: Path = None, tracker=None, include_offline: bool = True,
            guarded_categories=(), target_precision: float = 0.95) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    judgments = store.all()  # time-ordered
    if len(judgments) < 20:
        raise SystemExit(f"only {len(judgments)} judged live requests; need at least 20")
    split = int(len(judgments) * (1 - holdout))
    train, test = judgments[:split], judgments[split:]
    rows = [(j.prompt, int(j.small_ok)) for j in train]
    if include_offline:
        rows += offline_rows()
    X = np.array([featurize(p)[0] for p, _ in rows])
    y = np.array([label for _, label in rows])
    if len(set(y)) < 2:
        raise SystemExit("labels are all one class; nothing to learn yet")
    # features live on very different scales (length ~0.03, keyword ratios
    # 0-1); standardize so regularization doesn't silently mute the small ones
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X, y)
    clf.small_threshold_ = pick_threshold(X, y, target_precision)

    out_path = out_path or (MODELS_DIR / "v2_live.pkl")
    out_path.parent.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".candidate.pkl")
    with open(tmp, "wb") as fh:
        pickle.dump(clf, fh)
    candidate = V2Learned(tmp)
    candidate.name = "v2_live_candidate"

    registry = build_registry()
    routers = {"candidate": candidate, "always_large": AlwaysLarge()}
    if active_name in registry:
        # the bar is what production serves today: the active router plus
        # whatever the quality guard is currently overriding
        routers["active"] = GuardedRouter(registry[active_name], guarded_categories)
    scores = shadow_compare(test, routers)
    active = scores.get("active", scores["always_large"])
    promote, why = decide_promotion(scores["candidate"], active)

    metrics = {f"{r}.{k}": v for r, s in scores.items() if not r.startswith("_")
               for k, v in s.items() if k != "n"}
    metrics.update({"train_rows": len(rows), "holdout_live": len(test),
                    "train_live": len(train), "promote": float(promote)})
    params = {"model": "standard_scaler+logistic_regression", "class_weight": "balanced",
              "small_threshold": clf.small_threshold_, "target_precision": target_precision,
              "holdout": holdout, "active_router": active_name,
              "guarded_categories": ",".join(sorted(guarded_categories)) or "none",
              "include_offline": include_offline}
    tracker = tracker or build_tracker(get_config().online_eval)
    run_id = tracker.log_router_run("v2_live", params, metrics, tmp, register=promote)

    if promote:
        tmp.replace(out_path)
    return {"scores": scores, "promote": promote, "reason": why, "run": run_id,
            "model_path": str(out_path if promote else tmp)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--guarded", default="",
                        help="comma-separated categories the guard currently overrides "
                             "(see GET /admin/online-eval)")
    args = parser.parse_args()
    result = retrain(JudgmentStore(), get_config().router.active, args.holdout,
                     include_offline=not args.no_offline,
                     guarded_categories=[c for c in args.guarded.split(",") if c])
    print(json.dumps(result, indent=2))
    if result["promote"]:
        print("candidate saved as models/v2_live.pkl; activate with "
              "POST /admin/routers/v2_live after a restart")


if __name__ == "__main__":
    main()

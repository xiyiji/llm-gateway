"""Train the learned (v2) and PPO (v3) routers from the offline results cache.

v2: logistic regression predicting "is the small model sufficient here",
    label = quality_small >= quality_large - 0.05.
v3: PPO on reward = quality - lambda * scaled cost, optimizing the business
    objective directly rather than imitating labels.

Usage: python scripts/train_routers.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.config import MODELS_DIR, get_config  # noqa: E402
from gateway.features import featurize  # noqa: E402
from gateway.policy import train_ppo  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "var" / "results.json"
COST_SCALE = 1000.0  # dollars are tiny per call; scale so lambda is a sane knob


def load_dataset(lam=None):
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    X, labels, rewards, ids = [], [], [], []
    lam = get_config().router.quality_cost_lambda if lam is None else lam
    for task_id, slot in sorted(results.items()):
        if "small" not in slot or "large" not in slot:
            continue
        vec, _ = featurize(slot["prompt"])
        qs, ql = slot["small"]["quality"], slot["large"]["quality"]
        cs, cl = slot["small"]["cost_usd"], slot["large"]["cost_usd"]
        X.append(vec)
        labels.append(1 if qs >= ql - 0.05 else 0)  # 1 = small sufficient
        rewards.append([qs - lam * cs * COST_SCALE / 1000.0 * 10,
                       ql - lam * cl * COST_SCALE / 1000.0 * 10])
        ids.append(task_id)
    return np.array(X), np.array(labels), np.array(rewards), ids


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=None,
                        help="cost-aversion dial; overrides config")
    parser.add_argument("--dry-run", action="store_true",
                        help="train and report but do not save models")
    args = parser.parse_args()

    X, labels, rewards, ids = load_dataset(args.lam)
    n = len(X)
    rng = np.random.default_rng(3)
    order = rng.permutation(n)
    split = int(n * 0.75)
    train_idx, test_idx = order[:split], order[split:]
    print(f"dataset: {n} prompts, {labels.mean():.0%} small-sufficient")

    # ---- v2: logistic regression ------------------------------------------
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[train_idx], labels[train_idx])
    train_acc = clf.score(X[train_idx], labels[train_idx])
    test_acc = clf.score(X[test_idx], labels[test_idx])
    MODELS_DIR.mkdir(exist_ok=True)
    if not args.dry_run:
        with open(MODELS_DIR / "v2_logreg.pkl", "wb") as fh:
            pickle.dump(clf, fh)
    print(f"v2 logistic regression: train acc {train_acc:.2%}, test acc {test_acc:.2%}")

    # ---- v3: PPO policy ----------------------------------------------------
    W, b, info = train_ppo(X[train_idx], rewards[train_idx])
    if not args.dry_run:
        np.savez(MODELS_DIR / "v3_policy.npz", W=W, b=b)
    print("v3 PPO:", json.dumps(info, indent=2))

    # held-out reward comparison
    from gateway.policy import softmax

    greedy = softmax(X[test_idx] @ W.T + b).argmax(axis=1)
    ppo_reward = rewards[test_idx][np.arange(len(test_idx)), greedy].mean()
    static_small = rewards[test_idx][:, 0].mean()
    static_large = rewards[test_idx][:, 1].mean()
    oracle = rewards[test_idx].max(axis=1).mean()
    print(f"held-out mean reward: ppo={ppo_reward:.4f} "
          f"always_small={static_small:.4f} always_large={static_large:.4f} oracle={oracle:.4f}")


if __name__ == "__main__":
    main()

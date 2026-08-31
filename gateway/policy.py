"""PPO training for the routing policy.

The router is a two-action contextual bandit: given prompt features, pick the
small or large model. Reward is quality minus a cost penalty, so the policy
optimizes the business objective directly instead of imitating labels.

Policy and value function are linear (a few hundred parameters); training
runs in seconds on a CPU against the offline results cache, so iteration is
free once model outputs have been collected.
"""

from typing import Dict, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train_ppo(
    X: np.ndarray,            # (N, D) prompt features
    rewards: np.ndarray,      # (N, 2) reward for [small, large] per prompt
    epochs: int = 300,
    inner_epochs: int = 4,
    lr: float = 0.08,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Returns (W, b, training_info). Actions: 0 = small, 1 = large."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((2, d))
    b = np.zeros(2)
    v_w = np.zeros(d)   # linear value baseline
    v_b = 0.0

    history = []
    for _ in range(epochs):
        # rollout: sample one action per prompt from the current policy
        probs = softmax(X @ W.T + b)                       # (N, 2)
        actions = (rng.random(n) < probs[:, 1]).astype(int)  # sample via P(large)
        r = rewards[np.arange(n), actions]                  # observed reward

        # value baseline and advantage
        values = X @ v_w + v_b
        adv = r - values
        adv_std = adv.std() + 1e-8
        adv_norm = (adv - adv.mean()) / adv_std

        old_probs = probs[np.arange(n), actions].copy()

        # PPO inner updates with clipped surrogate
        for _ in range(inner_epochs):
            probs_new = softmax(X @ W.T + b)
            pi_a = probs_new[np.arange(n), actions]
            ratio = pi_a / old_probs
            clipped = np.clip(ratio, 1 - clip, 1 + clip)
            use_unclipped = (ratio * adv_norm) <= (clipped * adv_norm)
            # gradient of log pi(a|x) for linear softmax: (onehot - probs) x
            onehot = np.eye(2)[actions]
            coef = np.where(use_unclipped, ratio, 0.0) * adv_norm  # zero grad when clipped
            grad_logits = (onehot - probs_new) * coef[:, None]
            # entropy bonus keeps the policy from collapsing early
            ent_grad = -(np.log(probs_new + 1e-9) + 1.0) * entropy_coef
            grad_logits += (ent_grad - (ent_grad * probs_new).sum(axis=1, keepdims=True)) * probs_new
            W += lr * (grad_logits.T @ X) / n
            b += lr * grad_logits.mean(axis=0)

        # value regression step
        v_err = r - (X @ v_w + v_b)
        v_w += 0.05 * (v_err @ X) / n
        v_b += 0.05 * v_err.mean()

        history.append(float(r.mean()))

    final_probs = softmax(X @ W.T + b)
    greedy = final_probs.argmax(axis=1)
    greedy_reward = rewards[np.arange(n), greedy].mean()
    info = {
        "epochs": epochs,
        "mean_sampled_reward_first": round(history[0], 4),
        "mean_sampled_reward_last": round(history[-1], 4),
        "greedy_reward": round(float(greedy_reward), 4),
        "best_static_reward": round(float(max(rewards[:, 0].mean(), rewards[:, 1].mean())), 4),
        "oracle_reward": round(float(rewards.max(axis=1).mean()), 4),
        "share_routed_large": round(float(greedy.mean()), 4),
    }
    return W, b, info

"""Offline tests: features, routers, PPO trainer, verification."""

import numpy as np

from evalsuite.tasks import all_tasks, check_math, math_tasks
from evalsuite.verify import extract_code, run_code_tests
from gateway.features import FEATURE_NAMES, featurize
from gateway.policy import softmax, train_ppo
from gateway.routers import V0Rules, V1Score, build_registry


def test_feature_vector_shape_and_signal():
    vec, named = featurize("Implement a python function to parse SQL")
    assert len(vec) == len(FEATURE_NAMES)
    assert named["code_kw"] > 0
    vec2, named2 = featurize("What is the capital of France?")
    assert named2["simple_kw"] > 0 and named2["code_kw"] == 0


def test_v0_and_v1_route_sanity():
    for router in (V0Rules(), V1Score()):
        assert router.route("What is the capital of France?").action == "small"
        decision = router.route(
            "Implement a python function with unit tests to parse nested JSON, "
            "then refactor it step by step and explain the algorithm."
        )
        assert decision.action == "large"
        assert decision.reason


def test_registry_skips_untrained_models(tmp_path, monkeypatch):
    import gateway.routers as routers_mod

    monkeypatch.setattr(routers_mod, "MODELS_DIR", tmp_path)
    registry = build_registry()
    assert "v0_rules" in registry and "v1_score" in registry


def test_ppo_learns_a_separable_policy():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 4))
    # action 1 (large) is better when x0 > 0, else action 0
    rewards = np.zeros((400, 2))
    rewards[:, 0] = np.where(X[:, 0] <= 0, 1.0, 0.0)
    rewards[:, 1] = np.where(X[:, 0] > 0, 1.0, 0.0)
    W, b, info = train_ppo(X, rewards, epochs=200)
    greedy = softmax(X @ W.T + b).argmax(axis=1)
    accuracy = (greedy == (X[:, 0] > 0)).mean()
    assert accuracy > 0.85
    assert info["greedy_reward"] > info["best_static_reward"]


def test_math_checker_accepts_exact_answers():
    for task in math_tasks():
        assert check_math(task, f"The answer is {task['answer']}") == 1.0
    assert check_math(math_tasks()[0], "no idea") == 0.0


def test_code_sandbox_pass_and_fail():
    task = {
        "id": "t", "kind": "code",
        "test": "assert solution.double(2) == 4",
    }
    good = "```python\ndef double(x):\n    return x * 2\n```"
    bad = "def double(x):\n    return x + 1"
    assert run_code_tests(task, good) == 1.0
    assert run_code_tests(task, bad) == 0.0
    assert extract_code(good).strip().startswith("def double")


def test_task_suite_shape():
    tasks = all_tasks()
    kinds = {t["kind"] for t in tasks}
    assert kinds == {"math", "code", "qa"}
    assert len(tasks) >= 35
    assert len({t["id"] for t in tasks}) == len(tasks)

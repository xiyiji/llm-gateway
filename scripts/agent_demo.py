"""Agent workload demo: the same multi-step agent, direct-to-large vs gateway-routed.

The agent solves small research-and-compute tasks in three LLM steps:
plan -> extract numbers and compute (with a calculator tool) -> summarize.
Each step is an LLM call. Run once with every call pinned to the large model,
once with the gateway's router deciding per step, and compare cost, latency,
and whether the final numeric answer is correct.

Usage: DEEPSEEK_API_KEY=... python scripts/agent_demo.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.providers import DeepSeekProvider  # noqa: E402
from gateway.routers import V1Score  # noqa: E402

TASKS = [
    {
        "brief": "A customer ordered 3 chairs at $47.50 each and 2 tables at $129 each. "
                 "Shipping is 8% of the merchandise total.",
        "question": "What is the grand total including shipping?",
        "answer": round((3 * 47.5 + 2 * 129) * 1.08, 2),
    },
    {
        "brief": "A warehouse processed 1240 packages on Monday, 15% more on Tuesday, "
                 "and on Wednesday 200 fewer than Tuesday.",
        "question": "How many packages were processed across the three days?",
        "answer": 1240 + 1240 * 1.15 + (1240 * 1.15 - 200),
    },
    {
        "brief": "A driver's route is 180 km. The first half is driven at 60 km/h, "
                 "the second half at 90 km/h.",
        "question": "How many minutes does the full route take?",
        "answer": (90 / 60 + 90 / 90) * 60,
    },
    {
        "brief": "A subscription costs $24/month. An annual plan costs $216 upfront.",
        "question": "How many dollars are saved per year with the annual plan?",
        "answer": 24 * 12 - 216,
    },
    {
        "brief": "A tank holds 500 liters and is 40% full. A pump adds 25 liters per minute.",
        "question": "How many minutes until the tank is full?",
        "answer": (500 * 0.6) / 25,
    },
]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def calculator(expression: str) -> str:
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return "error: only arithmetic allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - charset-restricted arithmetic
    except Exception as exc:
        return f"error: {exc}"


def run_agent(provider: DeepSeekProvider, task: dict, mode: str, router: V1Score) -> dict:
    total_cost, total_latency, steps = 0.0, 0, []

    def call(prompt: str) -> str:
        nonlocal total_cost, total_latency
        tier = "large" if mode == "direct_large" else router.route(prompt).action
        result = provider.chat(tier, [{"role": "user", "content": prompt}])
        total_cost += result.cost_usd
        total_latency += result.latency_ms
        steps.append(tier)
        return result.text

    plan = call(f"Task: {task['brief']} Question: {task['question']} "
                "List the calculation steps needed, briefly, no final answer.")
    expr = call(f"Task: {task['brief']} Question: {task['question']} Plan: {plan[:400]} "
                "Write ONE arithmetic expression using only numbers and + - * / ( ) "
                "that computes the answer. Reply with just the expression.")
    computed = calculator(expr.strip().strip("`"))
    summary = call(f"The computed result is {computed}. Question was: {task['question']} "
                   "State the final numeric answer. Reply with just the number.")

    numbers = NUM_RE.findall(summary.replace(",", ""))
    final = float(numbers[-1]) if numbers else float("nan")
    correct = abs(final - task["answer"]) < 0.51
    return {"correct": correct, "cost_usd": round(total_cost, 6),
            "latency_ms": total_latency, "steps": steps}


def main() -> None:
    provider = DeepSeekProvider()
    ok, reason = provider.available()
    if not ok:
        raise SystemExit(f"cannot run: {reason}")
    router = V1Score()

    report = {}
    for mode in ("direct_large", "gateway_routed"):
        runs = [run_agent(provider, task, mode, router) for task in TASKS]
        report[mode] = {
            "tasks": len(runs),
            "correct": sum(r["correct"] for r in runs),
            "total_cost_usd": round(sum(r["cost_usd"] for r in runs), 5),
            "total_latency_s": round(sum(r["latency_ms"] for r in runs) / 1000, 1),
            "large_calls": sum(r["steps"].count("large") for r in runs),
            "small_calls": sum(r["steps"].count("small") for r in runs),
        }

    direct, routed = report["direct_large"], report["gateway_routed"]
    report["savings"] = {
        "cost_reduction_pct": round((1 - routed["total_cost_usd"] / direct["total_cost_usd"]) * 100, 1)
        if direct["total_cost_usd"] else 0.0,
        "latency_reduction_pct": round((1 - routed["total_latency_s"] / direct["total_latency_s"]) * 100, 1)
        if direct["total_latency_s"] else 0.0,
        "accuracy_delta": routed["correct"] - direct["correct"],
    }
    out = Path(__file__).resolve().parents[1] / "var" / "agent_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

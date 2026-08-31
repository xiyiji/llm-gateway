"""Collect the offline results cache: every eval task against both models.

This is the one step that costs real money (a few cents). Everything after
it, training the learned and PPO routers and benchmarking all router
generations, iterates on this cache for free. The script is resumable: tasks
already in results.json are skipped.

Usage: DEEPSEEK_API_KEY=... python scripts/collect_results.py [--workers 6]
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalsuite.tasks import all_tasks  # noqa: E402
from evalsuite.verify import judge_qa, quality_of  # noqa: E402
from gateway.providers import DeepSeekProvider  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "var" / "results.json"
_lock = threading.Lock()


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_one(provider: DeepSeekProvider, task: dict, tier: str) -> dict:
    result = provider.chat(tier, [{"role": "user", "content": task["prompt"]}])
    entry = {
        "response": result.text,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    if task["kind"] == "qa":
        verdict = judge_qa(task, result.text, provider)
        entry["quality"] = (verdict["score"] - 1) / 4.0
        entry["judge_cost_usd"] = verdict["judge_cost_usd"]
    else:
        entry["quality"] = quality_of(task, result.text)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    provider = DeepSeekProvider()
    ok, reason = provider.available()
    if not ok:
        raise SystemExit(f"cannot collect: {reason}")

    tasks = all_tasks()
    results = load_results()
    todo = []
    for task in tasks:
        for tier in ("small", "large"):
            if results.get(task["id"], {}).get(tier) is None:
                todo.append((task, tier))
    print(f"{len(tasks)} tasks, {len(todo)} model runs to collect")

    def work(item):
        task, tier = item
        return task, tier, run_one(provider, task, tier)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for future in as_completed(futures):
            task, tier, entry = future.result()
            with _lock:
                slot = results.setdefault(task["id"], {"kind": task["kind"],
                                                       "difficulty": task["difficulty"],
                                                       "prompt": task["prompt"]})
                slot[tier] = entry
                done += 1
                if done % 10 == 0:
                    save_results(results)
                    print(f"  {done}/{len(todo)} collected")
    save_results(results)

    total_cost = sum(
        slot[tier]["cost_usd"] + slot[tier].get("judge_cost_usd", 0.0)
        for slot in results.values() for tier in ("small", "large") if tier in slot
    )
    small_q = [s["small"]["quality"] for s in results.values() if "small" in s]
    large_q = [s["large"]["quality"] for s in results.values() if "large" in s]
    print(f"done. collection cost: ${total_cost:.4f}")
    print(f"avg quality small={sum(small_q)/len(small_q):.3f} large={sum(large_q)/len(large_q):.3f}")


if __name__ == "__main__":
    main()

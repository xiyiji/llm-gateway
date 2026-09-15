"""Convert the existing eval suite into a frozen, deterministic manifest.

This is the bridge used for the pilot. Public dataset adapters can emit the
same BenchmarkTask schema without changing the collector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List

from evalsuite.tasks import all_tasks
from research.schema import BenchmarkTask


def stable_split(task_id: str, seed: int = 20260915) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def existing_tasks(seed: int = 20260915) -> List[BenchmarkTask]:
    result: List[BenchmarkTask] = []
    for task in all_tasks():
        kind = task["kind"]
        if kind == "math":
            verifier, reference = "exact_math", task.get("answer")
        elif kind == "code":
            verifier, reference = "python_tests", task.get("test")
        else:
            verifier, reference = "llm_judge", task.get("reference")
        result.append(
            BenchmarkTask(
                task_id=task["id"],
                dataset="llm-gateway-original",
                category=kind,
                difficulty=task.get("difficulty", "unknown"),
                split=stable_split(task["id"], seed),
                prompt=task["prompt"],
                verifier=verifier,
                reference=reference,
                metadata={"source": "evalsuite.tasks"},
            )
        )
    return result


def write_jsonl(tasks: Iterable[BenchmarkTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="research/manifests/pilot.jsonl")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    tasks = existing_tasks(args.seed)
    write_jsonl(tasks, Path(args.out))
    counts = {}
    for task in tasks:
        counts[(task.category, task.split)] = counts.get((task.category, task.split), 0) + 1
    print(f"wrote {len(tasks)} tasks to {args.out}")
    for key in sorted(counts):
        print(key, counts[key])


if __name__ == "__main__":
    main()

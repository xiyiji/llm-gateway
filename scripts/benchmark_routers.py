"""Benchmark every router generation against the offline results cache.

For each router, replay every eval prompt through its route() decision and
look up the measured quality/cost/latency of the model it picked. Output: the
headline table, quality retained vs cost spent relative to always-large.

Usage: python scripts/benchmark_routers.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.routers import build_registry  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "var" / "results.json"
BENCH_PATH = Path(__file__).resolve().parents[1] / "var" / "benchmarks.json"


def table_for(slots, registry, label):
    large_quality = sum(s["large"]["quality"] for s in slots) / len(slots)
    large_cost = sum(s["large"]["cost_usd"] for s in slots)
    rows = []
    for name in ["always_small", "always_large", "v0_rules", "v1_score", "v2_learned", "v3_ppo"]:
        router = registry.get(name)
        if router is None:
            continue
        quality = cost = latency = to_large = 0.0
        for slot in slots:
            tier = router.route(slot["prompt"]).action
            entry = slot[tier]
            quality += entry["quality"]
            cost += entry["cost_usd"]
            latency += entry["latency_ms"]
            to_large += tier == "large"
        n = len(slots)
        rows.append({
            "set": label,
            "router": name,
            "avg_quality": round(quality / n, 4),
            "quality_retention_pct": round(quality / n / large_quality * 100, 1),
            "total_cost_usd": round(cost, 4),
            "cost_vs_large_pct": round(cost / large_cost * 100, 1),
            "avg_latency_ms": round(latency / n, 0),
            "share_routed_large_pct": round(to_large / n * 100, 1),
        })
    return rows


def print_table(rows):
    header = ["router", "avg_quality", "quality_retention_pct", "total_cost_usd",
              "cost_vs_large_pct", "avg_latency_ms", "share_routed_large_pct"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in header) + " |")


def main() -> None:
    import numpy as np

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    items = sorted(results.items())
    slots = [s for _, s in items if "small" in s and "large" in s]
    registry = build_registry()

    # same split as scripts/train_routers.py, so held-out really is held out
    rng = np.random.default_rng(3)
    order = rng.permutation(len(slots))
    test_idx = set(order[int(len(slots) * 0.75):].tolist())
    heldout = [s for i, s in enumerate(slots) if i in test_idx]

    full_rows = table_for(slots, registry, "full")
    heldout_rows = table_for(heldout, registry, "heldout")
    BENCH_PATH.write_text(json.dumps(full_rows + heldout_rows, indent=2), encoding="utf-8")

    print(f"=== full suite ({len(slots)} prompts; v2/v3 saw 75% of these in training) ===")
    print_table(full_rows)
    print()
    print(f"=== held-out only ({len(heldout)} prompts never seen by v2/v3) ===")
    print_table(heldout_rows)


if __name__ == "__main__":
    main()

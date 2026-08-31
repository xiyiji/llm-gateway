"""Async load test: fixed concurrency, reports RPS and latency percentiles.

Usage:
  python scripts/load_test.py --url http://localhost:8080/route --concurrency 50 --requests 2000
"""

import argparse
import asyncio
import json
import random
import time

import httpx

QUERIES = [
    "hello, what is a cache hit rate?",
    "write a python function to parse json",
    "analyze tradeoffs of caching vs compression",
    "what is the capital of france?",
    "debug this class method for me",
    "explain why the sky is blue step by step",
]


async def worker(client, url, method, n, latencies, errors, unique):
    for i in range(n):
        if method == "POST":
            query = random.choice(QUERIES)
            if unique:
                query = f"{query} #{random.randint(0, 10**9)}"
            payload = {"query": query, "user_id": f"load-{i}", "user_tier": random.choice(["free", "premium"])}
            start = time.perf_counter()
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except Exception:
                errors.append(1)
                continue
        else:
            start = time.perf_counter()
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception:
                errors.append(1)
                continue
        latencies.append((time.perf_counter() - start) * 1000)


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--method", default="POST", choices=["GET", "POST"])
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--unique", action="store_true", help="make every query unique (defeat cache)")
    args = parser.parse_args()

    per_worker = args.requests // args.concurrency
    latencies, errors = [], []
    async with httpx.AsyncClient(timeout=30) as client:
        start = time.perf_counter()
        await asyncio.gather(*[
            worker(client, args.url, args.method, per_worker, latencies, errors, args.unique)
            for _ in range(args.concurrency)
        ])
        wall = time.perf_counter() - start

    latencies.sort()
    total = len(latencies)
    print(json.dumps({
        "url": args.url,
        "concurrency": args.concurrency,
        "completed": total,
        "errors": len(errors),
        "wall_seconds": round(wall, 2),
        "rps": round(total / wall, 1),
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "max_ms": round(percentile(latencies, 1.0), 1),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

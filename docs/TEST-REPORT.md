# Test report

Date: 2026-08-30. Environment: macOS (M-series), Python 3.14 venv, single
uvicorn worker. Models: deepseek-chat (small), deepseek-reasoner (large),
live DeepSeek API. All numbers below are measured, not estimated.

## Unit and integration tests

20 tests, all passing, run offline with scripted providers (CI runs them on
every push):

- features: vector shape, keyword signals
- routers: v0/v1 routing sanity, registry behavior for untrained models
- PPO: learns a separable toy policy to >85% accuracy and beats the best
  static baseline on reward
- verification: math answer parsing, code sandbox pass/fail, code-block
  extraction
- API: OpenAI response shape, auto vs explicit model, fallback on provider
  failure, validation errors, admin metrics/routers, landing page, health
- streaming: SSE chunk format, pre-first-byte fallback, cache population
  and cached replay, usage/cost recording
- auth: 401 unknown key, per-key spend tracking, 429 on exhausted budget

## Router benchmark

Eval suite: 46 prompts across math (26, exact-match), code (10, unit tests
executed in a sandbox), QA (10, judged by the large model). One collection
run against both models cost $0.046; all training and benchmarking below
replays that offline cache at zero cost.

Full suite (v2/v3 saw 75% of these prompts during training):

| router | avg_quality | quality kept | total cost | cost vs large | routed large |
|---|---|---|---|---|---|
| always_small | 0.870 | 90.9% | $0.0099 | 44.6% | 0% |
| always_large | 0.957 | 100% | $0.0222 | 100% | 100% |
| v0_rules | 0.848 | 88.6% | $0.0180 | 80.8% | 30.4% |
| v1_score | 0.870 | 90.9% | $0.0117 | 52.7% | 8.7% |
| v2_learned | 0.978 | 102.3% | $0.0121 | 54.6% | 43.5% |
| v3_ppo | 0.978 | 102.3% | $0.0177 | 79.6% | 82.6% |

Held-out only (12 prompts never seen by the trained routers):

| router | avg_quality | quality kept | cost vs large | routed large |
|---|---|---|---|---|
| always_small | 0.917 | 91.7% | 36.9% | 0% |
| always_large | 1.000 | 100% | 100% | 100% |
| v0_rules | 0.917 | 91.7% | 77.5% | 33.3% |
| v1_score | 0.917 | 91.7% | 53.1% | 16.7% |
| v2_learned | 1.000 | 100% | 48.0% | 41.7% |
| v3_ppo | 1.000 | 100% | 71.0% | 75.0% |

Retention above 100% is real: on several prompts the small model succeeds
where the large reasoning model fails (typically strict output formatting on
easy tasks), so a good router can beat both fixed choices.

## PPO training and the lambda dial

Training: linear policy + linear value baseline, clipped surrogate
(epsilon 0.2), 300 epochs, seconds on CPU.

| lambda (cost aversion) | routed large (train greedy) | held-out reward vs baselines |
|---|---|---|
| 2 (quality-leaning) | 85% | ppo 0.995 > large 0.993 > small 0.914 |
| 50 (frugal) | 65% | ppo 0.910 > small 0.847 > large 0.811 |

At both settings the learned policy beats both fixed baselines on the
objective it was trained for.

## Agent workload (live API)

5 multi-step tasks, 3 LLM calls each (plan, compute with a calculator tool,
summarize):

| mode | correct | total cost | total latency | large/small calls |
|---|---|---|---|---|
| every call large | 5/5 | $0.00568 | 27.4s | 15 / 0 |
| gateway-routed | 5/5 | $0.00189 (-66.7%) | 16.0s (-41.6%) | 6 / 9 |

## Gateway throughput (cache-hit path)

`scripts/load_test.py`, 2000 requests at 20 concurrency against a primed
cache (this isolates gateway machinery: routing, cache, accounting, HTTP):

| metric | value |
|---|---|
| RPS | 803 |
| P50 / P95 / P99 | 12.5 / 80.0 / 146.1 ms |
| errors | 0 |
| cache hit rate during run | 99.7% (2000 hits, 6 misses) |

## Live smoke checks

- Auto routing served an easy prompt and a hard prompt with the routing
  verdict, cost and latency attached to each response.
- Repeating a request returned `cached: true, cost_usd: 0, latency 0ms`.
- Streaming returned 10 SSE chunks for a counting prompt; cost $0.000014
  was recorded from the final usage chunk.
- With no API key the service reports `provider_available: false` on
  /healthz and returns a structured 502 on completions.

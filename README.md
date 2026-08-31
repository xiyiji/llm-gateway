# LLM Gateway

One OpenAI-compatible endpoint that sends every request to the cheapest model
that can handle it, proves the quality held up, and shows you the bill.

## The problem this solves

Every company that adopts LLMs at scale runs into the same three walls within
months.

The bill. An AI feature serving a million requests a day on a frontier model
costs five figures daily, and most of that traffic is questions a model
twenty times cheaper answers just as well. Everyone knows this. Nobody acts
on it, because of the second wall.

Nobody can prove quality won't drop. An engineer switches some traffic to
the cheap model, support tickets maybe tick up, maybe don't, there's no
measurement, everyone gets nervous, traffic goes back to the expensive
model. This is the actual blocker: not the routing, the proving.

The vendor line. Model names are hard-coded in forty files. When the
provider has a bad day, the product has a bad day. Switching providers is a
migration project instead of a config change.

This isn't a hypothetical market. OpenRouter, LiteLLM, Martian, and
Cloudflare AI Gateway all exist to sell or open-source exactly this layer,
and ChatGPT itself now routes requests across models under the hood. Teams
that build this in-house call the role AI platform engineering.

## Who uses a gateway

- A support bot where "how do I reset my password" should not cost the same
  as "walk me through migrating my data warehouse"
- An internal copilot whose finance team wants cost per team and per feature
- Agent systems, the heaviest traffic of all: an agent makes many model
  calls per task, and pinning every step to the big model burns money on
  steps as simple as "extract the number"

## What this gateway does

Point any OpenAI SDK at it and ask for model "auto":

```python
client = OpenAI(base_url="http://localhost:8020/v1", api_key="anything")
reply = client.chat.completions.create(model="auto", messages=[...])
```

Per request it: picks small or large via the active routing policy, falls
back to the other tier if the call fails, serves repeats from an LRU+TTL
cache, prices the call from actual token usage, and returns the routing
verdict in a `gateway` field on the response so every decision is
inspectable. `/admin/metrics` aggregates spend by model, decision and
router; `/admin/routers` hot-swaps the routing policy at runtime.

## Four generations of routing policy

The interesting question is the router. This repo contains its whole
evolution, all behind one interface, all benchmarkable against each other:

**v0 rules.** Keyword heuristics: code words or long prompts go large.
Everyone's first router. Cheap, explainable, wrong in both directions.

**v1 score.** Sixteen hand-crafted, named features (length, math operators,
code markers, multi-step language...) combined with hand-tuned weights into
a difficulty score. Better, still guessing.

**v2 learned.** Stop guessing, measure. Run the eval suite through both
models once, label each prompt "was the small model sufficient", train a
logistic regression on the same features. The router now encodes where each
model actually fails, not where a human assumes it fails.

**v3 PPO.** Stop imitating labels, optimize the objective. The router is a
two-action policy; reward is quality minus lambda times cost; PPO with a
clipped surrogate trains it directly against the measured outcomes. Lambda
is the business dial: at lambda=2 the trained policy buys quality and routes
85% of traffic large; retrained at lambda=50 it turns frugal, routes 65%
less traffic large, and still beats both fixed baselines on reward. Policy
and value function are linear, training takes seconds on a CPU, and after
one data collection run costing $0.05, every retrain is free.

## Measured results

The eval suite spans three verification regimes on purpose: math (exact
answer), code (unit tests executed in a sandbox), open QA (LLM judge), so
routing is scored under weak and strong verification alike. Models:
deepseek-chat (small) vs deepseek-reasoner (large). Collection cost: $0.046.

Held-out prompts only (never seen by the trained routers):

| router | quality retained | cost vs always-large | routed large |
|---|---|---|---|
| always_small | 91.7% | 37% | 0% |
| always_large | 100% | 100% | 100% |
| v0_rules | 91.7% | 78% | 33% |
| v1_score | 91.7% | 53% | 17% |
| v2_learned | **100%** | **48%** | 42% |
| v3_ppo | **100%** | 71% | 75% |

The learned router kept full large-model quality at less than half the
cost. A finding worth being honest about: its decisions don't always match
intuition (it sometimes sends an easy factual question large), because it
learned where each model measurably fails, and the reasoner in this suite
sometimes stumbles on strictly formatted easy answers. With 46 prompts it
also inherits dataset quirks; the fix is more data, and the architecture is
built to collect it.

Agent workload (5 multi-step tasks, 3 LLM calls each, live API):

| mode | correct | cost | latency |
|---|---|---|---|
| every call to large | 5/5 | $0.0057 | 27.4s |
| gateway-routed | 5/5 | **$0.0019 (-67%)** | **16.0s (-42%)** |

Same accuracy, two-thirds cheaper, forty percent faster, because 9 of 15
agent steps didn't need the big model.

## Architecture

```
client (any OpenAI SDK, model="auto")
   |
   v
gateway  --- /v1/chat/completions (OpenAI-compatible)
   |\--- router registry: v0 rules | v1 score | v2 learned | v3 PPO
   |\--- fallback: chosen tier fails -> other tier, errors attached
   |\--- cache: LRU+TTL on full request, real hit counters
   |\--- usage store: per-request cost from actual token counts (SQLite)
   v
DeepSeek API (deepseek-chat / deepseek-reasoner)

offline loop: evalsuite -> collect_results (once, $0.05)
              -> train_routers (v2 + v3, seconds, free)
              -> benchmark_routers (all generations, free)
```

Repo map: `gateway/` (api, providers, routers, features, PPO, metrics),
`evalsuite/` (tasks, verification), `scripts/` (collect, train, benchmark,
agent demo, load test), `console.py` (Streamlit: traffic, benchmark, agent
demo), `tests/` (14 offline tests, CI on push).

## Run it

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest -q                       # offline, no key needed

export DEEPSEEK_API_KEY=sk-...
./venv/bin/python main.py                            # gateway on :8020

# rebuild the numbers yourself
./venv/bin/python scripts/collect_results.py         # ~$0.05, resumable
./venv/bin/python scripts/train_routers.py           # v2 + v3, CPU, seconds
./venv/bin/python scripts/benchmark_routers.py       # the tables above
./venv/bin/python scripts/agent_demo.py              # the agent comparison
```

Switch policies live: `POST /admin/routers/v2_learned`.

## Honest limits

Streaming responses are not implemented yet, which matters for chat UX.
The eval suite is 46 prompts; strong enough to demonstrate the method,
too small to certify a production router, and the learned policies carry
dataset quirks accordingly. One provider family is wired up (DeepSeek's
OpenAI-compatible API); adding OpenAI or Anthropic tiers is a provider
subclass plus config. Auth and per-client budgets are not built.

Stack: Python, FastAPI, numpy (PPO implemented from scratch), scikit-learn,
SQLite, Streamlit, pytest, GitHub Actions, Docker.

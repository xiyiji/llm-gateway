# Deployment guide

## What you need

- The repo (this one)
- A DeepSeek API key (platform.deepseek.com), the only secret in the system
- A Render account for the API (free tier works)
- Optionally a Streamlit Community Cloud account for the console

The gateway boots without the key and answers /healthz with
`provider_available: false`; completions return 502 until the key is set.
So the key is required for serving traffic, not for the deploy itself.

## Option A: Render (recommended)

1. Log into dashboard.render.com with GitHub.
2. New -> Blueprint -> pick this repo -> Apply. `render.yaml` builds from
   `requirements.txt` and starts uvicorn on `$PORT` with health checks on
   `/healthz`.
3. When prompted (the env var is marked `sync: false`), paste
   `DEEPSEEK_API_KEY`. It can also be added later under the service's
   Environment tab; saving restarts the service.
4. Note the public URL, e.g. `https://llm-gateway-xxxx.onrender.com`.

The trained router files in `models/` (v2_logreg.pkl, v3_policy.npz) ship
with the repo, so v2_learned and v3_ppo are available immediately.

## Option B: Docker

```bash
docker build -t llm-gateway .
docker run -p 8020:8020 -e DEEPSEEK_API_KEY=sk-... llm-gateway
```

## Option C: bare metal

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...
./venv/bin/python main.py        # :8020
```

## Post-deploy verification

```bash
BASE=https://your-url

curl -s $BASE/healthz
# expect: provider_available true, routers_loaded includes v2_learned, v3_ppo

curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is the capital of Japan?"}]}'
# expect: an answer plus a "gateway" block with router/decision/cost_usd

curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Count to 3"}]}'
# expect: SSE chunks ending with data: [DONE]

curl -s $BASE/admin/metrics
# expect: requests > 0, per-model cost split

curl -s -X POST $BASE/admin/routers/v2_learned
# expect: {"active":"v2_learned"}
```

Repeat the first completion twice: the second response shows
`"cached": true, "cost_usd": 0`.

## Console (optional)

share.streamlit.io -> Create app -> this repo -> main file `console.py` ->
Secrets: `GATEWAY_API = "https://your-render-url"`. The Traffic page reads
the live gateway; the benchmark and agent pages read local artifacts in
`var/`, so they render on the machine where the scripts were run.

## Enabling client auth

In `config.yaml`:

```yaml
auth:
  enabled: true
  keys:
    sk-team-a: {name: "team-a", daily_budget_usd: 5.0}
```

Clients then send `Authorization: Bearer sk-team-a`. Unknown keys get 401,
an exhausted daily budget gets 429, and `/admin/metrics` reports spend per
key. For a production deployment, put the admin endpoints behind the same
check or a reverse-proxy rule.

## Operating notes

- Free-tier Render sleeps when idle; first request after a while takes ~30s.
- Every push to main redeploys automatically.
- Retraining cycle after collecting more traffic:
  `scripts/collect_results.py` -> `scripts/train_routers.py` ->
  `scripts/benchmark_routers.py`, then commit the updated `models/` files.
- Router selection is runtime state; it resets to `config.yaml`'s
  `router.active` on restart.
- Expected spend: the demo eval collection is about $0.05; routine traffic
  costs fractions of a cent per uncached request and $0 per cache hit.

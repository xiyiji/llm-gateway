# Large-Scale LLM Routing Study

## Research question

How much inference cost can learned model routing eliminate while preserving strong-model quality across heterogeneous workloads, and in which workload regimes does each routing strategy dominate?

## Principles

1. Count genuine model calls separately from offline routing-policy replays.
2. Freeze the held-out test set before the main collection run.
3. Never train a learned router on held-out outcomes.
4. Report per-domain results and uncertainty, not only an aggregate score.
5. Keep raw observations immutable and make every aggregate reproducible from them.
6. Scale is useful only when it supports a defensible finding; the pilot is a go/no-go gate.

## Observation schema

Each benchmark task should have:

- `task_id`
- `dataset`
- `category`
- `difficulty`
- `split`
- `prompt`
- `verifier`
- `reference` or verifier-specific metadata

Each live model outcome should have:

- `run_id`
- `task_id`
- `model`
- `response`
- `score`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `cost_usd`
- `error`
- `timestamp`

The pair `(benchmark_version, task_id, model, replicate)` is the stable identity for resumability.

## Experimental stages

### Stage 0 — infrastructure

Build dataset adapters, immutable JSONL outcome storage, resumable collection, deterministic train/validation/test manifests, model pricing metadata, and analysis scripts.

### Stage 1 — pilot

Run 300–500 prompts across at least two model tiers. The pilot must answer:

- Are verifier results trustworthy on a manually audited sample?
- Is there enough disagreement between model tiers for routing to matter?
- Are domains represented without one category dominating the aggregate?
- What are the observed API cost and runtime per 1,000 outcomes?
- Are latency and token-usage fields consistently available?

Do not launch the main study if the cheap and strong model are nearly indistinguishable or if verification is noisy enough to swamp routing differences.

### Stage 2 — main outcome matrix

Minimum target: 10,000 prompts and 40,000 genuine model outcomes when four model tiers are available. Stretch target: 25,000 prompts × 4 models = 100,000 genuine outcomes.

Suggested categories:

- math/reasoning
- coding
- factual/open QA
- extraction/classification
- instruction following
- summarization
- tool-use/agent steps

Prefer public, reproducible datasets. Preserve dataset provenance and license information in the manifest.

### Stage 3 — routing policies

Evaluate on the same held-out outcomes:

1. always-small
2. always-large
3. v0 heuristic rules
4. v1 weighted feature score
5. v2 supervised learned router
6. v3 PPO/cost-aware policy

Report quality, normalized cost, routed-large fraction, latency when meaningful, and reward for explicitly declared cost-aversion values.

### Stage 4 — ablations

Required:

- feature-group ablation
- training-data scaling curve
- escalation/confidence threshold sweep
- cost-aversion lambda sweep
- per-category breakdown

Optional:

- cheap-first cascade vs direct routing
- model-pair sensitivity
- verifier sensitivity

### Stage 5 — statistics

For headline quality/cost results:

- bootstrap the held-out tasks for 95% confidence intervals
- use paired comparisons because policies are evaluated on the same task outcomes
- publish numerator/denominator alongside percentages
- do not call tiny differences meaningful when intervals overlap substantially

## Headline artifacts

The final report should include:

1. quality–cost Pareto frontier
2. quality and cost by workload category
3. routing rate by difficulty/category
4. lambda/threshold sweep
5. training-data scaling curve
6. error taxonomy and representative routing failures

## Resume claim policy

Do not write a scale claim until the immutable result files exist. Distinguish carefully between:

- `N model outcomes`: actual model executions
- `N routing-policy evaluations`: offline policy decisions over cached outcomes
- `N prompts/tasks`: unique benchmark inputs

A defensible final bullet should have the form:

> Led the design and execution of a large-scale LLM routing study across **N prompts and M genuine model outcomes**, comparing heuristic, supervised, and cost-aware RL policies across heterogeneous workloads; retained **X% of strong-model quality at Y% of inference cost** on a frozen held-out set.

# ARC-AGI-2 results (public eval, 120 tasks)

Model: gpt-5.6-luna (snapshot 2026-07-09), reasoning effort medium, official
prompt + parsing from arc-agi-benchmarking, official fractional scoring
(strict = all test inputs of a task correct). Search config: 3 seed nodes,
10 rounds, evolve every 2 (with opt-out), inspirations 2, tau 0.2, deg 0.3.
Each run dir holds attempts/, logs/ (full event streams), summary.json.

## Main: full search, 3 repeated runs (main/)

| run | fractional | strict |
|-----|-----------:|-------:|
| search120-r10-seed3-s0 | 0.1694 | 18/120 |
| search120-r10-seed3-s1 | 0.1653 | 18/120 |
| search120-r10-seed3-s2 | 0.1625 | 16/120 |
| **mean ± std** | **0.1657 ± 0.0035** | **17.3 ± 1.2** |

Compute-scaling checkpoints (s0, via arcagi2/search/checkpoints.py):
strict 13 / 16 / 17 / 17 / 18 at rounds 2 / 4 / 6 / 8 / 10.

Also in main/: search120-r6-seed2-s0 (2 seed nodes, 6 rounds: 0.1250, 14/120,
$17.24) — smaller-budget comparison point.

## Baselines (baselines/), single runs

| method | fractional | strict |
|--------|-----------:|-------:|
| direct (2 attempts) | 0.0792 | 8/120 |
| self-refine k=10 | 0.0889 | 8/120 |
| majority vote N=20 | 0.1236 | 12/120 |

majority20-120-s0 attempts files include a `pool` field (distinct grids +
votes per test pair) for verifier reranking / majority@k merges.

## Ablations (ablations/), single runs at full config minus one component

| config | fractional | strict |
|--------|-----------:|-------:|
| full search (mean) | 0.1657 | 17.3 |
| − parent guidance (random parents) | 0.1625 | 17/120 |
| − verifier feedback | 0.1319 | 14/120 |
| − verifier evolution | 0.1319 | 11/120 |

Component ordering: evolution > feedback > sampling guidance.

Costs (OpenRouter, true pricing): search ≈ $27/run; baselines were run via
the abaka gateway at 5x-inflated pre-July-30 luna pricing (true-cost
equivalents ≈ $3 direct / $11 self-refine / $28 majority@20).

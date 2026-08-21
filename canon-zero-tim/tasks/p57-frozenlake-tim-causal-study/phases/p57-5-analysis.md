# P57.5 — Preregistered analysis and claim decision

## Purpose

Turn the paired campaign into an auditable result without changing endpoints
or exclusions after seeing the arm comparison.

## Primary analysis

- For each paired seed, compute held-out solve-rate AUC from update 0 through
  200 using the registered evaluation checkpoints.
- Report the paired zero-minus-mismatch differences for every seed, their mean,
  uncertainty interval, and all raw seed trajectories.
- A campaign with unresolved uncertainty is `INCONCLUSIVE`; three seeds are not
  inflated into a universal population claim.

## Secondary analyses

- Held-out solve rate at update 200.
- Time-to-preregistered solve thresholds.
- Collapse incidence and cross-seed variance.
- Training reward/solve curves as supportive, not replacement, outcomes.

## Mechanistic analyses

- Relate the preregistered A-B mismatch dose to importance-ratio tails,
  clipping, gradient/update norms, and effective groups.
- Stratify by context length, turn, completion length, and action validity.
- B-C must remain exact; any B-C failure invalidates the affected run rather
  than becoming an explanatory covariate.
- Analyses discovered after unblinding are labeled exploratory.

## Systems analyses

- Report wall seconds/update, sampled tokens/second, checkpoint/evaluation
  cost, and fixed-input warm replay overhead.
- Separate contract overhead from behavior-mediated cost. If one arm generates
  longer trajectories, report both per-update time and token-normalized
  throughput.
- Do not claim that learning improvement is free merely because fixed-input
  kernel overhead is small.

## Decision table

| Evidence | Decision |
|---|---|
| Zero arm exact, mismatch dose admitted, zero arm has better primary/stability outcomes | TIM harms this registered recipe |
| Zero arm exact, mismatch dose admitted, paired outcomes unresolved | `INCONCLUSIVE` |
| Zero arm exact, mismatch dose admitted, no material paired difference | This recipe is robust to this measured dose |
| Mismatch arm has no A-B dose | `NO_TREATMENT`; no causal learning claim |
| Zero arm is not exact | Invalid experiment |
| Task hits floor/ceiling or excessive truncation | Invalid benchmark |
| B-C or other non-treatment correctness gate fails | Invalid affected run/campaign |

## Optional follow-up

An unclipped importance-sampling or CISPO-like objective may be tested later as
a separately approved sensitivity phase. Its objective, thresholds, and claim
must be preregistered before launch; it cannot be introduced into the main
study after observing results.

## Final claim ceiling

The strongest valid conclusion is conditional on dense Qwen3-8B, the frozen
FrozenLake maps and horizon, the registered GSPO-token/RLOO recipe, and the
measured mismatch dose. Generalization to MoE, other tasks, or all RL training
requires additional experiments.

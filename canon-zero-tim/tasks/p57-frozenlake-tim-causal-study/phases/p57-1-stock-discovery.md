# P57.1 — Stock-only rollout calibration and full-curve selection

## Purpose

Select one learnable, nonsaturated, long-context FrozenLake workload without
using any zero-TIM learning result. The first action is one no-update
calibration, not a short training sweep. Every run in this phase uses the
untreated `stock-fast` serving regime. Fixed lm-head off is only one member of
the required 37-switch zero-TIM-off attestation.

## Calibration recipes

All maps are deterministic (`is_slippery=false`) and materialized with exact
map bytes, generation seed, shortest safe path, row index, split, and SHA-256.
Mixed recipes use frozen probability 0.82 and the path envelope
`max(4, side-1) .. min(max_turns, side+5)`.

| Recipe | Grid mixture | Max turns | Context hard cap | Role |
|---|---|---:|---:|---|
| m10 | balanced 5x5–10x10 | 10 | 8,192 | easy mixed candidate |
| m15 | balanced 5x5–12x12 | 15 | 12,288 | middle mixed candidate |
| m20 | balanced 5x5–15x15 | 20 | 16,384 | hard/long-context candidate |

One JobSet initializes from immutable base weights, then evaluates M10, M15,
and M20 sequentially at temperature 0.7 with eight generations for each of 100
maps. Every recipe result is independently signed and has a dataset SHA inside
the one aggregate stochastic receipt.

## Calibration outputs and hard gates

Each receipt contains solve rate, all-solved/all-failed/mixed prompt ratios,
nonzero-advantage ratio, invalid-action and terminal-status counts, turn
p50/p90/p99/max, prompt/context/completion token percentiles and maxima,
physical prompt/response cap hits, recipe context-cap excess, elapsed seconds,
and sampled tokens/second.

- Any malformed/nonfinite record, state mutation, source/image drift, or missing
  receipt makes the calibration invalid. A physical cap hit or recipe context
  excess makes that recipe ineligible. Calibration intentionally does not
  compute trainer/rescore logprobs, so it makes no A-B-C claim.
- The manifest gate, resolved-container marker, JSON v2 attestation, and offline
  classifier must all agree that 12 numerical switches are absent, 25 gates are
  zero, and the canonical excess-precision pin is absent. Fixed lm-head off by
  itself is invalid evidence.
- An eligible mixed recipe has stochastic solve 15–35%, mixed prompt ratio and
  inferred nonzero-advantage ratio at least 25%, and no physical or
  recipe-specific context cap hit.
- Choose the eligible recipe closest to 20% stochastic solve; exact ties prefer
  M15, then M10, then M20.
- If no recipe is eligible, stop for one user-approved task adjustment. Do not
  train a floor or ceiling recipe.

## Full stock curve

`p57cal6` is complete. The immutable original receipt had sentinel map
provenance but exact group/pair identities; a separately hashed deterministic
rematerialization plus group-id join derived only those missing fields. The
unchanged classifier returned `PASS / FREEZE_M15`. M15 measured 24.625% solve,
56% mixed/nonzero-advantage groups, context max 7,403, completion max 6,223,
and zero cap hits.

The full-bundle stock/mismatch training and evaluation paths are now
fail-closed: the profile and entrypoint skip the canonical overlay, validators
require the complete numerical bundle off, and runtime postflight requires
zero canonical markers. M15 is rebuilt from immutable base weights on its
disjoint `selection` maps and trained for 200 signed updates.
There is no train-20 screen. Independent held-out evaluations occur at updates
0, 50, 100, 150, and 200. Each evaluates 100 maps with eight deterministic
generations. Eight is the minimum admitted global row count for trainer-side
rescore on DP8 (global M=8, shard-local M=1); the repeated greedy generations
are coverage replicas, not independent map samples. LatestN(1) means pausing at
each registered boundary, evaluating, and resuming with the original final
horizon of 200.

The physical envelope is prompt 4,096 plus response 8,192. Training uses
DP8xTP8, resident optimizer state, batch 32, eight generations, trajectory
mini/micro 32/8, AdamW `1e-6`, GSPO-token/RLOO, and temperature 0.7. No
in-process evaluation is admitted.

Automatic freeze requires update-200 solve 60–70% and at least 15 percentage
points improvement from update 0. The 55–75% range is a review guardrail;
outside it is floor/ceiling. Only after freeze is the unseen `main` split signed
and the zero arm unblinded.

## Exit gate

- The three-recipe stochastic calibration is complete and classifier-valid.
- Exactly one eligible mixed recipe has a complete stock 200-update curve.
- The recipe, context contract, data SHAs, and unseen main split are frozen.
- No zero-TIM learning result has been launched or inspected.

## Claim boundary

Calibration measures initial capability and training-signal density. The stock
curve selects a workload. Neither result estimates the causal effect of TIM.

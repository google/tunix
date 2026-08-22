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

The stock/mismatch training and optional evaluation paths are fail-closed: the
profile and entrypoint skip the canonical overlay, validators require the
rollout/trainer numerical treatment bundle off, and runtime postflight requires
zero canonical markers. Training has one observer-only exception. After all
six stock engine files are verified, a signed two-file delta changes only the
runner's explicit prompt-logprob branch and adds its helper. Processed-B then
applies temperature 0.7 and gathers target IDs from absolute request history,
not a roll over the DP-packed input. This branch is dormant during rollout A,
does not supply `old_per_token_logps`, and is not a gradient input.
The training command pins `--sampler_is=none`. The learner therefore keeps
rollout A as `old_per_token_logps` and emits no TIS weights; a runtime purity
gate verifies those identities on the first real training batch. Standard
GSPO ratio clipping remains unchanged and shared with the later zero arm.
Calibration/evaluation keep the stock runner byte-identical. M15 is rebuilt from immutable base
weights on its disjoint `selection` maps and trained for 200 signed updates.
There is no train-20 screen. By user decision on 2026-08-21, P57.1 runs one
uninterrupted stock/mismatch JobSet from update 0 through 200. It does not run
eval-0 and does not pause at updates 50/100/150. Checkpoints remain every 10
updates with LatestN(1), but serve only infrastructure recovery. The discovery
curve is the signed on-policy training trajectory. An isolated eval-200 remains
optional after completion and is not a prerequisite for launching training.

Eval-0 attempts 1–3 remain `INCONCLUSIVE`: leaf runtime admission, DP8 row
divisibility, and finally a stale workload-entrypoint geometry assertion each
stopped before a complete receipt. The current repair makes the renderer and
real entrypoint share `GENERATIONS_PER_PROMPT=8`; no target result is inferred
from the local admission gates. They are preserved but no fourth attempt is
required under the direct-run decision.

Direct-training attempt `p57_stock_full_att1` is also `INCONCLUSIVE`. It
completed one real 256-trajectory rollout but stopped before backward/update 0
because the observer required processed `S_prefill` while the profile forced
the engine interface off. The repair admits processed-B only for stock
training observation; it does not resume or reuse attempt 1.

Direct-training attempt `p57_stock_full_att2` is `INCONCLUSIVE` as well. Source
`c5cc71b5...` passed the repaired admission, completed 256 trajectories, and
spent 27.114 seconds in a real B rescore. Before backward/update 0, target token
304 was absent from the returned dictionary, whose only key was 5795. The
committed 2,715-line log is SHA-256
`f49f35f4243cbe98af6b12f9632b88224c530f415be1386c6f360604da0cb749`;
there is no complete terminal package or checkpoint. Pinned-image inspection
showed two coupled causes: stock prompt scoring uses
`roll(input_ids, -1)` across the packed buffer, and it always scores raw prompt
logits even when decode reports processed logprobs. The current observer delta
fixes both without enabling fixed-M or any canonical training kernel.

The physical envelope is prompt 4,096 plus response 8,192. Training uses
DP8xTP8, resident optimizer state, batch 32, eight generations, trajectory
mini/micro 32/8, AdamW `1e-6`, GSPO-token/RLOO, and temperature 0.7. No
in-process evaluation is admitted. No mismatch-aware weighting, filtering,
old-logprob substitution, learning-rate change, clipping change, or advantage
change is admitted.

Automatic freeze requires the preregistered trailing update-200 on-policy solve
statistic to be 60–70%, with finite mismatch dose, valid B-C/structural gates,
and no cap/truncation failure. Without a same-split eval-0, this phase makes no
held-out improvement-from-baseline claim. The 55–75% range is a review
guardrail; outside it is floor/ceiling. Only after freeze is the unseen `main`
split signed and the zero arm unblinded.

## Exit gate

- The three-recipe stochastic calibration is complete and classifier-valid.
- Exactly one eligible mixed recipe has a complete stock 200-update curve.
- The recipe, context contract, data SHAs, and unseen main split are frozen.
- No zero-TIM learning result has been launched or inspected.

## Claim boundary

Calibration measures initial capability and training-signal density. The stock
curve selects a workload. Neither result estimates the causal effect of TIM.

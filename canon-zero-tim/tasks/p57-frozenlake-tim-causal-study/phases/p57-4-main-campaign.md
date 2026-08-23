# P57.4 — Paired multi-seed main campaign

## Purpose

Measure capability, stability, and cost under the frozen causal contract.

## Design

- Arms: zero TIM and finite TIM.
- Paired seeds: 42, 43, and 44; identical seed/order/checkpoint pairing across
  arms.
- Primary horizon: 300 updates.
- Checkpoints: every 10 updates.
- Rollout-only held-out evaluations: updates 0, 50, 100, 150, 200, 250, 300.
- Evaluation contract: immutable 100-row held-out maps, common temperature-0.7
  sampling, eight generations per map, fixed map order, and no prefix state
  shared with training. The exact 800-reward inventory is required at every
  point and evaluation examples never enter trainer forward/backward.
- Each expensive arm launch requires explicit user approval.

## Horizon rule

The 300-update horizon is fixed before inspecting the final arm gap. Do not
stop the apparently losing arm early and do not extend only one treatment.

## Run receipts

Every run must persist:

- source/image/model/recipe digests and intent diff;
- train transaction, checkpoint, and rollout-only evaluation receipts;
- per-step A-B/B-C dose summaries and zero exactness;
- solve/reward, effective/mixed groups, context/turn/completion lengths,
  truncation and invalid actions;
- importance ratios, clipping, gradient/update norms, nonfinite counters;
- wall time, sampled tokens, TPU topology/HBM, and failure/restart history.

Cluster recovery may restart a run, but scientific pairing must resume from the
same signed checkpoint and data cursor. A restart from initialization is a new
attempt, not a continuation of the original seed.

## Exit gate

All six primary runs complete through update 300 with valid paired receipts and
the registered rollout-only evaluations, or the phase records an explicit invalid
or inconclusive terminal state.

## Claim boundary

Do not inspect partial results to stop the apparently losing arm. Equal budget
and the fixed horizon are part of the causal contract.

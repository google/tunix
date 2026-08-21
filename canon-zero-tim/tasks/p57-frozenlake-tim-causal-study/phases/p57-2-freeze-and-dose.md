# P57.2 — Recipe freeze and treatment-dose admission

## Purpose

Freeze every non-treatment choice before observing mismatch-arm learning, then
prove that the two arms actually differ by a finite, reproducible TIM dose.

## `FROZEN_RECIPE.json`

The immutable receipt must contain at least:

- source commit, container image digest, model/checkpoint digest;
- model dimensions, topology, mesh order, dtype, kernel registrations;
- train/held-out map hashes and exact ordering;
- selected-candidate proof, all stock discovery receipts, and proof that no
  zero-arm outcome was observed before freeze;
- paired seeds `[42, 43, 44]` and seed-to-arm assignment;
- rollout sampling, prompt/generation counts, context/turn limits;
- optimizer, learning rate, objective, clipping, microbatching, update horizon;
- shared performance flags, checkpoint cadence, evaluation schedule;
- the complete arm-specific numerical treatment bundle and warning/fatal policy;
- primary/secondary outcomes, stopping conditions, extension rule, and claim
  ceiling.

The file is signed and stored before the first zero-arm learning outcome is
launched or inspected. Stock discovery has already run; any later recipe
change creates a new study version and cannot overwrite this receipt.

## Treatment admission

Use fixed-input bounded rounds so stochastic trajectories cannot masquerade as
arm differences.

- zero arm: every A/B/C value is bytewise exact;
- mismatch arm: A-B is finite and nonzero in at least two repeated bounded
  rounds with the same structural signature; B-C remains bytewise exact;
- tokens and all nonnumerical receipts are equal between arms; every numerical
  difference is explicitly registered in the treatment bundle;
- no correctness gate outside the allowed finite A-B treatment is weakened.

Do not choose or modify the FrozenLake task based on mismatch magnitude. The
dose is measured after the task is frozen.

## Decision branches

- Both gates pass: proceed to P57.3.
- Zero arm nonexact: `INVALID_ZERO_ARM`; repair and recertify P57.0.
- Mismatch arm exact: `NO_TREATMENT`; stop without a causal learning claim.
- B-C mismatch or other correctness failure: `INVALID_TREATMENT_ARM`; repair
  rather than warning through it.

## Exit gate

Signed `FROZEN_RECIPE.json`, exact zero-arm receipt, reproducible mismatch-dose
receipt, and a complete registered bundle diff are all present.

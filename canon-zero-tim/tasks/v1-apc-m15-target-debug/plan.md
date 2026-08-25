# Plan

## Outcome

Localize the real M15 DP8xTP8 APC-on serving mismatch to the first exact tensor interval and source boundary before changing numerical code. B remains an independent full recomputation with `reset_prefix_cache=True`; all production recipes remain APC-off. No TPU, pinned-image, Kubernetes, commit, or push action is implicit in this plan.

The immutable numerical contract is:

```text
A = APC-on rollout decode
B = serving prefill rescore of A's exact action IDs with reset_prefix_cache=True
C = trainer old-policy forward

A - B = 0 bytes
B - C = 0 bytes
```

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| A | `M15_FIRST_RED_INPUT_CONTRACT` with mismatch distribution, identity hashes, and artifact-completeness matrix | every field is traced to immutable `file:line` or explicitly marked missing | complete: audit PASS, replay inputs missing |
| B | fresh captured-red replay carrier and APC-off/APC-on decision table | static/host carrier tests prove full 256-row producer, every-call envelope, exact first-red join, GCS durability, A cache-readable decode, B full reset, zero backward/commit | Attempt-0 bootstrap invalid; Attempt-1 exposed legacy geometry cross-talk; bounded geometry repair host PASS; exact-image/target pending |
| C | single-variable reproduction ladder | control A-B=0; treatment either reproduces red or is classified `ONEHOST_NOT_REPRODUCED` without mechanism claims | pending |
| D | observer-neutral coarse-to-fine first-red walk | `FIRST_RED_LOCALIZED` names last exact and first red tensor, shape ledger, request/token/cache coordinate, and `file:line` | pending |
| E | minimal localized repair, default off or experiment-bound | reproducer flips red to zero; APC-off and B are unchanged; adjacent and dirty-page negatives fire | pending |
| F | certification ladder | host -> exact-image -> one-host clean/repeat/dirty -> matched profile -> separately approved DP8xTP8 G-E | pending |

## Phase-C decision table

| Observation | Interpretation | Next action | Forbidden claim |
|---|---|---|---|
| replay tokens/order differ from `m15i` | comparison invalid | repair capture/join | APC mechanism conclusion |
| APC-off control is red | carrier or shared serving contract is invalid | stop before APC treatment | prefix-cache root cause |
| APC-on one-host is exact | production envelope not reproduced | add exactly one missing M15 variable | bug disappeared |
| APC-on clean run is red | deterministic carrier available | observer-neutral Phase D | RoPE/page cause before localization |
| observer changes endpoint bytes | instrumentation invalid | redesign observer | first-red conclusion |
| first red is cache read with stored bytes exact | transient mapping/read interval | test one mapping/order degree of freedom | stale stored-page claim |
| target repair is exact | target forward candidate admitted | run required negative/repeat gates | production default-on before all gates |

## Decisions

- Confirmed: one-host Qwen3-8B DP1xTP4 G-A through G-D is green; no APC numerical repair was made there.
- Confirmed: M15 `m15i` on DP8xTP8 is a strict A-B red and B-C exact at step 0; production APC is off.
- Confirmed: first mismatch occurs at logical prefix 1226, so the older 1686-depth heuristic is not a valid lower bound for this case.
- Confirmed: all 760 mismatches belong to prompt-major group 24; rows 192, 193, 194, 196, 197, 198, and 199 are red while row 195/generation 3 is clean.
- Confirmed: only 6/760 red coordinates are exactly on a 256-token boundary; the first red is at offset 202. A simple block-boundary trigger is not supported.
- Confirmed: the archived `m15i` bundle contains hashes and all mismatch coordinates but no reversible full arrays or serving chronology. Historical `m15i` cannot be called an exact replay input.
- Confirmed: the receipt source SHA conflicts with the runtime sync receipt. The executed source is `71d889a32f4668353c758d5c00df88299e6c0d35`; the receipt's `7a2a456c...` is retained as a provenance defect, not used for replay.
- Hypothesis: M15 chronology, cache residency/churn, or TP8/multihost placement supplies a condition absent from the one-host carrier.
- Decision: do not change RoPE, attention, KV values, lm-head, loss, backward, or optimizer until Phase D passes.
- Decision: XProf/Perfetto work is deferred until a numerical red is reproduced; profiles cannot decide equality.
- Decision: reuse the checked-in P38 mismatch capsule, request journal, incident ledger, and fail-closed join classifier. Do not add a second cache observer until these existing host-only fields prove insufficient.
- Decision: because `m15i` raw replay inputs are absent, the next strict replay source must be a fresh captured target red. A new stochastic run may be compared structurally with `m15i`, but it is not the same historical trajectory.
- Decision: one fresh real rollout is required to create the carrier. Once
  frozen, Phase C reuses the saved token streams and serving chronology, so it
  does not need to solve FrozenLake or resample actions on every debug run.
- Decision: large producer/envelope payloads remain in the registered GCS
  attempt. The GCS-side audit uploads only small, self-hashed receipts under a
  versioned derived prefix.
- Confirmed: Attempt 0 did not reach an APC numerical boundary. Its command
  selected `m15/main` on the CLI but omitted the matching signed environment
  fields, so the FrozenLake entrypoint rejected the split identity before
  learner construction. The repaired carrier transports and checks both sides
  of that identity; this is a launcher-contract repair, not an APC repair.
- Confirmed: Attempt 1 also did not reach an APC numerical boundary. It passed
  overlay and GCS preflight, then the entrypoint treated every P38 precheck as
  the legacy DP16 carrier and rejected the signed M15 target's
  `mini_batch_size=32` and `sampler_is=none`. A second hardcoded legacy check
  would also have rejected `frozenlake-dp8-tp8` and the one-unit DP8 geometry.
- Decision: admission now has two explicit contracts. Legacy P38 remains
  `frozenlake`, DP16, 8 x 4 prompts, token IS; the experiment-scoped
  `CANON_APC_M15_TARGET_DEBUG=off|on` path is exactly
  `frozenlake-dp8-tp8`, DP8, 1 x 32 prompts, no IS. Cross-use and partial
  geometries fail closed. This changes no APC or model arithmetic.

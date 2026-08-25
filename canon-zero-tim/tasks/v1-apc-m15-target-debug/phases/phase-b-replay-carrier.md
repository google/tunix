# Phase B — freeze a strict replay carrier

- Status: static carrier complete; target execution not approved

## Finding

The historical `m15i` log proves a real APC-on A-B failure, but it cannot be
replayed exactly because its full arrays and serving chronology were not
archived. A hash is an identity check, not an inverse encoding. Therefore the
carrier must preserve the next clean-run red as a new immutable source while
retaining `m15i` only as the structural reference signature.

The repository already contains the needed primitive instruments:

- `tunix/rl/alignment.py` can write a compressed mismatch capsule containing
  prompt IDs/mask, completion IDs/valid mask, action mask, A/B/C values,
  policy version, sampling values, and prompt-major row identity;
- TPU runner P38 patches 09/13/14 can write exact pre-dispatch tensors, request
  IDs/order, token histories, `num_computed_tokens`, physical block mapping,
  observed page generations, DP slot, and co-batch membership;
- `classify_p38_serving_capture.py` already validates those artifacts and joins
  capsule rows to exact incident calls by token-history SHA.

The old P38 production bands begin at 1536, which misses the M15 first red at
1226. This phase may change only carrier admission/bounds and postflight; it
must not change RoPE, attention, K/V values, lm-head, loss, backward, optimizer,
or the B-arm reset.

## Static deliverables

1. A default-off M15 APC debug identity with exact DP8xTP8/M15/step-0 geometry.
2. APC-off control and APC-on treatment rendered from one source tree; both
   stop before backward and optimizer commit.
3. A first-red capture band containing 1226 plus adjacent controls, with a
   bounded incident ledger and immutable unique paths.
4. A mismatch capsule with enough selected red rows for token-history joins.
5. A complete producer/envelope carrier:
   - all 256 prompt/completion rows, masks, A/B/C values, policy version and
     sampling values in one compressed producer NPZ;
   - one host-only JSONL record for every serving call, including exact
     request dispatch order, A/B arm, DP rank/local slot, token-history SHA,
     logical position and physical pages;
   - a mechanical join from every request history back to one or more
     byte-identical producer rows, with the first-red source row required to
     be among its exact candidates.
6. Fail-closed postflight requiring:
   - production-congruent A (`prompt_logprobs=None`, sampled `logprobs=1`,
     `skip_reading_prefix_cache=False`);
   - independent B (`reset_prefix_cache=True`);
   - complete capsule/journal/incident SHA validation;
   - at least one exact incident-to-capsule join if treatment is red;
   - zero backward and zero optimizer commits;
   - source/runtime/model/policy and topology receipts.
7. Host synthetic positives and negatives for missing capsule, missing ledger,
   wrong source, wrong arm, wrong bounds, broken token join, B cache reuse, and
   any optimizer marker.

## Run decision table

| Arm result | Classification | Next action |
|---|---|---|
| APC-off A-B nonzero | invalid carrier/shared serving red | stop; do not interpret APC |
| APC-off exact, APC-on red, all joins complete | `FRESH_TARGET_RED_FROZEN` | use its exact inputs in Phase C/D |
| APC-off exact, APC-on exact, sufficient M15 depth/occupancy | `TARGET_NOT_REPRODUCED` | repeat only under preregistered stochastic rule or add one variable |
| APC-on red but capsule/incident join missing | `RED_NOT_REPLAYABLE` | repair capture only; no mechanism claim |
| Any B-C nonzero | independent contract red | stop; not an APC-only case |

## Claim ceiling

Until an approved run returns a red with complete joins, the ceiling is
`PHASE_B_STATIC_CARRIER_ONLY`. A newly captured trajectory is not historical
`m15i`; only its source/model/config and red signature may be compared with
`m15i`.

## Static result

- Added an exact DP8xTP8/M15 debug profile and renderer for a matched APC-off
  control and APC-on treatment. Both use one full 32-prompt producer unit and
  all 256 trajectories, then stop before backward/optimizer.
- Extended the existing P38 classifier to attest the mismatch-capsule SHA and
  diagnostic round, while preserving clean-run behavior when no capsule is
  expected.
- Added a high-level fail-closed M15 classifier. It accepts only
  `CONTROL_GREEN`, `FRESH_TARGET_RED_FROZEN`, or sufficiently representative
  `TARGET_NOT_REPRODUCED`; invalid evidence exits nonzero.
- Added A and B runtime receipts without modifying tensor values. A asserts
  the production cache-readable sampled-logprob request, while B asserts full
  reset and zero cached tokens.
- Added a first-red packager that returns one complete offending row plus exact
  incident mapping in a small nested SHA-verified bundle.
- Added a full producer-unit carrier and a runner patch that records every
  serving call without fetching device tensors. The full packager validates
  contiguous call chronology, both A/B arms, every token-history SHA against
  the 256-row producer, page geometry, and byte identity between the first-red
  row and the full producer. It emits `FULL_REPLAY_CARRIER_FROZEN`, not a
  replay or mechanism verdict.
- The existing P38 GCS pipeline now includes the growing replay envelope in
  live snapshots and all replay inputs/derived contracts in
  `serving-capture.tar`. A GCS-side audit validates root/nested manifests and
  uploads only small receipts to `derived/m15-replay-audit-v1`.
- Real CPU environment resolution found and fixed one launch blocker: the old
  mismatch-capsule admission recognized only workload literal `frozenlake` and
  rejected the exact debug identity `frozenlake-dp8-tp8`. The exception is
  scoped to the new debug selector/profile.
- Postflight now checks the M15 classifier and replay packager independently;
  expected controlled exit 42 cannot hide their failures.

Host/static gates passed: task tests 33/33, P38 classifier 37/37, Phase3
contract 12/12, Phase3 profile/boundary classifiers 11/11, V1 Phase4 CPU
contracts 29/29, flag audit 370/370, shell syntax, Python compile, and
`git diff --check`. Wider P33/P45 discovery ran all dependency-free cases but
could not import two tests because the host lacks `datasets` and `metrax`.
The existing V1 exact-image runner includes all 33 carrier tests. A full run
against pinned image `sha256:418dc632...e53a` passed with
`V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=33 ... manifests=3`. This does not
promote the claim beyond exact-image admission; one-host replay and target
topology remain unrun.

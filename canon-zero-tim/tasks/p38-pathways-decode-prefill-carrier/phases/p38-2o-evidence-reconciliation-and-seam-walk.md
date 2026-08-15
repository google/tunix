# P38.2o — Evidence reconciliation and decode seam walk

Status: active. O0 and O1 are locally complete. One hierarchical production
seam run is the next target only after review, commit, and push; this worktree
does not authorize or launch it.

## Goal

Turn P38s17 into a reproducible branch decision, then locate the first
decode-versus-prefill divergence without changing weights, KV contents,
production fixed-M geometry, sampling semantics, or training state.

## Entering evidence

- All three P38s17 rounds are A-B red and B-C bitwise exact.
- Reclassification from the six observer records and the three immutable round
  capsules joins row 255 in every round and reports zero aggregate/sample
  fingerprint differences over every valid KV extent.
- The earlier committed `live_kv_fingerprint_differs_on_red_row` JSON is not
  reproducible from the committed inputs. It joined different rows and
  coordinates.
- The committed directory is a `LIVE.json` snapshot. It has no
  `COLLECTED.json` or `COMPLETE.json`, and its old manifest included its own
  checksum. It is analysis-level evidence, not a terminal admitted bundle.
- Equality is at the registered diagnostic-fingerprint claim level, not a
  mathematical full-byte proof. It nevertheless selects the preregistered
  program-envelope branch and strongly lowers the stale-KV hypothesis.

## O0 — Reproducible evidence chain

Deliverables:

1. When immutable round capsules exist, `90_run.sh` passes only those capsules;
   the stable latest-round alias is a fallback, never a fourth input.
2. The classifier records its source SHA, every observer JSON/NPZ SHA, every
   capsule SHA, and the exact valid-token extents used for masking.
3. `--require-red-join` requires every observer A/B pair to join a red round.
4. Invalid page-tail changes remain masked; a valid-region one-bit change is
   red.
5. A live snapshot cannot be described as collected or complete. Evidence
   manifests exclude themselves and verify locally.

Exit gate: focused classifier and outer-postflight tests pass, the corrected
P38s17 classification re-runs byte-for-byte, and the P38s17 manifest verifies.

## O1 — Observer-neutral hierarchical seam instrument

Build a default-off diagnostic for bounded deep rows. The hierarchy prevents a
single 64-chip run from materializing every internal tensor of all 36 layers:

1. `layer` mode records only layer input/output for all 36 layers and final
   norm. It identifies the first divergent layer.
2. `full` mode records the internal seams below for exactly one selected layer.
   It is admitted only after the layer run names that layer.

The selected-layer checkpoints are:

1. layer input;
2. Q/K/V projections;
3. Q/K normalization;
4. post-RoPE Q/K;
5. RPA output;
6. output projection and residual;
7. post-attention normalization;
8. MLP output and layer output; and
9. final norm.

Raw target logit, normalizer, and logprob are not yet part of the seam payload.
If all hidden/final-norm fingerprints remain exact while A-B is red, classify
the result as a tail-localization requirement and add one bounded shared tail
observer. Do not describe an unobserved tail as exact.

The production endpoint remains authoritative. The instrument is invalid
unless observer-off and observer-on endpoints are bitwise equal, the one-bit
negative is detected, and all shape/sharding/weight/position contracts match.
Existing `CANON_CUT` controls are falsifiers only: they replace tensors and
change the graph, so they cannot be the final localization evidence.

Exit gate: PASS. Both pinned Qwen3 overlays verify the patch manifest and
runner tests. The real Qwen3-8B DP1xTP4 rehearsal completed three frozen
rounds with backward/optimizer zero. Observer-on produced 130 bounded records
for rounds 0 and 2; round 1 had no row in the registered 1400..3072 band.
Observer-off/on tokens, action masks, and all three endpoint arrays were
bitwise equal in every round. Local A-B was exact and is not a carrier verdict.

## O2 — Hierarchical production seam runs

O2a is one stock `layer` run at DP16xTP4, concurrency 256, prefix-cache off,
three frozen rounds, backward zero, and optimizer commits zero. O2b is one
stock `full` run for the layer selected by O2a. O2b is not rendered before the
O2a classifier succeeds. No KV-unified, concurrency, batch-size,
full-training, or repair arm is admitted in this phase.

| First red checkpoint | Selected branch |
|---|---|
| layer input | position, metadata, embedding, or upstream decode envelope |
| Q/K norm exact; post-RoPE red | RoPE/position application |
| post-RoPE exact; RPA red | RPA query/metadata/output envelope, not stale valid KV content |
| RPA exact; residual/MLP red | output projection, residual, cast, norm, or MLP seam |
| hidden chain exact; logit/normalizer red | logits/logprob tail |
| hidden/final norm exact while endpoint A-B is red | tail observer required; no hidden-op claim |
| endpoint no longer matches the unobserved arm | invalid observer; discard the run |

O2a exit gate: every capsule red action joins an A/B layer record, the target
classifier is PASS, and at least one first divergent layer is measured. O2b
exit gate: the selected-layer internal checkpoint is measured, or the result
is explicitly inconclusive/tail-only. Only a measured checkpoint may select a
repair.

## Claim ceiling

- P38s17 rejects the old *differing fingerprint* claim; it does not prove full
  KV byte identity.
- No operator, including RoPE, is guilty until the ordered first-red checkpoint
  is measured.
- No P38 result admits full training until the eventual repair restores strict
  A=B=C and passes backward/optimizer gates.

## Rollback

Leave the new seam observer unset. The production model, canonical kernels,
KV cache, optimizer, evaluation, and full-training paths remain unchanged.

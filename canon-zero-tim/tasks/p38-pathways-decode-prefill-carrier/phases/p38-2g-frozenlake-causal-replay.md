# P38.2g: FrozenLake single-row causal replay

## Purpose

Use one hash-verified P38.2f capsule to separate logical depth, multi-turn
MIXED scheduling, fused cache update, and broader cache/fresh behavior before
changing a production attention path.

Historical constraint: Phase 13's PATHTRACE-proven `CANON_KV_UNIFIED` arm had
zero numerical effect, and Phase 14 observed bitwise equality between
full-fresh and cache-plus-fresh inputs inside one MIXED kernel. The current
v0.25/long-context/Pathways domain may differ, but the two-pass arm is a new
causal retest, not a previously proven repair.

## Admission

- Input capsule passes transport, schema, and every embedded array SHA check.
- Prefix cache remains disabled.
- Every arm starts from independently reconstructed or snapshotted cache
  state; no arm consumes a donated or mutated cache from another arm.
- Stock production code remains unchanged. Counterfactuals are default-off and
  report-only, with no backward, optimizer, checkpoint, or W&B claim.
- Record actual valid prompt lengths. Do not infer logical KV depth from padded
  caps or completion-length summaries.

## Implementation status

The R0/R1 infrastructure is locally admitted. The verified P38e1 target
capsule at `../../debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz` now
passes file SHA, schema, embedded-array SHA, and schedule-coverage checks for
source rows 191 and 199, but those target rows have not yet run on real TPU
weights. `tunix/rl/p38_frozenlake_replay.py` verifies every embedded capsule
array hash, compacts tokens, builds the two schedules, and proves that every
action predictor is covered exactly once. The live adapter lowers those calls
to global/local M=256 metadata, runs every arm twice with independent fresh
cache state, keeps full logits on device, and returns only bounded target
diagnostics.

The capsule does not contain the original serving scheduler's per-call page
tables or distributions. R0 is therefore explicitly `mask-derived-v1`, not an
exact captured-scheduler replay. If it does not reproduce the local red against
the unchanged fixed-chunk reference, classify `LOCAL_CARRIER_NOT_REPRODUCED`
and move the shadow measurement to source-pinned Pathways.

R2/R3 are intentionally not implemented yet. The capsule prerequisite is now
met; they remain gated on real target-input R0/R1 repeat and negative-control
evidence. Local gate
evidence is in `../artifacts/p38_2g_local_gate.md`; the real-Qwen synthetic
deep/shallow controls are in `../artifacts/p38_2g_onehost_synthetic_0811.md`.

Target row 191 subsequently completed on real Qwen3-8B. R0 and R1 were exact
at all measured stages, but both differed from REF at 395 of 517 logprobs.
REF exactly reproduced the captured `S_prefill`/`T_old` SHA, while R0/R1 did
not reproduce captured decode. The local serving-envelope prerequisite failed,
so R2/R3 remain gated and the shadow measurement moves to source-pinned
Pathways. Evidence is in `../artifacts/p38_2g_onehost_target_row191_0811.md`.

The synthetic controls produced bitwise-exact R0 versus R1 at prompt lengths
256 and 1788, while both arms differed from REF at every scored action. The
shallow difference was larger, so this probe does not reproduce the production
KV-1791 onset. It measures a broader incremental-cache versus fixed-chunk
boundary and cannot be used to promote a KV-unified candidate.

## Arms

| Arm | Single controlled change | Question |
|---|---|---|
| R0 | Reproduce the captured multi-turn call schedule | Can the local carrier measure the known red? |
| R1 | Same tokens and logical depth, one continuous turn | Is multi-turn/MIXED scheduling required? |
| R2 | R0 plus two-pass cache-write/read only for MIXED calls | Is fused MIXED cache update the carrier? |
| R3 | R0 plus two-pass cache-write/read for every distribution | Is a broader cache/fresh split required? |

R2 and R3 may be implemented only after R0/R1 infrastructure passes a
one-bit negative control and proves the intended branch executes. Prefer the
dedicated cache-update kernel over computing attention twice, but preserve the
exact bf16 cache layout and page mapping.

## Measurements

At every arm and scan point, record:

- raw target logit, vocabulary normalizer, processed logprob, and exact bits;
- `q_len`, logical `kv_len`, `request_distribution`, `update_kv_cache`, and the
  cache/fresh split point;
- exact logical-to-physical page IDs around the target, not only a digest;
- configured `(bq, bkv, bq_compute, bkv_compute)` values;
- per-layer current-page K/V hash, attention-output hash, and hidden hash until
  the first differing layer.

Scan `1536`, `1792`, `2048`, `3840`, and `4096` with at least `±4` positions.
This separates page-only boundaries (`1792`, `3840`) from boundaries that are
also multiples of the pinned 512-token KV block.

## Exit and decision table

R0 must reproduce the known red. If it does not, classify the local result as
`NOT_REPRODUCED` and move the same shadow arms to a source-pinned Pathways
diagnostic; do not interpret R1-R3.

| Observed result | Classification | Next action |
|---|---|---|
| R0 red, R1 exact | multi-turn/MIXED scheduling carrier | localize first differing call and layer |
| R0 red, R2 exact | fused MIXED cache-update carrier | promote MIXED-only candidate to P38.2h |
| R2 red, R3 exact | broader cache/fresh carrier | promote all-distribution candidate with explicit performance risk |
| R0-R3 red, cache K/V first differs | cache write or page-placement carrier | isolate the first changed page/layer |
| R0-R3 red, cache exact but attention output differs | RPA read/gather specialization carrier | create a minimal kernel reproducer or upstream patch |
| hidden exact, normalizer red | vocabulary reduction carrier | return to canonical logprob-tail work |

No single point is a PASS. The carrier classification must form a stable
boundary pattern or a two-point local plateau and retain the negative control.

## Candidate boundary

P38.2g selects a candidate; it does not authorize production. Any two-pass
candidate must next pass P38.2h with exact A=B=C, actual-model q/k/v/o gradient
health and correctness, VJP2 dcache-to-K/V routing, DP replica equality, and
zero optimizer commits. A forward-only green result cannot promote the
candidate because separating cache write from attention can silently cut the
K/V gradient dependency.

## Rollback

Leave all replay and KV-unified controls unset. Delete no capsule or failed
evidence; discard only the isolated diagnostic worktree or revert its bounded
commit after preserving the reports.

## Command after P38.2f

```bash
canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  /absolute/path/to/recovered-p38-capsule.npz <unique-label>
```

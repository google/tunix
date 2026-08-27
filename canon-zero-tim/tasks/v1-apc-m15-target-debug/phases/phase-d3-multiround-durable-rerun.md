# Phase D3 — three-round durable Layer-0 rerun

## Goal

Run one matched APC-off/APC-on DP8xTP8 pair that produces three independent,
frozen-weight diagnostic rounds per arm.  Each round must be sealed, uploaded,
read back, classified, and acknowledged before the next rollout begins.  The
primary next run uses the full observer at Layer 0, because Attempt 12's
analysis-grade coarse result placed the first red interval between Layer-0
input and output.

This phase changes evidence transport and repetition only.  It does not fix or
alter prefix-cache, RoPE, attention, KV, LM-head, loss, backward, optimizer, or
the independent full-reset B arm.

## Why another bounded run is justified

Attempt 12 showed that a useful numerical result can exist while the returned
root evidence chain is incomplete.  A single diagnostic round leaves one
post-rollout exit window between useful data and the terminal root markers.
Simply increasing training steps would be wrong: it would allow backward or
weight changes and would not make any individual observation durable.

The D3 contract instead performs three evaluation-only rounds with the same
weights.  Round `r` cannot advance until the live worker has written and read
back `wide/rounds/rrrrrr/WIDE_ROUND_COMPLETE.json`.  Consequently, a later pod
death cannot invalidate an earlier sealed round.  Root `COLLECTED.json` and
`COMPLETE.json` remain necessary for a full target PASS, but they are no longer
necessary to recover already sealed per-round classifiers.

## Implementation contract

- `CANON_P38_DIAGNOSTIC_ROUNDS=3` only for M15 `m15-wide-v1` layer/full runs;
  observer-none carriers remain one round.
- Backward and optimizer commits remain zero in all three rounds.
- Seam/tail record indices remain process-global and never repeat.
- The seam and tail byte counters reset only at a strictly sequential round
  transition.  The record counter does not reset.
- Local shards live under
  `p38_m15_wide_shards/round-rrrrrr/`; a classifier can never ingest another
  round's shard.
- The cumulative replay ledger is schema-checked line by line and filtered to
  the current round during assembly.
- Each classifier, input receipt, and completion receipt carries the same
  `diagnostic_round` and full source SHA.
- Root collection binds the final round (`2`), while all three immutable
  per-round objects remain addressable in GCS.

## Local gates

```bash
python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
bash -n \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_gcs_return.sh
git diff --check
```

Required negative cases include a second-round shard not reading round 0, a
combined replay ledger returning only the selected round, wrong diagnostic
round/source/hash rejection, root-incomplete recovery, partial-round recovery,
and one-bit/tampered-classifier detection.

## Target procedure (separately approval-gated)

After publication, render but do not launch with:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
```

The pair may be submitted concurrently.  Every `kubectl apply` is standalone.
After both jobs terminate, the bucket-capable executor runs:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_gcs_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

The return contains only the six small classifier JSONs, summary, packaging
receipt, and hashes.  It never downloads or returns the token-bearing tars.

## Target gate and decision table

| Observation | Status | Decision |
|---|---|---|
| Six rounds sealed and both roots terminal | `COMPLETE` | compare full Layer-0 first-red signatures across rounds |
| Six rounds sealed, root terminal missing | `ROUNDS_RECOVERED_ROOT_INCOMPLETE` | numerical evidence is usable; whole-run claim remains analysis-grade |
| One to five rounds sealed | `PARTIAL_ROUNDS_RECOVERED` | use recovered rounds, but do not claim paired target completion |
| No sealed round | `NO_DURABLE_ROUND` | debug worker/upload before another launch |
| off red, any B-C red, source/round/hash mismatch | hard RED | do not interpret APC mechanism |

Stable full-observer first-red signatures across independent on rounds support
a deterministic Layer-0 program/cache-read seam.  Different signatures imply
runtime state sensitivity.  Exact on rounds are retained as non-reproductions;
they do not erase red rounds.  No repair is authorized until at least one red
full-observer round reaches `FIRST_RED_LOCALIZED` with its last exact and first
red checkpoint.

## Claim ceiling

Before a fresh target run:

```text
MULTIROUND_DURABILITY_IMPLEMENTED / HOST_PASS /
EXACT_IMAGE_NOT_RUN / TARGET_NOT_RUN / NUMERICAL_FIX_NOT_AUTHORIZED
```

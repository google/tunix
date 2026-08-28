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
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_gcs_return.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_operator_return.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/recover_m15_attempt14_d33_operator_return.sh
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
After both jobs terminate, an executor with read-only Kubernetes and bucket
access runs:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_operator_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data default
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

The return contains the six small classifier JSONs, numerical summary,
sanitized JobSet terminal receipts, remote raw-log SHA/size receipts, packaging
receipts, and one final manifest.  It never downloads or returns `run.log` or
the token-bearing tars.  The underlying GCS-only wrapper remains an internal
primitive and must not be run separately by the remote executor.

## Attempt-14 return recovery (active)

d33 has already run. Its submitted five-file directory did not use the
operator wrapper above and therefore does not satisfy this phase gate. Do not
relaunch it and do not require its vanished original render directory. A
bucket/Kubernetes-capable executor now runs:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/recover_m15_attempt14_d33_operator_return.sh \
  "$RETURN" /mnt/disks/tunix-data default
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

The recovery script treats the committed subset only as a locator: it verifies
the subset manifest, derives the exact source/JobSet identities, emits
`RECOVERY_INPUT_RECEIPT.json`, and then runs the official per-round and
operator audits. It never uses the subset's numerical prose as a classifier.
Phase D3 remains open until the recovered machine package is reviewed.

## Target gate and decision table

| Observation | Status | Decision |
|---|---|---|
| Six rounds sealed and both roots terminal | `COMPLETE` | compare full Layer-0 first-red signatures across rounds |
| Six rounds sealed, root terminal missing | `ROUNDS_RECOVERED_ROOT_INCOMPLETE` | numerical evidence is usable; whole-run claim remains analysis-grade |
| One to five rounds sealed | `PARTIAL_ROUNDS_RECOVERED` | use recovered rounds, but do not claim paired target completion |
| No sealed round | `NO_DURABLE_ROUND` | debug worker/upload before another launch |
| off red, any B-C red, source/round/hash mismatch | hard RED | do not interpret APC mechanism |

The operator summary adds `_OPERATOR_RECEIPTS_INCOMPLETE` when a JobSet is not
terminal or a raw-log manifest/size receipt is absent.  That suffix never
discards recovered rounds and never upgrades the numerical core status.

Stable full-observer first-red signatures across independent on rounds support
a deterministic Layer-0 program/cache-read seam.  Different signatures imply
runtime state sensitivity.  Exact on rounds are retained as non-reproductions;
they do not erase red rounds.  No repair is authorized until at least one red
full-observer round reaches `FIRST_RED_LOCALIZED` with its last exact and first
red checkpoint.

## Claim ceiling

After d33 execution but before a complete operator return:

```text
TARGET_EXECUTED / SUBMITTED_SUBSET_HASH_VALID /
EVIDENCE_COMPLETENESS_RED / ANALYSIS_GRADE_ONLY /
NUMERICAL_FIX_NOT_AUTHORIZED
```

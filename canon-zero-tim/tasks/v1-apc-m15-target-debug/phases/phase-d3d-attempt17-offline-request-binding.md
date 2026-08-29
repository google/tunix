# Phase D3d — Attempt-17 offline source-row/request binding

## Objective

Use the immutable APC-on Round-0 bundle from Attempt 17 (`d36`) to decide
whether source row 217 can be bound to one serving request without another TPU
rollout. This phase changes analysis identity only. It does not change APC,
RoPE, RPA/attention, KV values, A/B/C, production flags, backward, or optimizer
behavior.

## Bound inputs

- Runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.
- Published evidence commit:
  `6e4e7f587941ee7e0c83753bc321a995912c8021`.
- Evidence directory:
  `evidence/v1_apc_m15_attempt17_d36_operator_return_20260829/`.
- Current numerical fact: APC-on Round 0 has A-B=207 differing bytes / 95
  elements, B-C=0, and the official result is
  `M15_INTERNAL_FIRST_RED_CANDIDATE_SET`.
- Current candidate fork: the same action/source row has an exact-through
  candidate and a Layer-0 `rpa_output` first-difference candidate.

The returned Git package does not contain the token-bearing bundle. The
sealed bundle remains in the registered remote evidence root and must be read
on a bucket-capable machine.

## Implementation

1. The classifier may use later replay-ledger `token_history_sha256` receipts
   to disambiguate same-prefix A requests. It selects a request only if:
   - exactly one request has matching future source-row prefixes;
   - every alternative has an explicit conflicting future prefix;
   - the selected matching receipt reaches or exceeds the latest alternative
     elimination horizon.
2. Missing history, an out-of-range history, or insufficient horizon remains
   `UNRESOLVED`. Request disappearance is never treated as a contradiction.
3. A uniquely bound exact-through candidate still does not fabricate a first
   red tensor.
4. `review_m15_attempt17_d36_candidate.py` verifies safe tar membership and
   the internal manifest, binds the bundle classification byte-for-byte to the
   committed receipt, reruns the official classifier, and emits only a small
   self-hashed return.
5. `run_m15_attempt17_d36_offline_binding.sh` reconstructs the original d36
   render from the full runtime SHA, verifies both JobSet identities against
   committed receipts, performs read-only remote recovery, downloads only the
   sealed treatment Round-0 compact bundle, and invokes the reviewer.

The wrapper requires a clean `local/*` worktree at a published analysis
commit. It performs GCS reads only: no upload, Kubernetes query, JobSet launch,
TPU use, or remote mutation.

## Local gate

```bash
python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash -n \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d36_offline_binding.sh
python3 -m py_compile \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/classify_m15_apc_wide_seam.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/review_m15_attempt17_d36_candidate.py
git diff --check
```

Required negatives include conflicting same-request duplicates, multiple B
numeric variants, selected-prefix proof shorter than the elimination horizon,
unique binding to an exact-through candidate, missing future evidence, and a
bundle classification that differs from the committed receipt.

## Bucket-capable execution gate

After the analysis change has a user-approved published commit, use a fresh
output directory on the bucket-capable machine:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d36_offline_binding.sh \
  /mnt/disks/tunix-data/m15-d36-offline-binding-return
```

Expected terminal lines:

```text
[M15.D36.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D36.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Return only the three small files in the output directory and their manifest
SHA. Do not return or commit the compact tar, capsule, replay ledger, token
prefix payloads, remote root, or credentials. A failed wrapper preserves its
scratch path on the executing machine; return the error and path, not the
payload.

## Decision table

| Result | Meaning | Next action |
|---|---|---|
| `FIRST_RED_LOCALIZED` with one future-prefix binding, last exact, first red, shape ledger, request/call/token/cache/page coordinate, and source file:line | Attempt-17 data is sufficient; no new rollout is needed to open a repair discussion | run the separately approved pinned exact-image gate for this analysis commit, review the return, then ask before Phase E |
| `FIRST_RED_CANDIDATE_SET_PRESERVED` | the durable bundle lacks enough future chronology to identify one request | add one observational source-row/request provenance field, run host and exact-image gates, then request approval for a fresh matched DP8xTP8 pair |
| identity, manifest, tar, committed-classification, or B invariant failure | evidence is invalid for reclassification | preserve output and stop; do not launch TPU |
| GCS/auth/network failure | infrastructure `INCONCLUSIVE` | preserve failure and retry read-only recovery; do not interpret numerically |

Phase E remains closed until the first row of the table is satisfied and the
return has been independently reviewed.

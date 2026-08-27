# Phase D2: durable M15 wide-observer shards

## Purpose

Make one DP8xTP8 coarse observer round survive a post-rollout process or pod
failure. This phase repairs evidence transport only. It does not change APC,
RoPE, attention, KV values, LM-head, loss, backward, optimizer, or the
independent full-reset B arm.

Attempt 9 left only `PREFLIGHT.json` in each registered GCS root. Attempt 11
showed that the observer can generate more than two thousand records, while
the redundant legacy incident ledger can reach its 2-GiB bound before the
post-exit classifier runs. The exact cause of Attempt 9's loss is unknown; the
two incidents establish that end-only packaging is not an admissible carrier.

## Contract

The M15 `layer|full` observer uses a dedicated diagnostic durability profile:

```text
CANON_P38_DURABILITY_PROFILE=m15-wide-v1
```

The profile is valid only for the signed M15 target, one diagnostic round, and
an enabled seam observer. It has four properties:

1. the legacy incident ledger is bypassed with one runtime receipt because the
   M15 replay envelope plus seam/tail records are the authoritative join input;
2. complete JSON/NPZ observer pairs are copied into bounded, immutable shards
   and each shard is uploaded and read-back verified while the worker lives;
3. the round classifier consumes the union of those locally sealed shards,
   not the mutable live observer directory;
4. the worker seals the classifier and compact bundle before acknowledging the
   learner, then writes root `COLLECTED` and `COMPLETE` markers only after the
   post-exit runner requests them.

Every marker records both the rendered expected commit and the commit resolved
from the executing checkout. A mismatch fails before the workload starts.
Periodic staging reads and hashes only records not already sealed. The final
round assembly re-hashes every sealed member once before classification. This
keeps periodic work proportional to new evidence without weakening the final
integrity gate.

## Boundedness

- one diagnostic round only;
- at most 32 observer record pairs or 256 MiB per shard;
- a single record pair larger than the byte cap is fatal;
- JSON publication is the record-completion boundary; a missing/invalid paired
  NPZ is fatal;
- no shard name or remote marker may be overwritten;
- the root manifest is generated after the terminal files and uploaded before
  `COLLECTED.json`; `COMPLETE.json` is last.

The first target retry remains one coarse round. A second round is conditional
on an explicit coverage or ambiguity finding and must use an independently
sealed round namespace.

## Local gates

- renderer and real Step-00 truth table for `m15-wide-v1`;
- neighboring non-M15 and observer-off rejection;
- legacy P38 durability profiles unchanged;
- shard positive with exact JSON/NPZ SHA and no duplicate membership;
- missing pair, tampered NPZ, oversize pair, source mismatch, and overwrite
  negatives;
- forced death after the first shard leaves a read-back-verifiable remote
  shard, no `COLLECTED`, and no `COMPLETE`;
- final assembly rejects an unsealed live record and classifies only the sealed
  union;
- terminal ordering and manifest tamper negatives;
- flag audit, syntax checks, and `git diff --check`.

## Target gate

After a separately approved publication and target launch, both arms must show:

```text
[P38.GCS] RUNTIME_SOURCE_PASS
[P38.GCS] M15_SHARD_COMPLETE ...             # at least one
[P38.GCS] M15_WIDE_ROUND_COMPLETE round=0 ...
[P38.GCS] COLLECTED ...
[P38.GCS] COMPLETE ...
```

Independent GCS verification must prove every shard manifest, the root
manifest, and the compact bundle manifest. APC-off must remain A-B/B-C exact.
APC-on may be exact or red; a red is useful only when the machine classifier
returns a joined first-red interval.

## Claim ceiling

Until the target pair passes, the maximum claim is:

```text
DURABILITY_IMPLEMENTED_HOST_PASS / EXACT_IMAGE_PASS /
TARGET_NOT_RUN / ROOT_CAUSE_NOT_LOCALIZED
```

No commit, push, exact-image execution, or target launch is implied by this
phase.

## Host result

Host durability, classifier, renderer-contract, source-mismatch, tamper,
forced-death and terminal-order gates pass on the uncommitted working tree.
The standalone repository renderer test cannot import on this host because
`metrax` is absent; the task-local renderer tests pass, and the real import
passes in the pinned exact-image gate documented below. No target result is
inferred.

## Exact-image result

Pinned image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
passes the aggregate gate with:

```text
V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=66 m15_durability=1 ...
```

The first aggregate attempt exposed an exact-image harness defect: the
read-only worktree's `.git` file named a host path that was absent in the
container. The harness now mounts the real Git common directory read-only and
marks `/workspace` safe, so `persist_p38_gcs.sh` still resolves and compares
the live checkout HEAD. A proposed mutable receipt-only replacement was
rejected and was not implemented. The stale P67 wrong-profile negative was
also corrected to expect the earlier, stronger profile-admission error.

The resulting claim ceiling is:

```text
DURABILITY_IMPLEMENTED_HOST_PASS / EXACT_IMAGE_PASS /
TARGET_NOT_RUN / ROOT_CAUSE_NOT_LOCALIZED
```

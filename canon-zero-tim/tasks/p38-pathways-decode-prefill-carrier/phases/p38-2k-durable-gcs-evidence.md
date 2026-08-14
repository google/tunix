# P38.2k: durable GCS evidence before root-cause replay

## Goal

Remove stdout and pod lifetime from the P38 evidence path. Preserve the exact
mismatch capsule, request journal, serving archive, classification, pre-align
record, and run log under a unique GCS prefix before the controlled diagnostic
exit is accepted.

## Established input

- P38s12f was a real Attempt-0 concurrency-32 run at source `b4391703`.
- It reached logical KV 1972, kept B-C exact, and reproduced A-B red at 11 /
  46,390 elements with `max_abs=0.16271209716796875`.
- Therefore concurrency 32 is insufficient and must not be repeated as a
  repair arm.
- Its `pre-alignment.jsonl` survived, but `head.full.log` ended before the
  terminal record and the committed bundle omitted both the mismatch capsule
  and serving archive. It cannot construct strict E0.

## Deliverable

1. Every rendered P38 JobSet owns exactly one prefix:

   `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0`

2. Before the workload starts, upload and read back `PREFLIGHT.json`.
3. After capsule/classification/archive creation, upload the five core
   artifacts and `SHA256SUMS`, then write `COLLECTED.json` last.
4. Write `COMPLETE.json` only after the existing depth, coverage, capture,
   journal-join, controlled-exit, and no-backward postflight gates pass.
5. Any upload/readback failure is fatal. A failed numerical or infrastructure
   run may retain `COLLECTED.json`; it must not receive `COMPLETE.json`.

## Local exit gate

- renderer contract covers the exact bucket prefix and rejects drift;
- env preflight rejects missing/incorrect GCS configuration;
- fake-GCS tests prove preflight readback, SHA-preserving collection,
  completion-last ordering, missing artifact rejection, duplicate completion
  rejection, reused-prefix rejection, and upload failure rejection;
- the existing P38 postflight suite remains green;
- the pinned-image P33 CPU/adjacent gate passes;
- exact-image Qwen3-1.7B and Qwen3-8B overlays install from the anchored image
  and retain the P38 capture contract.

Local status: PASS. The complete CPU gate ended in
`[P33.WORKLOAD] CPU_GATE PASS`; the exact-image gate ended in
`P33_EXACT_IMAGE_PASS`.

## Publication gate

Publish the final-artifact transport after its local gates pass. Do not launch
P38s13a directly from this phase: upload currently begins only after the
workload returns. P38.2l must add crash-time log/journal snapshots, rehearse and
freeze the incident schema, and then admit the known-red stock target.

## Decision after return

| Returned evidence | Decision |
|---|---|
| `COMPLETE.json`, A-B red, exact joins | Use the P38.2l decision table; build strict E0 before any seam claim. |
| `COLLECTED.json` only | Diagnose the failed postflight using the durable raw artifacts; do not claim completion. |
| `PREFLIGHT.json` only | Workload failed before core artifacts; fix infrastructure without interpreting numerical absence. |
| no `PREFLIGHT.json` | Bucket/auth/prefix setup failed; do not spend another TPU run. |

## Boundary

This phase does not change model math, precision, canonical kernels, sampling,
training, optimizer state, W&B, HF credentials, or prefix-cache behavior. It
adds GCS durability only. The current implementation does not mount a PVC;
adding live Persistent Disk writes requires the actual PVC claim name and is a
separate manifest change.

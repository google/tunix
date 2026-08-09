# P35.2 three-arm producer

Status: locally complete; r21 failed before the producer, target measurement not run

## Implemented

- The learner keeps A on the unchanged native serving path and labels its compact metadata.
- The producer finds the current first A-C mismatch and selects the exact rank-strided 16-row C
  group containing it. A no-red batch is rejected rather than reported as success.
- B submits those same source rows through native serving as one complete 16-request group,
  resets prefix cache and preserves processed temperature/top-k/top-p semantics.
- Compact metadata records verify local M256, tokens, positions, sequence/query lengths, one
  request per data rank, request distribution, active block tables, explicit prefix-cache reset,
  cache freshness and concrete mesh order.
- Mapped trainer-anchor and live engine leaves are compared bytewise on device. A one-bit change
  and signed zero both fail the equality gate.
- The schema contains A-B, B-C and direct A-C. Exact/exact is inconclusive unless the historical
  red was truly removed in the unchanged A arm; a red A-C with exact/exact is a transitivity
  failure.
- The workload is bounded to GSM8K response 256, max step 1 and no commit. The 256 cap keeps the
  native reference Splash sequence length `1024 + 256 = 1280` divisible by its fixed query block
  size 256. The producer writes one
  immutable report and intentionally exits before backward.
- The postflight accepts only exit 1, one stop marker, a nonempty report and a complete mechanical
  classification.

## Engine instrumentation

Patch `07-tpu-runner-p35-metadata.patch` adds a compact, arm-labelled P35 record instead of reusing
the large P18 tensor dump. It stores metadata arrays only; hidden states, logits, model weights and
cache contents are not serialized. Evidence paths are exclusive-create and stale paths are
rejected before launch.

## Target admission

The producer, classifier and renderer are locally complete. Attempt r21 used response 64 and
failed in the native reference Splash forward before any A/B/C report was produced. The target
measurement therefore remains NOT RUN. The response-256 repair must be published and its
source-pinned manifest must pass a server-side Kubernetes dry run before the next attempt. An
unchanged r18/r19/r21 rerun cannot answer the P35 question.

## Rollback

Leave `CANON_P35_ENVELOPE` unset. The grouped method, metadata patch and diagnostic termination are
then unreachable; normal serving, rescore, training, W&B, credentials and optimizer behavior are
unchanged.

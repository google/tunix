# P35.2 three-arm producer

Status: mixed-memory repair published; r26 failed before report, r27 target not run

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
- One request may span multiple local-M256 calls. B preserves the native request across scheduler
  invocations, while C preserves its rank-local cache across the same fixed-M sequence. The
  metadata gate reconstructs tokens and positions across all B records, checks cumulative KV
  lengths and rejects missing, duplicated or malformed chunks. Local M remains 256 on every call.
- The postflight accepts only exit 1, one stop marker, a nonempty report and a complete mechanical
  classification.

## Engine instrumentation

Patch `07-tpu-runner-p35-metadata.patch` adds a compact, arm-labelled P35 record instead of reusing
the large P18 tensor dump. It stores metadata arrays only; hidden states, logits, model weights and
cache contents are not serialized. Evidence paths are exclusive-create and stale paths are
rejected before launch.

## Target admission

Attempt r21 used response 64 and failed in the native reference Splash forward. Attempt r24
confirmed the response-256 repair and completed A, but the prototype probe rejected a real
multi-chunk sequence before B. Attempt r25 stopped in the Pathways compilation service before the
producer. Attempt r26 reached A and B execution but passed a host-memory live engine leaf and a
device-memory mapped trainer leaf to one JAX equality operation; it stopped before the report.
None of these attempts produced an A/B/C classification.

The mixed-memory repair explicitly places the host leaf into the existing device sharding one
leaf at a time, then runs the unchanged exact bytewise reduction. It records the original
memory-kind pairs and normalized-leaf count. Local, exact-image and four-chip one-host TPU gates
pass. Implementation commit `d9c2d690` is published; the current branch SHA must pass a
source-pinned server-side Kubernetes dry run before r27. An unchanged earlier attempt cannot
answer the P35 question.

## Rollback

Leave `CANON_P35_ENVELOPE` unset. The grouped method, metadata patch and diagnostic termination are
then unreachable; normal serving, rescore, training, W&B, credentials and optimizer behavior are
unchanged.

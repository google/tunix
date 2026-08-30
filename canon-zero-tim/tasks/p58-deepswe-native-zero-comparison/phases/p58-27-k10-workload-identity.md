# P58.27 — K10 common workload identity

Status: `IMPLEMENTED / HOST PASS / PINNED-IMAGE PASS / TARGET NOT RERUN`

## Incident

K10 ran source `0e954153cdfd21ee79ebf57eaa6afb4bf273aff0`. It completed
the full 128-row multi-turn rollout, produced 404,028 action tokens, rescored
all rows, and passed strict pre-alignment with zero A-B and B-C differences.
The first grouped update then stopped before forward/backward:

```text
AttributeError: 'DeepSWEWorkload' object has no attribute 'name'
```

The immutable incident is
`canon-zero-tim/evidence/p58_k10_deepswe_workload_attribute_incident/`; raw
error SHA-256 is
`08578bf7a2bad38212bf9906ef1ce184634bfbffa17d96d51ea6a90a08a91cad`.

## Root cause and repair

The shared segmented adapter consumes a common workload interface whose
identity member is `.name`. Generic DP workloads store that field directly;
DeepSWE stores the same signed identity as `contract_name`. K10 was the first
full DeepSWE target to enter the generic token-width/adaptor path far enough
to expose the interface mismatch.

`DeepSWEWorkload.name` is now a read-only property returning
`contract_name`. This keeps one source of truth and does not add a serialized
dataclass/recipe field. It closes every downstream `.name` read in the shared
adapter rather than patching only the first failing line. No recipe or
numerical setting changes.

## Gates

- Every registered P34/P39/P43/P44/P46/P58 DeepSWE workload returns
  `name == contract_name`; P44 recipe signatures contain no new `name` field.
- The real shared token-width helper accepts P58 and returns 4096/16384.
- P34 static passes ten suites; focused contract passes 6/6.
- Deterministic flag audit passes 409/409 with `changed_names=0`.
- Complete pinned P58 image gate passes with
  `deepswe_workload_identity=1` and `P58_EXACT_IMAGE_CPU_PASS`.
- The same complete gate was rerun after fast-forwarding to operator parent
  `98d102eb27fe05fcee327688d0aa6d236b32be4a`; its final container exited 0.
  This post-rebase check covers the newly landed M15 token-continuity and
  rollout neighbors but remains transcript-only construction evidence.

## Claim ceiling and next target

Source/image gates close the K10 `AttributeError`. They do not prove the
repaired DP8xTP8 segmented forward, backward, optimizer commit, checkpoint,
or 1,000-step completion. A separately approved fresh Attempt-0 must start
from the published clean SHA, preserve the K10 strict alignment receipts,
cross grouped forward/backward, and produce the first valid optimizer commit.

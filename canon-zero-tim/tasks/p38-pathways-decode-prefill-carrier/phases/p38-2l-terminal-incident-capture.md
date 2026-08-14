# P38.2l: incident-durable terminal capture

## Goal

Freeze one locally rehearsed diagnostic envelope that can return a strictly
joinable production incident for at least one real A-B mismatch. Do not launch
another 64-chip acquisition run until the mid-run durability and observer
contracts below pass locally.

## Established facts

- Concurrency 32 remained red. This proves that 256 simultaneous requests are
  not necessary; it does not remove sequential page churn, changing co-batch
  composition, cache state, or other live-serving dependencies.
- Row-231 E0-lite already ran and did not reproduce production A. Do not repeat
  mask-derived E0-lite.
- The request journal already records token history, block mapping, and the
  scheduled co-batch at prefix-band observations. It does not record every
  red request at its exact mismatch call.
- P38.2k preserves the final capsule/archive once the workload returns, but a
  pod killed during the workload can still leave only `PREFLIGHT.json`.

## Deliverable A: immutable mid-run GCS snapshots

1. Start a background snapshot worker before `CANON_RUN_CMD` and stop/wait for
   it after the command returns.
2. Snapshot only host files: the growing run log and request journal. Never
   fetch a model, cache, or device buffer in this worker.
3. Write immutable, monotonically named objects under `live/<sequence>/`, with
   a SHA manifest and a `LIVE.json` marker written last. Do not mutate or reuse
   an earlier snapshot prefix.
4. Retain the P38.2k `COLLECTED.json` and `COMPLETE.json` meanings. A live
   snapshot is crash evidence, not numerical completion.
5. Prove the contract with a local fault injection that terminates the mock
   workload between producer units and recovers the latest valid snapshot.

## Deliverable B: frozen incident schema

1. Preserve every mismatch row, bounded by the known 256-trajectory diagnostic
   geometry and an explicit byte-size guard. Do not assume 16 rows is always
   sufficient.
2. Add a compact deep-call ledger with stable joins across call id, request id,
   token-history SHA, logical position, KV length, block table, DP rank, and
   scheduled co-batch.
3. Rehearse the complete schema in the pinned Qwen3-8B one-host runner. Local
   carrier reproduction is not required; hook reachability, deterministic
   joins, bounded volume, archive integrity, and unchanged numerical outputs
   are required.
4. Freeze the instrumentation after this rehearsal. A target run must not be
   used to discover another missing field.

## Device-content observer gate

Device KV evidence is optional until it independently passes this gate. A hash
of live KV compared with an offline recomputation from token history is not a
causal discriminator because two program envelopes may legitimately produce
different KV. An admitted observer must instead relate content inside one live
program, such as write-time page content to later read-time content using the
same physical page and allocation generation.

Before target use it must prove on one host:

- observer off/on final outputs are bitwise equal on identical inputs;
- capture volume is bounded and does not OOM;
- no new numerical JIT boundary is introduced into the tested arm;
- a one-bit or poisoned-page negative control is detected.

If this gate is not met, omit device-content evidence from the first target
run and interpret the return only as strict-E0 input, not as proof of stale KV.

## Local exit gate

- P38.2k CPU and exact-image gates remain green;
- mid-run kill injection recovers a SHA-valid immutable live snapshot;
- all-red-row and exact-call join positive/negative tests pass;
- the pinned Qwen3-8B dress rehearsal passes and records its artifact sizes;
- rendered target intent remains stock, concurrency 256, Attempt 0, no
  backward, no optimizer commit, and one unique GCS prefix.

## Target gate

Only after the local exit gate, execute one stock P38s13a at concurrency 256.
Require A-B red, exact B-C, sufficient depth, complete coverage, all selected
red rows joined to incident calls, and durable GCS evidence. A green stochastic
run is inconclusive and does not prove a repair.

## Decision table

| Returned result | Decision |
|---|---|
| Strict E0 reproduces production A bitwise | Begin the first-divergence seam walk and repair the first red program boundary. |
| Strict E0 reproduces B/REF but not A | Live Pathways/cache/state remains necessary; use only observer-neutral in-situ evidence. |
| Strict E0 produces a third value | Replay harness is invalid; stop numerical interpretation. |
| Device observer proves write/read content changed for the same page generation | Investigate cache write/lifecycle ownership. |
| Device content is stable through the first red call | Move to positions/RoPE/RPA/residual/logits program seams. |

After a repair, a separate 64-chip backward-no-commit run must establish
strict A=B=C before any zero-TIM claim. One terminal capture cannot substitute
for repair validation.

## Boundary

This phase does not change training, evaluation, model math, precision,
sampling, optimizer placement, W&B, HF credentials, or prefix-cache behavior.
PVC integration remains out of scope.

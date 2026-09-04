# T9c — full-training TiTO record mode and durable extraction

- Status: implementation complete; target unrun

## Motivation

T9b proves the construction of a bounded rollout-only collector, but it stops
after the initial-policy pass and cannot observe later-turn token continuity
or numerical alignment across a 300-update FrozenLake run. The user selected a
different target: keep an explicitly selected P45/M15 exact-TiTO full train
running, preserve bounded evidence when a token difference occurs, and return
that evidence durably without silently calling the run Zero-TIM.

This phase does not weaken the historical strict contract. It adds a separate
data-collection policy whose output is a training curve plus a classified red
or green token/alignment record.

## Closed policy and admission

- Extend `CANON_P57_TOKEN_CONTINUITY_DEBUG` with the closed value
  `record-full`.
- The value is absent by default. Legacy P45/M15, ordinary exact full runs,
  `first-diff`, and `collect-64` retain their existing behavior.
- `record-full` is legal only with the generic exact selector, the registered
  P45 or M15 Zero-HP full identity, DP8xTP8, 300 expected updates, evaluation
  disabled, checkpoints disabled, and the existing alignment-warning carrier.
- It is forbidden on Native/IS, DeepSWE, rollout-only diagnostics, neighboring
  profiles, mixed selectors, or partial flag bundles.
- Base JobSet topology, autoscaling, exclusive-topology annotations, node
  selectors, and production resource requests are immutable in this phase.

## Runtime semantics

- A later-turn reconstructed-prompt versus served-prompt token difference is
  a scientific `DIFFERENT`: record one bounded capsule for that trajectory,
  increment all coverage counters, and continue the same ordinary trajectory
  into unchanged GRPO training. Do not mask, drop, retry, replace, or reweight
  the row.
- A same-request-ID submitted-versus-engine-echo token difference is handled
  the same way. A missing, duplicate, swapped, or foreign request ID is an
  infrastructure identity failure and remains fatal because its output cannot
  be safely attributed.
- The existing finite A-versus-B alignment warning may continue under the
  registered full carrier. B-versus-C, T-old-versus-T-current, non-finite
  gradients, replica disagreement, backward/optimizer faults, and evidence
  identity corruption remain fatal.
- Any token difference or allowed A-versus-B alignment difference makes the
  run `NON_ZERO_TIM_DATA_COLLECTION`. Completion is not a Zero-TIM PASS.
- `first-diff` remains immediate-fatal. `collect-64` remains rollout-only and
  never enters loss/backward/update. The new policy must not change either.

## Evidence and joins

- Add `trajectory_id` to every capsule and propagate the same stable ID into
  the returned trajectory batch metadata.
- Join token evidence to the numerical seam with
  `trajectory_id`, request ID, policy/global step, group ID, and sequence row.
  Missing or ambiguous joins are classifier failures, not guessed mappings.
- Distinguish `compared_trajectories` from
  `unexercised_single_turn_trajectories`. A trajectory with no later turn is
  never counted as token-equal.
- Count every observed trajectory and difference. Raw token capsules remain
  bounded to the first 64 different trajectories per process and are written
  atomically in a mode-0700 directory as mode-0600 files.
- Report separate verdicts:
  `execution_verdict`, `token_verdict=EQUAL|DIFFERENT|UNEXERCISED`,
  `zero_tim_verdict=PASS|FAIL`, and `evidence_verdict`.
- Backward calls, optimizer commits, alignment updates, and checkpoint writes
  must come from runtime receipts/counters. They may not be hard-coded literals.

## Durable return and observer budget

- Reuse the protected, no-clobber evidence root and pre-workload upload,
  download, and SHA probe from T9b.
- Replace periodic full-tree rehash/full-tar snapshots with immutable
  per-writer deltas. Hash and upload only newly atomically completed files;
  the final manifest references the ordered delta receipts and the complete
  local inventory.
- Run the uploader at reduced CPU/I/O priority when the platform supports it,
  retry transient failures with bounded exponential backoff, publish a health
  heartbeat, and retain a final EXIT flush without overwriting another cleanup
  trap.
- Live upload failure must not mutate training data or numerical paths. It is
  retried and makes `evidence_verdict=FAIL` if final completeness cannot be
  proved. No evidence directory, failed capsule, or remote object is deleted.

## Gates

1. Host unit tests prove closed admission, exact legacy isolation,
   record-and-continue behavior, same-row/no-mask semantics, request-identity
   fatal controls, truthful coverage accounting, bounded permissions, and
   trajectory-to-row joins.
2. GCS fake-remote tests prove incremental uploads, retry/backoff, idempotent
   finalization, tamper/missing-delta rejection, heartbeat, and trap-safe final
   flush.
3. P45 and M15 renderer tests prove two explicit full record arms while
   preserving all existing YAML topology/resource fields byte-for-byte.
4. Full P57, V1, flag audit, syntax, `git diff --check`, and complete pinned
   image gates pass.
5. A separately approved one-host TPU carrier proves observer neutrality and
   the request/trajectory join. Host or image construction cannot certify it.
6. Only a later explicit launch approval may run the P45/M15 DP8xTP8 full
   record pair. Each target returns all four verdicts, 300 measured commits,
   training curves, bounded capsules, alignment artifacts, and a verified GCS
   final manifest.

## Rollback

The full-record policy, batch join, incremental uploader, and renderer wiring
remain hunk-separable. Removing `record-full` restores the T9b state: legacy is
default, exact full is first-diff fatal, and `collect-64` is rollout-only. No
rollback may delete evidence or alter existing P45/M15 launch topology.

## Result log

- Preregistered before runtime edits. No TPU/Kubernetes launch, real GCS
  mutation, commit, or push is authorized in this phase yet.
- 2026-09-02 host gates: P57 216/216 and V1 102/102 pass. The focused
  full-record classifier covers strict green, completed scientific red, and
  broken request/capsule/counter/checkpoint evidence. The renderer covers
  legacy, per-workload exact, both-exact, first-diff, and both-exact
  `record-full` plus malformed/partial-profile negatives. Flag audit is
  421/421; Python compilation, shell syntax, and `git diff --check` pass.
- Runtime implementation: one stable trajectory ID follows each row; every
  submitted request ID is joined to that trajectory and the policy step/group/
  sequence row. Single-turn rows are `UNEXERCISED`. Same-ID token differences
  reserve at most one of 64 process-wide raw capsules for that trajectory but
  do not mask or replace its row. Missing, duplicate, swapped, or foreign
  request identity remains fatal. Backward, microbatch, commit, alignment, and
  checkpoint claims are checked against runtime receipts.
- Durability implementation: the low-priority worker uploads only new atomic
  files as immutable deltas, retries transient upload/readback failures with
  bounded backoff, publishes a local heartbeat, and finalizes only after every
  delta and the complete final inventory re-hash. Fake-remote retry, tamper,
  missing-delta, idempotence, heartbeat, and finalization gates pass. Real GCS
  is unverified.
- The first complete fixed-image run exposed a regression in diagnostic latch
  scope: the T9c per-trajectory latch had accidentally replaced the legacy
  `first-diff` process-lifetime latch. The modes now have independent latches;
  the complete image gate was rerun from the repaired tree and passed.
- 2026-09-02 immutable-image gate: `bash tests/v1_phase4/run_exact_image.sh
  tunix_frozenlake_image:vllm-tpu0.25.0` exits 0 on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  The terminal marker includes `frozenlake_tito_engine_witness=1`,
  `frozenlake_tito_collect64=1`, `frozenlake_tito_record_full=1`,
  `frozenlake_tito_gcs=1`, and `frozenlake_tito_default=legacy`.
- Claim boundary: verified by host tests and immutable-image execution;
  unverified because no one-host observer-neutrality, real-GCS, DP8xTP8 full
  run, 300-update curve, or target capsule has run. No manifest was rendered,
  and no TPU/Kubernetes launch, real GCS mutation, commit, or push occurred.

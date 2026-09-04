# T9e — all-event token-difference stream

- Status: implemented; host and pinned-image construction PASS; execution gates pending

## Motivation

The released `record-full` carrier does continue ordinary training after a
same-request token difference, except at update 0, but it preserves at most one
raw-token capsule per trajectory and at most 64 capsules per process.  That is
not enough for a data-collection run whose purpose is to study every observed
token discontinuity.  A later difference in the same trajectory, or the 65th
difference in a run, currently has only a bounded receipt and cannot be
reconstructed token-for-token.

This phase changes only the explicit, default-off `record-full` diagnostic.
It does not relax request identity, numerical, backward, optimizer, or
evidence-integrity failures.

## Closed behavior contract

- Every unequal submitted-versus-engine-echo comparison and every unequal
  later-turn actual-versus-ledger comparison is one token-difference event.
- Every event reserves a monotonically increasing process-wide event ordinal
  and writes one immutable mode-0600 capsule containing the complete actual
  and expected token arrays, their segment ledger, request/trajectory/policy
  identity, turn, group/pair attribution, hashes, and the event ordinal.
- `record-full` has no per-trajectory latch and no 64-event collection cap.
  Storage therefore grows with the number and length of differences.  This is
  an intentional diagnostic cost and makes the run ineligible as performance
  evidence.
- A same-request, structurally valid token difference at update 0 or any later
  update records evidence and the unchanged row continues through reward,
  loss, backward, and optimizer work.  It is not masked, dropped, retried,
  replaced, or reweighted.
- Missing, duplicate, swapped, or foreign request identity; malformed token
  arrays; capsule write failure; missing/duplicate/tampered capsule; GCS final
  inventory mismatch; B-C or T-old/current red; nonfinite values; and
  backward/optimizer failure remain fatal.
- The terminal classifier requires a bijection between measured difference
  events, successful capsule emissions, immutable local capsule files, and the
  final evidence inventory.  Any token difference still forces
  `zero_tim_verdict=FAIL` and claim `NON_ZERO_TIM_DATA_COLLECTION`, even when
  execution and evidence both pass.
- Historical `first-diff` and `collect-64` retain their released latch, bound,
  masking, and rollout-only semantics.  Legacy, ordinary exact, Native/IS,
  DeepSWE, GSM8K, and neighboring profiles remain unreachable from this
  change.

## Admission gates

1. Host positives create multiple differences in one trajectory, including an
   update-0 row, and prove every event gets a distinct replay capsule while the
   original training row remains unchanged.
2. A record-full unit stress reserves and emits more than 64 events without
   omission; the existing collect-64 negative still caps at exactly 64.
3. The full classifier accepts token-red update 0 as completed data collection
   only when every event is joined and persisted.  Missing, duplicate,
   reordered-ordinal, foreign-request, and tampered capsules fail evidence.
4. P57, V1, APC, flag audit, syntax/diff/secret scans, and the complete pinned
   image gate pass.  One-host observer neutrality, real GCS, and DP8xTP8 remain
   separate execution gates requiring explicit approval.

## Result log

- 2026-09-04 preregistration: verified from source that released record-full
  still stops update 0 on any token difference and that its raw capsules are
  limited to the first difference per trajectory and 64 per process.  No code
  change, target launch, commit, or push preceded this registration.
- 2026-09-04 implementation: `record-full` now reserves one monotonically
  increasing event ordinal and writes one immutable mode-0600 replay capsule
  for every unequal engine-echo or later-turn ledger comparison.  The
  per-trajectory latch and 64-event cap remain in `collect-64` but are absent
  from `record-full`.  An update-0 token difference is reported as
  `OBSERVED_DIFFERENT continue_training=1`; the unchanged row continues into
  the ordinary reward, loss, backward, and optimizer path.  Capsule write or
  final-inventory failure still stops the run because the requested replay
  evidence would otherwise be missing.
- 2026-09-04 host verification: P57 passes 234/234, V1 passes 102/102, the APC
  contract passes 12/12, and the flag audit passes 422/422 with no new flag.
  Positives cover two differences in one trajectory and 66 record-full event
  reservations; negatives reject missing, duplicate, foreign, malformed, and
  tampered event evidence.  Python/shell syntax and `git diff --check` pass.
- 2026-09-04 pinned-image verification: the complete gate exits zero on
  `tunix_frozenlake_image:vllm-tpu0.25.0`, image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`,
  with terminal `V1_HP_EXACT_IMAGE_PASS` and explicit
  `frozenlake_tito_record_full=1`, capsule-integrity, engine-witness, and GCS
  durability receipts.  The first attempt stopped in a changed test because
  that fixture lacked `import json`; only the test import was repaired before
  the successful complete rerun.
- Claim ceiling: verified by host tests and the digest-pinned installed-image
  construction gate.  A token-difference capsule is replay-complete for the
  token transport/ledger comparison; it is not a checkpoint of the model at
  every difference.  Existing bounded actor snapshots cover numerical A-B
  replay categories separately.  Matched one-host observer neutrality, real
  GCS/Orbax transport, abrupt-exit recovery, production-volume DP8xTP8,
  DeepSWE adjacency, commit, push, render, and Kubernetes launch remain
  unverified or unauthorized.
- 2026-09-04 pre-push integration: the published branch advanced through two
  P67 cluster-renderer commits to `90fd0e55`. T9e rebased without conflict.
  Post-rebase P57 234/234, V1 102/102, APC 12/12, flag audit 422/422, and the
  complete pinned-image gate all pass; the terminal remains
  `V1_HP_EXACT_IMAGE_PASS`. The user approved commit/push of this T9e concern,
  but no TPU/Kubernetes launch.

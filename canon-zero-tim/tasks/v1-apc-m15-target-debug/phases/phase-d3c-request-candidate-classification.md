# Phase D3c request-aware candidate classification

## Objective

Repair the Attempt-16 classify-stage failure without changing APC, model, or
alignment arithmetic. A token prefix is an input identity, not a serving
request identity: several concurrent requests may legitimately carry the same
`token_prefix_sha256`. The classifier must preserve those requests as a
candidate set, reject genuinely conflicting duplicates within one request and
call, and never manufacture one first-red interval from mixed candidates.

## Immutable Attempt-16 facts

- Runtime source: `af006872b64c2d6327588b4d4cef757242ddc222`.
- APC-on Round 0 reached 92.5% prefix-cache hits and produced a strict
  A-B red: 1,711 bytes / 786 elements; B-C remained 0 bytes.
- The mismatch capsule selected rows 22, 138, 178, and 181.
- Round assembly passed with 70 shards and 2,187 record pairs.
- Classification then failed on `(diagnostic_round, token_prefix_sha256)`
  because distinct request IDs were treated as aliases.
- The leading `0` in the exception tuple is diagnostic round 0, not token
  position 0.
- APC-off produced three exact numerical rounds. The returned evidence proves
  complete seal/upload/ACK for rounds 0 and 1; it shows only the round-2 seal
  request, so 3/3 terminal durability is not claimed.
- The incident manifest verifies its listed files, but it is not a complete
  treatment-round package: the raw assembled classifier input stayed local
  because classification failed before upload.

## Implementation

1. Resolve duplicates only inside the exact serving identity
   `(request_id, call_index, position, token/target identity)`.
2. Preserve distinct requests sharing one prefix as observations and group
   them only by exact numerical payload.
3. Require full-reset B to have one exact numerical variant. Multiple B
   variants remain a hard error.
4. Classify every A numerical variant, including candidates that stay exact
   through the observer. A unique first-red signature with no exact candidate
   may emit `FIRST_RED_LOCALIZED`; mixed/exact candidates emit
   `FIRST_RED_CANDIDATE_SET` with no selected layer, boundary, or fabricated
   source interval.
5. Count joined red coordinates separately from candidate anchors so prefix
   collisions cannot make coverage exceed 100% or produce a negative
   unobserved count.
6. Preserve every selected candidate record in the compact evidence bundle and
   require replay-ledger receipts for every request/call candidate.
7. Before running the classifier, durably upload the assembled round receipt,
   pre-alignment report, replay envelope, and red capsule (when present) under
   a self-hashed `classifier-input` checkpoint. Observer tensors remain in the
   already verified shards. This makes future analysis-code failures
   recoverable without another rollout.

No RoPE, attention/RPA, KV value, LM-head, loss, backward, optimizer, request
chronology, A, B, or C arithmetic changes.

## Gates

- Request-aware classifier and packager: 18/18 PASS, including same-prefix
  distinct requests, mixed signatures, same-request conflicts, B conflicts,
  tail duplicates, coverage accounting, and complete candidate packaging.
- Classifier-input checkpoint and wide durability: 11/11 PASS.
- Full task-local discovery: 151/151 PASS.
- P38 persistence integration: `PERSISTENCE_TEST_PASS`.
- Python compilation, Bash syntax, and `git diff --check`: PASS.

## Remaining gates

Pinned exact-image and a new DP8xTP8 matched pair remain separate user approval
boundaries. Source publication does not raise the numerical claim. Attempt 16
predates the classifier-input checkpoint
and cannot be reconstructed from its checked-in incident subset alone. If its
original pod-local assembled round no longer exists, the next target pair is
still required; after this phase, a later classifier failure will not require
another rollout.

## Claim ceiling

`REQUEST_AWARE_CLASSIFIER_LOCAL_PASS / PRECLASSIFY_INPUT_DURABILITY_LOCAL_PASS /
NUMERICAL_PATH_UNCHANGED / ATTEMPT16_TARGET_RED_PRESERVED /
FIRST_RED_NOT_YET_LOCALIZED / APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
EXACT_IMAGE_NOT_RUN / TARGET_NOT_RERUN / PHASE_E_CLOSED`.

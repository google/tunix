# T9d — replay-complete TiTO evidence and red-policy snapshots

- Status: T9d-3 host and pinned-image construction PASS;
  one-host/real-GCS/target unrun

## Motivation

T9c can complete a 300-update exact-TiTO record run and join token differences
to numerical alignment rows, but it does not preserve enough information to
replay a later A-versus-B red. In particular, it keeps full token arrays only
for token-continuity differences, keeps only summary coordinates for numerical
reds, and keeps no actor weights at the policy version that produced a red.
Mutable append journals are also copied only during normal finalization, so an
abrupt pod loss can discard the most useful row-map and alignment tail.

This phase adds host-only evidence. It does not change tokenization, rollout,
loss, masks, backward, optimizer, sampling, or any JAX device program.

## Closed capture contract

- Capture is reachable only through the existing default-off
  `CANON_P57_TOKEN_CONTINUITY_DEBUG=record-full` P45/M15 exact full identity.
  Legacy, ordinary exact, `first-diff`, `collect-64`, Native/IS, DeepSWE, and
  neighboring profiles remain byte-for-byte behaviorally isolated.
- Persist one immutable A/B/C trajectory sidecar for every completed alignment
  update, not only red updates. The sidecar contains the already-materialized
  host arrays: prompt/completion token IDs and masks, action mask, S-decode,
  S-prefill, T-old, policy version, stable trajectory/request/group/row joins,
  shapes, dtypes, and per-array SHA256 values. It contains no Python object
  arrays and never relies on pickle.
- Sidecars are uncompressed NPZ files to bound observer CPU cost. The writer
  uses a mode-0700 directory, mode-0600 partial file, flush/fsync, and atomic
  rename to `step-%06d.npz`. It reports bytes and write seconds. These
  instrumented runs are not performance evidence.
- Preserve append-only `full-row-map.jsonl`, `pre_alignment.jsonl`,
  `alignment.jsonl`, and `updates.jsonl` during execution by deriving immutable
  byte-range chunks containing only complete newline-terminated records.
  Previously published chunk content may never change. At finalization, the
  ordered chunks must concatenate byte-for-byte to each source journal.
- The 30-second worker poll remains the crash-durability boundary. No claim is
  made for a partial line, partial sidecar, pending snapshot, or file created
  after the latest successful poll.

## Actor-only red-policy snapshots

- The existing training checkpoint contract remains disabled and must continue
  to report zero checkpoint writes. A replay snapshot is a distinct actor-only
  evidence artifact: no optimizer state, no dataloader state, not resumable,
  and not eligible for ordinary retention or recovery claims.
- Reserve the first policy version per arm in each of four bounded categories:
  any finite A-versus-B warning, and the first warning whose maximum absolute
  difference reaches `1.0`, `8.0`, and `32.0` nats. One update may satisfy
  several categories, so the hard bound is four actor snapshots; if no
  qualifying red appears, zero snapshots is correct.
- The alignment producer writes an immutable request keyed by policy step. The
  training consumer, immediately before backward/optimizer work for that step,
  must prove its actor train step equals the requested policy version before it
  saves. A future/stale/duplicate request is fatal evidence corruption; the
  producer thread may not read or save mutable actor state directly.
- Each snapshot records source SHA, image identity, workload, DP/TP, policy
  step, trigger category/value, the complete model leaf shape/dtype inventory,
  and bounded deterministic leaf fingerprints. Saving
  must complete before the corresponding optimizer mutation. A failed required
  snapshot makes the evidence verdict fail but may not alter training rows.
- Actor snapshots are written to the registered protected evidence destination
  through a dedicated manager. They are never packed into the small-file GCS
  delta tar stream.

## Strict-classifier hardening

- `zero_tim_verdict=PASS` requires every pre/post alignment row verdict to be
  `PASS` and directly requires `blocking_reds`, `warning_reds`, and
  `reported_reds` to be empty lists. A forged `PASS` row containing any red
  entry is rejected by a poison negative.
- Sidecar count, step set, metadata, array shapes/dtypes/hashes, and row joins
  must agree with measured alignment/update receipts. Missing, duplicate,
  truncated, world-readable, pickle-bearing, or tampered sidecars fail.
- Immutable journal chunks must have a contiguous byte-range chain and exactly
  reconstruct the final journals. Missing tails, duplicate ranges, mutable
  content, and incomplete final lines fail.
- Snapshot classification distinguishes `not_triggered`, `complete`, and
  `failed`. It validates bounded trigger selection, exact pre-update policy
  step, actor-only contents, and protected destination receipts without
  reclassifying snapshots as ordinary checkpoints.

## T9d-3 admission repair (pre-registered 2026-09-03)

The first carrier audit was incomplete: it checked the pre-alignment-only M15
rehearsal and the fixed DP8xTP8 P64 capsule, but missed
`tasks/p57-frozenlake-tim-causal-study/scripts/run_perf_v2_onehost.sh`.  That
runner is the existing Qwen3-8B FrozenLake DP1xTP4 carrier with three real
backward/AdamW commits.  T9d must extend that carrier; it must not introduce a
second training vehicle.

The repair is deliberately split into four independently testable boundaries:

1. Add a default-absent diagnostic selector
   `CANON_P57_TITO_ONEHOST_NEUTRALITY=off|on`.  It opens exactly the existing
   Perf-v2 identity: P45, Qwen3-8B, DP1xTP4, three updates, strict alignment,
   APC off, checkpoint off, and the pinned image/source receipts.  Both arms
   use exact TiTO; only `on` enables local T9d sidecar/journal persistence.
   Every production DP8xTP8 identity and every neighboring workload rejects
   the selector.
2. Reuse the Perf-v2 runner and apply the `onehost-geometry-certify` decision
   rule.  The historical r7 DP1xTP4 gradient anchor is
   `6.42560338973999 / 10.10729694366455 / 7.489109516143799`; it is a
   preregistered sentinel, not permission to silently re-pin.  The off and on
   arms must have identical seven input hashes, strict A/B/C rows, the three
   commit-gradient norms, post-update fingerprints, and forward module/census
   inventories.  Input drift is `INCONCLUSIVE_INPUT_MISMATCH`.  A current off
   arm that differs from r7 stops the gate for investigation.
3. Add a startup Orbax probe distinct from the existing `gcloud storage cp`
   probe.  It uses the same Tunix `CheckpointManager`, Pathways persistence
   configuration, and protected attempt prefix as actor snapshots to save and
   restore a tiny deterministic model before rollout.  Missing readback,
   content/metadata drift, or transport failure prevents training from
   starting.  The immutable probe remains evidence.
4. Strengthen target collection without changing accepted training rows:
   record one single-writer runtime receipt; stop before the first backward if
   update 0 contains any token-continuity difference; later token differences
   remain record-and-continue.  Snapshot categories become the first finite
   red and the first red reaching each of `1`, `8`, and `32` nats.  One step
   may satisfy several categories, so the hard bound is four actor snapshots.
   Snapshot-trigger and all later wall-time rows are diagnostic-only and may
   not enter a performance comparison.

The JobSet currently has one Python `jax-tpu` application container and sixteen
Pathways server worker pods.  The Python evidence producers therefore have one
writer.  Do not add a speculative `jax.process_index()==0` branch; instead
require the runtime writer receipt and a negative that rejects duplicate
writer initialization.

## Gates

1. Host poison gates for mutable-journal chunking and reconstruction, strict
   red-list validation, all-update sidecars, permissions, tamper, joins, and
   bounded snapshot trigger selection.
2. Existing Perf-v2 DP1xTP4 observer neutrality with capture off/on: identical
   seven input hashes, A/B/C, three registered gradient anchors, post-update
   fingerprints, request/row joins, and no additional JAX modules. Sidecar and
   journal I/O time and bytes are reported separately.
3. Full P57 and V1 suites, flag audit, Python/shell syntax, `git diff --check`,
   renderer negatives, GCS fake-remote gates, and complete pinned-image gate.
4. Only after separate launch approval may the P45/M15 DP8xTP8 full record pair
   run. It must return all sidecars, journal reconstruction receipts, bounded
   red-policy snapshots when triggered, ordinary curves, and the existing four
   T9c verdicts.

## Rollback

T9d is an evidence-only extension of `record-full`. Remove the sidecar writer,
immutable journal derivation, snapshot-request consumer, classifier additions,
and matching renderer/profile wiring together. T9c's token capsules, joins,
incremental uploader, and closed carrier remain intact. Never delete existing
local or remote evidence while rolling back.

## Result log

- 2026-09-02 preregistered before T9d runtime edits. Verified by source review:
  T9c's live uploader includes immutable host/runner/capsule files but defers
  the four mutable append journals to finalization; simply globbing those files
  live would violate the uploader's immutability check. The alignment host
  sidecar already owns copied A/B/C arrays, so persistence need not introduce a
  device read or change a compiled program.
- 2026-09-02 implementation result: verified by P57 225/225, V1 102/102,
  flag audit 421/421, Python/shell syntax, and `git diff --check`. The poison
  gates cover hidden red lists under nominal PASS, missing/tampered/no-pickle
  sidecars, exact row/request/step joins, bounded snapshot categories,
  source/image/DP/TP identity, complete-line journal deltas, partial final
  lines, chunk tamper, and live reuse versus terminal full re-hash.
- 2026-09-02 immutable-image result: verified by the complete
  `tests/v1_phase4/run_exact_image.sh` terminal
  `V1_HP_EXACT_IMAGE_PASS ... frozenlake_tito_record_full=1
  frozenlake_tito_gcs=1 frozenlake_tito_default=legacy` on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  Output was observed directly and was not durably redirected, so no raw-log
  SHA is claimed.
- Verified by source placement: the alignment producer persists the update
  sidecar and immutable request; the learner consumes the request after the
  rollout-policy step is captured and before any backward/update branch. The
  dedicated synchronous manager saves full actor model state with
  `optimizer=None`; the receipt binds source SHA, image, workload, DP/TP,
  trigger, policy step, leaf inventory, and bounded fingerprint. Save failure
  preserves the training row but makes terminal evidence fail.
- Verified by source and host poison gate: live GCS polling hashes only new
  immutable files and reuses signed prior identities, preventing an
  O(total-sidecar-history) 30-second loop; finalization re-hashes all local
  evidence and every prior tar member before its final manifest.
- Unverified because the matched one-host observer-neutrality carrier, real
  GCS/Orbax snapshot transport, abrupt-pod-loss recovery, sidecar volume at
  production shape, and DP8xTP8 target have not run. No commit, push, durable
  render, TPU/Kubernetes launch, or remote mutation occurred.
- 2026-09-02 one-host carrier audit: the existing
  `scripts/run_m15_onehost_verify.sh` cannot close Gate 2 because it exits
  after pre-alignment with zero backward and zero optimizer commits. The P64
  frozen training-capsule carrier also cannot be relabelled as this gate: its
  identity, tensors, and replay contract are fixed to P45 DP8xTP8 and
  backward-no-commit. Therefore no existing command honestly proves the
  required gradient/update neutrality. Gate 2 remains unrun. A future carrier
  must be a closed DP1xTP4 one-update pair, execute identical input hashes and
  the same host gradient-fingerprint observation in both arms, vary only the
  T9d host persistence arm, and fail `INCONCLUSIVE_INPUT_MISMATCH` if its seven
  input hashes differ. It must be implemented and host-negative-tested before
  asking to occupy the direct-attached TPU.
- 2026-09-02 repeat host gate: verified again by P57 225/225, V1 102/102,
  flag audit 421/421, and `git diff --check`. This repeat made no source,
  remote, or TPU mutation and does not upgrade the one-host or target claim.
- 2026-09-03 correction: the preceding carrier conclusion is superseded.
  `run_perf_v2_onehost.sh` is an existing three-update DP1xTP4 FrozenLake
  backward/optimizer carrier.  T9d-3 will reuse it and open a closed local
  capture identity rather than create a new trainer.  Source review also
  confirmed that only the single Pathways-head Python application writes T9d
  files; the sixteen TPU worker pods run the Pathways server binary and are not
  duplicate Python writers.  Real Orbax/GCS and the one-host pair remain
  unrun.
- 2026-09-03 T9d-3 implementation result: the existing Perf-v2 carrier now has
  a closed `off|on` exact-TiTO identity and a sequential pair wrapper with a
  120-second continuous-idle check. The pair judge requires the complete
  seven-hash contract, distinguishes input drift as
  `INCONCLUSIVE_INPUT_MISMATCH`, checks the exact historical r7 gradient norms,
  post-update state fingerprints, strict alignment rows, the full canonical
  implementation ID, and equal semantic event censuses. The `on` arm writes
  sidecars/journal evidence locally but cannot request actor snapshots or GCS.
  Production record-full now also has an O_EXCL single-controller receipt, a
  distinct Tunix CheckpointManager save/restore startup probe for the actor
  snapshot destination, a before-backward update-0 token hard gate, and the
  four snapshot categories `first-any`, `first-ge-1`, `first-ge-8`, and
  `first-ge-32`.
- Verified by P57 232/232, V1 102/102, APC 31/31, flag audit 422/422, focused
  one-host judge 5/5, Python/shell syntax, and the complete pinned-image gate.
  Two stale `record-full` test callers were caught because they omitted the
  newly mandatory source/image or DP/TP identity; only their fixtures were
  repaired, without relaxing runtime admission. The final complete rerun exits
  zero on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS`. The matched v5p pair, real Orbax/GCS
  transport, abrupt-exit
  recovery, production volume, and DP8xTP8 target remain unverified. No commit,
  push, durable render, TPU/Kubernetes launch, or remote mutation occurred.
- 2026-09-04 release closeout: verified by P57 232/232, V1 102/102, APC 31/31,
  flags 422/422, Python/shell syntax, `git diff --check`, the complete
  digest-pinned image gate ending in `V1_HP_EXACT_IMAGE_PASS`, and a focused
  post-normalization installed-overlay gate ending in
  `P33_EXACT_IMAGE_PASS`. Durable raw logs and SHA256 values are registered
  under `evidence/release_closeout_20260904_r1/`. The implementation is split
  into four runtime CLs plus this documentation/evidence CL. Verified by host
  and immutable-image construction only; matched one-host, real GCS/Orbax,
  DeepSWE DP1xTP4 adjacency, and P45/M15 DP8xTP8 remain unverified because no
  TPU/Kubernetes launch was authorized.

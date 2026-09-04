# TiTO full-record and data-extraction handoff

## 1. Scope and immutable facts

- Worktree: `/home/yuxuan/code_rl_repro/worktrees/p57_tito_pair_0902`
- Package: `canon-zero-tim`
- Branch: `local/p57-tito-pair-0902`
- Historical task base: `6842edae88b5692c7d4c6ae4ecadfc9e2bf1e411`
- T9e integration base: `90fd0e55ea866c00c9084f231d5c739cfe13a233`
- Task ledger: `tasks/multiturn-tito-cross-workload`
- Published baseline bundle: five straight-line CLs. The runtime chain is
  `c5d5ddd9c25c8ef00fb8bdfeac1a5e404601f510`,
  `067cf3bf7f67bd976a361b514f245d71df829d71`,
  `dcde8a9105e2e7cd82748b7c2ffac6c0d81eb05a`, and
  `ba533dd7d8888c83d4c2ee50472a9346ccd3741c`; the fifth CL is
  documentation/evidence commit `a10c061a0bd768337403104069d7d50c3a261ffd`.
  T9e is one additive follow-up concern on that baseline. Always obtain and
  return the current full SHA with `git rev-parse HEAD`, and verify the
  published branch resolves to that same value before any target work.
- The user approved commit/push of the T9e follow-up only. That approval
  does not authorize TPU/Kubernetes, a durable manifest render, image
  publication, or any other remote mutation.
- Do not edit the base JobSet topology, autoscaling, exclusive-topology
  annotations, node selectors, or the existing production P45/M15 recipes.

## 2. Current target and policy split

Production behavior is unchanged by default: FrozenLake uses legacy token
transport, and an explicitly selected ordinary exact-TiTO full train remains
fatal on its first token-continuity difference.

T9c/T9d are implemented and T9e is the active evidence-completeness phase. T9c adds a
separate `record-full` value for explicit P45/M15 300-update exact-TiTO full
trains. T9e makes every structurally valid token-difference event write a
complete replay capsule, including update 0, repeated differences in one
trajectory, and events after the historical collect-64 bound. The same trajectory
continues unchanged through GRPO. It is not masked, dropped, retried, replaced,
or reweighted. Any such red makes the run `NON_ZERO_TIM_DATA_COLLECTION`; it
must never be reported as strict Zero-TIM success. Missing/duplicate/swapped
request identity, malformed token arrays, missing/duplicate/tampered evidence,
capsule write failure, and all non-whitelisted backward/numerical failures
remain fatal.

T9d-3 is implemented at host construction scope. It keeps that
runtime policy and adds host-only replay evidence: one immutable
A/B/C sidecar for every alignment update, crash-durable immutable chunks for
the four append journals, and at most four actor-only pre-update snapshots at
the first-any and first-`>=1`/`>=8`/`>=32`-nat A-B red policy versions. One
policy step may satisfy several categories. These snapshots
are not resumable training checkpoints, do not contain optimizer state, and do
not change the checkpoint-disabled full-recipe contract. The host gates and
complete post-T9d-3 pinned-image gate pass. Matched one-host
observer-neutrality and real GCS/Orbax transport have not run. Do not render or
launch the pair from Section 5 until those gates pass and the user separately
approves each external action.

The already implemented T9b diagnostic remains a separate P45 or M15 DP8xTP8
rollout-only carrier. It:

1. submits exact integer prompt IDs and binds every returned
   `RequestOutput.request_id` to the corresponding submitted future;
2. persists submitted and engine-echo prompt lengths/SHA256 values;
3. observes the A-path prompt slice actually present in
   `runner.input_batch.token_ids_cpu` through installed overlay patch 38 and
   persists request-ID/length/SHA256 only;
4. joins the three evidence layers by request ID;
5. captures at most 64 independent first-diff trajectory capsules, terminates
   only each different trajectory, and continues collecting;
6. executes zero backward calls, optimizer commits, checkpoint writes, and
   training-step mutations;
7. returns a mechanical `PASS|FAIL` separately from the scientific
   `token_verdict=EQUAL|DIFFERENT`.

Asynchronous TPU-runner capture order may differ from submit order. This is not
an error: request IDs are the join key, while runner record indices must be
unique and contiguous. B/rescore requests are intentionally excluded from the
runner witness.

## 3. Evidence and durability contract

In `collect-64` and `record-full`, raw token IDs exist only in atomic mode-0600
mismatch capsules under `$CANON_STATE/token-continuity-first-diff/`; their
stdout is token-free. The older immediate-fatal `first-diff` mode retains its
historical one-per-process reconstructable worker-log chunks and is not the
mode used by the full-record pair. `collect-64` retains its process-wide bound;
`record-full` has no capsule bound and assigns every diff event a unique,
contiguous event ordinal plus request/trajectory/policy/turn attribution. Its
disk and GCS volume therefore grows with the number and length of differences,
so it is diagnostic rather than performance evidence. Host and runner witnesses live under
`$CANON_STATE/p57_tito_witness/`.

Before rollout, the GCS worker performs a non-sensitive no-clobber upload,
download, and SHA verification. `90_run.sh` waits for the exact READY ACK, so a
credential, remote, worker, or readback failure prevents the workload from
starting. The low-priority worker hashes and uploads only new atomically
completed files as immutable deltas. Transient upload/readback failures receive
bounded exponential retry and a heartbeat; the training row is never changed
because of live transport state. Finalization re-hashes every prior delta,
proves its union equals the complete final inventory, and uploads a no-clobber
final manifest. Final retry reuses the same content identity.

The worker hashes only newly published files during live polls; it reuses the
signed size/SHA identity of already uploaded immutable files so multi-GB
sidecars do not create an O(total-history) scan every 30 seconds. Finalization
re-hashes all local evidence and all prior delta tar members. Abrupt pod loss
guarantees only complete files included by the most recent successful
30-second poll. A capsule killed before atomic rename, or created after that
poll, may be lost. Worker logs cannot reconstruct raw tokens.

The full-record arms must finish with these local artifacts:

- exactly one mode-0600 `p57_tito_witness/single-writer.json`
- one mode-0600 `p57_tito_gcs/orbax-probe.json` proving the distinct Tunix
  CheckpointManager save/restore path before rollout
- `p57_tito_witness/full-record-summary.json`
- `p57_tito_witness/full-row-map.jsonl`
- exactly one mode-0600
  `p57_tito_witness/update-sidecars/step-NNNNNN.npz` per completed alignment
  update; it contains token/mask/A/B/C arrays and stable row joins, never
  pickle/object arrays
- zero to four `p57_tito_witness/actor-snapshot-requests/step-NNNNNN.json`
  and matching `actor-snapshot-receipts/` files, selected mechanically from
  first-any and first `max_abs >= 1`, `>= 8`, and `>= 32` nat categories
- `p57_tito_witness/host/host-request-*.json` for echo differences
- `token-continuity-first-diff/*.json` when differences occur
- `pre_alignment.jsonl`, `alignment.jsonl`, and `updates.jsonl`
- immutable complete-line chunks under `p57_tito_gcs/journal-deltas/` and
  `p57_tito_witness/journal-reconstruction.json`
- `p33_frozenlake-dp8-tp8_full.classification.json`
- exactly one of `v1_hp_p45_full.classification.json` or
  `v1_hp_m15_full.classification.json`
- `p57_tito_full_record.classification.json`
- `p57_tito_gcs/final-manifest.json`
- `p57_tito_gcs/receipts/snapshot-*.json`

When a snapshot request exists, its synchronous trainer-thread save must
finish before backward/optimizer mutation at the same policy step. The remote
snapshot root is the run's registered evidence prefix plus
`actor-snapshots`; the artifact contains full actor model state and no
optimizer. A failed save does not modify or drop the training row, but the
terminal evidence verdict must fail. These files are replay snapshots, not
resume checkpoints.

Required terminal receipts are `[P57.TITO.FULL_RECORD] COMPLETE`,
`P57_TITO_FULL_RECORD_CLASSIFICATION`,
`[P57.TITO.FULL_RECORD] EVIDENCE`, and
`[P57.TITO.GCS] FINAL_ACKNOWLEDGED`. A `token_verdict=DIFFERENT` is valid
scientific output and the same rows still train, but its claim is
`NON_ZERO_TIM_DATA_COLLECTION`, never Zero-TIM success.

## 4. Local verification

Run from the package directory:

```bash
bash tests/p57_frozenlake_tim/run_cpu.sh
bash tests/v1_phase4/run_cpu.sh
python3 .claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo .. \
  --changed-base 6842edae88b5692c7d4c6ae4ecadfc9e2bf1e411
bash tests/v1_phase4/run_exact_image.sh tunix_frozenlake_image:vllm-tpu0.25.0
git diff --check
```

Current verified result: P57 234/234, V1 102/102, APC 12/12, flag audit
422/422, Python/shell syntax, and `git diff --check`. The P57 total includes
all-event token-difference coverage plus poison coverage for all-update sidecars,
source/image/mesh-bound actor snapshots, hidden-red rejection, complete-line
journal reconstruction, tamper detection, and live incremental SHA reuse with
terminal re-hash. Python compilation and shell syntax pass. The complete
post-T9e pinned-image gate exits zero on
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with terminal `V1_HP_EXACT_IMAGE_PASS`, including record-full,
capsule-integrity, engine-witness, and GCS-durability receipts. The earlier
  T9d durable closeout logs and their SHA256 ledger are under
`evidence/release_closeout_20260904_r1/`; the focused installed-overlay gate
was repeated after the patch-file whitespace-only normalization and ended in
`P33_EXACT_IMAGE_PASS`. These gates exercise real installed code paths but
fake actor-manager/GCS transports; real storage and one-host numerical
neutrality remain unverified. The gate caught two stale test callers missing
the new identity fields; only those fixtures were fixed, and runtime admission
was not weakened.

Before the approved T9e push, the published branch advanced from `a10c061a`
to `90fd0e55` through two P67 cluster-renderer commits. T9e rebased without a
conflict because those commits do not overlap its files. The complete P57,
V1, APC, flag-audit, and pinned-image gates above were rerun after that rebase;
the post-rebase pinned-image run again ended in `V1_HP_EXACT_IMAGE_PASS`.

### 4.1 One-host carrier is prepared but target-unrun

The gate reuses the existing three-update Qwen3-8B FrozenLake Perf-v2
DP1xTP4 runner; it does not create a second training vehicle. Do not substitute
the pre-alignment-only M15 verifier or P64 replay:

- `scripts/run_m15_onehost_verify.sh` runs three strict alignment rounds but
  deliberately executes zero backward calls and zero optimizer commits;
- P64's frozen training capsule is bound to P45 DP8xTP8,
  `backward-no-commit`, and its registered physical tensor shapes.

After checking out the published fifth CL in a clean worktree, and only after a
separate direct-TPU approval, run from the worktree root with a fresh label:

```bash
bash canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/run_tito_onehost_neutrality_pair.sh \
  UNIQUE_T9D3_LABEL
```

The wrapper observes 120 seconds of continuous idle, then runs `tito-off` and
`tito-on` sequentially. Both arms use exact TiTO, P45, DP1xTP4, three real
backward/AdamW commits, strict alignment, APC off, eval/checkpoint off, and the
same pinned source/image. Only `on` writes local sidecar/journal evidence; it
cannot write GCS or actor snapshots. The judge requires the complete
seven-hash bundles, three bitwise r7 gradient-norm anchors, state fingerprints,
12 strict alignment rows, canonical forward implementation ID, and equal
semantic event censuses. Unequal token/action-mask/policy-version hashes or
initial state fingerprints return `INCONCLUSIVE_INPUT_MISMATCH`, never PASS.
The resulting two run roots and `p57_tito_neutrality_UNIQUE_T9D3_LABEL.json`
must be returned with their SHA256 ledgers. This gate is unrun and does not
authorize a DP8xTP8 launch.

## 5. Rendering and target execution

### 5.1 Preconditions

Do not render from a dirty or unpublished tree. Rendering requires all of the
following:

1. the pushed fifth CL is read back and recorded as one full 40-character
   lowercase SHA;
2. the worktree is clean and checked out exactly at that SHA;
3. the output directory and both run IDs have never been used;
4. the user separately approves rendering and then launching the pair;
5. the matched T9d-3 one-host result is `PASS`; an inconclusive or failed pair
   does not admit the production render.

The wrapper enforces the clean-tree/HEAD/SHA/new-output conditions and only
renders; it never submits Kubernetes work.

### 5.2 Exact full-record render command

Run from `canon-zero-tim/` after substituting immutable values:

```bash
bash tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  FULL_40_CHAR_PUBLISHED_SHA \
  NEW_EMPTY_OUTPUT_DIR \
  UNIQUE_CAMPAIGN_ROOT \
  UNIQUE_P45_RUN_ID \
  UNIQUE_M15_RUN_ID \
  --token-continuity both-exact \
  --token-continuity-debug-mode record-full
```

Successful rendering prints
`V1_P67_FROZENLAKE_WAVE_READY ... token_continuity=both-exact
token_continuity_debug=record-full launch=not-executed` and creates exactly:

- `NEW_EMPTY_OUTPUT_DIR/manifest-index.json`
- `NEW_EMPTY_OUTPUT_DIR/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml`
- `NEW_EMPTY_OUTPUT_DIR/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml`

Review the index and both YAML files before launch. They must retain 300
updates, eval off, checkpoint off, DP8xTP8, the existing autoscaling,
exclusive-topology, node-selector, resources, system optimization bundle,
APC-off contract, exact TiTO, and `record-full`. The raw manifests must not
contain caller-supplied evidence destinations; the signed profile derives the
protected JobSet/attempt-specific identity at runtime.

### 5.3 Approval-gated launch commands

Only after a new explicit launch approval, run these as two separate commands
with no pipe or redirection appended:

```bash
kubectl apply -f NEW_EMPTY_OUTPUT_DIR/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml
kubectl apply -f NEW_EMPTY_OUTPUT_DIR/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml
```

Both may run concurrently when capacity is available. Never reuse either run
ID, output directory, or evidence attempt. Do not substitute the rollout-only
diagnostic renderer when the requested artifacts are 300-update curves.

### 5.4 Runtime and postflight decision

Before training begins, require `[P57.TITO.GCS] READY_ACKNOWLEDGED`,
`[P57.TITO.ORBAX_PROBE] PASS`, and `[P57.TITO.SINGLE_WRITER] PASS`. Update 0
must emit exactly one `[P57.TITO.FIRST_UPDATE_TOKEN_GATE]` observation before
backward: `PASS` when no row differs or `OBSERVED_DIFFERENT` when replay
capsules were preserved; both carry `continue_training=1`. During the
run, check the local heartbeat and the normal `[PERF]`, alignment, backward,
first-update, and optimizer receipts; a live upload retry is not allowed to
change a row. Also require one sidecar receipt per update and, when a finite
A-B red crosses a still-unseen threshold category, a same-step actor-snapshot
receipt before the update receipt.
At completion, require the four full-record receipts listed in Section 3, the
journal reconstruction receipt, and the ordinary P57/V1 classifiers.

Interpret the terminal classification mechanically:

- execution PASS + token EQUAL + Zero-TIM PASS + evidence PASS: strict
  Zero-TIM full record for the observed horizon;
- execution PASS + token DIFFERENT + Zero-TIM FAIL + evidence PASS: useful
  completed training/data-collection curve, but not Zero-TIM;
- execution/evidence FAIL, any B-C/T-old-current/non-finite/backward failure,
  or missing final acknowledgement: failed run; preserve all evidence and do
  not relabel it.

A matched one-host witness-off/on carrier remains the first scientific
observer-neutrality gate. It has not run. Host and fixed-image green do not
certify real GCS, TPU runner behavior, the 300-update curve, or DP8xTP8.

## 6. Return package

For each executed arm return:

- source SHA, image identity, JobSet/run ID, attempt number, profile, DP/TP,
  workload, and turn horizon;
- classifier JSON and SHA, including mechanical verdict and token verdict;
- summary counters: trajectories/equal/different/token-difference-events/
  reserved/emitted/omitted/emission failures, with event ordinals exactly
  `1..N` and no record-full omissions;
- measured backward, optimizer-commit, alignment-update, checkpoint, and step
  counts appropriate to the chosen carrier;
- host/engine-echo request counts and trajectory/request/step/row join result;
  only rollout-only `collect-64` claims a runner-input third witness;
- capsule inventory and per-file SHA values, without pasting raw tokens;
- sidecar count, total physical/logical bytes, total write seconds, exact step
  set, and token/mask/A/B/C plus row-join validation;
- snapshot trigger steps/categories/max-abs, actor-only model leaf inventory,
  pre-update step proof, save time, and PASS/FAIL receipt; never describe these
  artifacts as resumable checkpoints;
- journal delta ranges and final byte-for-byte reconstruction result for all
  four append journals;
- final GCS manifest path, manifest SHA, snapshot SHA, and readback receipt;
- every failure or missing artifact without deleting its evidence directory.

For the one-host admission pair additionally return both classifications and
SHA256 ledgers, the paired judge JSON/SHA, all three gradient norms, the exact
cross-arm seven-hash and state-fingerprint verdict, 12-row strict-alignment
verdict, forward implementation ID, semantic event counts, sidecar bytes/write
time, and `PASS|FAIL|INCONCLUSIVE_INPUT_MISMATCH` without re-pinning r7.

For each T9c/T9d full-record arm additionally return measured update/backward/
checkpoint counts, compared versus unexercised trajectory counts, stable
trajectory/request/step/sequence-row joins, the four separate execution/token/
Zero-TIM/evidence verdicts, and its ordinary training curve. A completed red
record arm is useful data but is not a Zero-TIM PASS.

Claim ceiling now: T9e host and pinned-image construction PASS. It does not
prove one-host observer neutrality,
real-GCS/Orbax durability, DP8xTP8
behavior, 300-update completion/convergence, or production exact-TiTO
correctness. Instrumented `record-full` timing is never performance evidence.

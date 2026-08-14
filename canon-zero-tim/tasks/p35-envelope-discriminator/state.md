# P35 envelope discriminator state

- **PRECONDITION(先读 `PRECONDITIONS.md` 再跑任何新 run,2026-08-13)**:
  P47a 后 `envelope_probe.py` 的 A 识别启发式过时,`native_A_observed` 会假红;
  未适配前的新 run 不可入证据链。
- Status: active; P35.2 target complete, P35.3 target r29/r30 infrastructure-inconclusive
- Active phase: P35.3c first-record stage localization; local gates complete, target not run
- Task directory: `canon-zero-tim/tasks/p35-envelope-discriminator/`
- Directory state: tracked
- Branch at bind: `codex/p34-scheduler-contract-0809`
- Reviewed implementation commit: `7484ab7844ca79fda6399f6f6dcd475ef8c6d632`
- Updated: 2026-08-10 UTC after r30 evidence reconciliation

## Objective

Causally separate packing/metadata geometry from wrapper/program context at the first red
production boundary, `S_prefill != T_old`, on 64-chip Pathways. Do not use backward, optimizer,
reward or correlation to classify this boundary.

## Last verified facts

1. In both returned r18 workloads the action-only `S_decode == S_prefill` boundary is bitwise
   exact, while `S_prefill != T_old` is red.
2. The historical byte counts were divided by action-token counts. That is not a token mismatch
   rate. The valid byte fractions are 20.0% for GSM8K and 23.7% for FrozenLake; an element-level
   count was not recorded.
3. GSM8K r18 used a wrong serving scheduler contract: DP16 expanded per-rank `M=4096` into a
   global `M=65536`, while the adapter used global `M=4096`, local `M=256`.
4. FrozenLake r18 already used per-rank `16/256`, so its serving and adapter canonical local M
   were both 256. M mismatch alone cannot explain the FrozenLake red boundary.
5. The real `T_old` hot path is an outer JIT plus `lax.map`, with each 256-token group calling
   the complete `runner.model_fn`. It is not the P28 per-layer segmented backward tool.
6. The generic Pathways way-count probe proves that `jit(f)` and
   `jit(value_and_grad(f)).primal` may differ even in a replicated no-reduction arm. It is a
   platform mechanism clue and a future THIRDPROG risk, not a causal proof for the current pair
   of forward-only envelopes.
7. The same-session canonical Qwen operator probe is bitwise exact across its tested program
   contexts. Canonical operators remain useful, but they do not prove the whole model envelope.
8. GSM8K r19 corrected the scheduler contract to global M4096/local M256, but the action-only
   `S_prefill != T_old` result was effectively unchanged. M mismatch is therefore excluded as
   the load-bearing carrier for that boundary.
9. P35 attempt r21 completed rollout but failed before the three-arm producer while computing
   native reference logprobs: Splash query block 256 did not divide sequence length 1088
   (`1024 + 64`). It produced no P35 report or classification and is not a numerical target.
10. P35 attempt r24 confirmed that response 256 removes the Splash divisibility failure and ran
    the unchanged A rescore, but a diagnostic-only single-chunk assertion rejected the selected
    sequence group before B. The assertion confused one request per rank with one chunk per
    request. r24 produced no report or classification and is not a carrier verdict.
11. P35 attempt r25 stopped in the Pathways compilation service before A/B/C and emitted no P35
    report. It is an infrastructure interruption, not a numerical verdict.
12. P35 attempt r26 completed rollout, native arm A, reference logprobs and two arm-B metadata
    records, then stopped in exact weight attestation because one leaf was in host memory and the
    other was in device memory. JAX rejects a single `eq` with mixed memory-space input types.
13. A four-chip one-host v5p probe reproduced the same mixed-memory exception on JAX 0.10.2.
    Explicitly placing the host leaf into the device `NamedSharding` made equal values pass and a
    changed-value negative control fail. This validates the placement repair direction, not the
    target A/B/C result.
14. P35 attempt r28 completed one source-pinned A/B/C measurement on 64-chip Pathways. A versus B
    was bitwise exact at all 3,244 selected action elements, while B versus C and direct A versus C
    each differed at 1,529/3,244 elements and 3,106/12,976 bytes. The mechanical classifier returned
    `COMPLETE / adapter_envelope_carrier` and the injected one-bit negative control was observed.
15. r28 attested all 310 mapped/live leaves bitwise equal, but every pair crossed
    `pinned_host->device`. This excludes different weight bits, not memory placement as an executable
    variable. Likewise `metadata_B_matches_C` proves the selected sequence semantics, positions,
    request distribution and active page contract; it is not a byte-for-byte equality claim for the
    complete B and C metadata/cache tensors.
16. P35.3b target r30 was Attempt 0 at source `78bde02f`. It completed rollout and wrote the
    preliminary A/B/C report, then lost the IFRT connection during `R0_live_first` record 1 of 2.
    It emitted no record-complete, replay-complete or final numerical-report marker.
17. r30 excludes accumulation across later replay records, but not state retained by the completed
    A/B/C work before replay. Its client log has no proxy/RM/worker exit reason, node event or
    memory-at-failure evidence, so the causal infrastructure mechanism remains unknown.
18. The r30 exception surfaced while calling the processed target-logprob path, but that path's
    canonical `compute_and_gather` callable is already `jax.jit`. Asynchronous error surfacing does
    not identify model, logits, sampling, logprob or gather as the failing stage.

## Current hypothesis split

- H1, packing/metadata carrier: real scheduler packing, page/block tables, positions, cache
  ownership or request grouping differ from the adapter-generated envelope.
- H2, program-context carrier: identical tensor inputs differ because the serving wrapper and
  adapter wrapper compile/lower the same `runner.model_fn` in different program contexts.
- H3, both carriers: H1 and H2 are independently load-bearing.

- H1 is not load-bearing for the measured r28 group: changing native serving from dynamic A to
  grouped B left every selected action logprob bitwise exact.
- The remaining carrier is inside the B-serving versus C-adapter envelope. P35.3 must separate
  exact captured metadata/cache inputs, weight memory placement, adapter outer-program context and
  the processed-logprob tail before naming a kernel or compiler cause.

## Completed locally

- P35 schema v2 records A-B, B-C and direct A-C element/byte counts and masked hashes.
- The producer selects the exact rank-strided C group containing the current first A-C mismatch;
  it refuses a batch that does not reproduce the known red boundary.
- A is the unchanged native serving rescore. B reuses the native serving API with one selected
  request per DP rank. C is the existing canonical adapter value.
- Compact arm-labelled engine evidence records tokens, positions, sequence lengths, query starts,
  request distribution, active block tables, prefix-cache reset, cache contract and concrete mesh order. It does not dump
  hidden states, logits, model weights or cache contents.
- Trainer-anchor leaves are mapped to live engine leaves and compared bitwise on device. Checksums
  remain provenance only; signed zero and one-bit drift are rejected.
- The classifier rejects missing arms, red attestations, ineffective negative controls,
  count/hash inconsistency, bitwise-transitivity violations and an exact A-C pair that failed to
  reproduce the known production red.
- The runner accepts only diagnostic exit 1 plus exactly one stop marker, one immutable report and
  a `COMPLETE` classification. It rejects missing marker/report and every other exit code.
- The P35 response contract is uniquely 256 in the renderer, workload command, recipe and cluster
  preflight. A renderer-to-preflight integration test accepts 256 and rejects the known-bad 64.
- B and C retain canonical local M256 while admitting sequences that span multiple scheduler
  records. The metadata gate reconstructs every rank's complete token stream across records and
  checks contiguous positions, cumulative KV lengths, at most one active request per rank,
  request distribution, active page IDs and complete sequence coverage. Missing tail chunks are
  rejected.
- Pinned-image CPU gate PASS; qwen1p7b and qwen8b overlay installs each matched all 29 manifest
  entries and passed 10/10 prompt/decode chunk tests.
- `git diff --check`, Python AST checks and shell syntax checks PASS.
- Mixed host/device exact comparison now normalizes one leaf at a time into the existing device
  sharding. Both operand orders pass on a four-chip one-host v5p; signed-zero and one-bit negative
  controls fail as required. The adapter suite, complete CPU gate and both exact-image model gates
  pass.
- P35.3 is implemented behind `CANON_P35_EXACT_REPLAY=1`. Its six-arm chain is
  B -> R0(captured/live) -> R1(captured/mapped) -> R2(adapter metadata/direct) ->
  R3(production adapter repeat) -> C(original production value). B/R0 and R3/C are hard anchors;
  placement, metadata/cache construction and outer-program boundaries are classified separately.
- The replay repeats R0, R1 and R2, reports compact target-stage equality, refuses ineffective
  negative controls and known-red reproduction failures, and prints SHA-256 for every returned
  JSON. Full model weights, full logits and cache tensors are not serialized.
- Focused pinned-image replay tests PASS; the complete P33/P35 CPU gate and both exact-image model
  installs PASS. The complete adapter/envelope suite is 40 PASS/5 skipped. A real four-device
  one-host v5p TP4 smoke passed the replay and exact-equality controls (2 PASS, 35.90s). The
  default-off implementation is published as `366ac2b1`; no target P35.3 run or cloud-resource
  lifecycle change has occurred.
- P35.3a one-host r2 passed source, install, overlay and four-device preflight but stopped before
  model forward because its offline Hugging Face cache lacked the snapshot directory named by an
  existing ref. The result is `VOID_CONTRACT`; no numerical boundary was measured. r3 will mount
  the already-present local model directory at that snapshot path read-only.
- P35.3a one-host r3 completed rollout and the real A/C production-boundary measurement on
  direct-attached DP1xTP4. The P35 selector performed an element-bitwise scan over all action
  positions and found no mismatch, so the pre-registered known-red guard stopped the run before B
  and before replay. Postflight was clean, and canonical traces were present (fixed AR 168, fixed
  embed 1, logprob M 1). The corrected verdict is `LOCAL_NOT_REPRODUCED`, not P35.3 PASS.
- The r3 schema-v1 wrapper output remains `INCONCLUSIVE` because it counted the exception's source
  echo and terminal line as two events. A tested schema-v2 reclassifier anchors the terminal line
  and returns `LOCAL_NOT_REPRODUCED` over the same immutable raw log. Raw/result-v2 SHA-256 are
  `13f77d5b13110b995582089a7a0f40be85f04dcb0e50116ee5ba240070534af6` and
  `516c1ad9c7bc3a963c856e674421df236b30a5a71b637e204310ae63903c8908`.
- P35.3 target r29 Attempt 0 ran the pinned commit `cf4c12e4` on all 64 Pathways devices, completed
  rollout and entered the first captured/live replay. It then lost the IFRT proxy connection and
  raised `UNAVAILABLE: Socket closed` while dispatching `jnp.take_along_axis`. It emitted neither
  `REPLAY_COMPLETE` nor `REPORT_COMPLETE`; P35.3 therefore remains target-inconclusive.
- r29 captured 24 A metadata records and two B records. The replay loop was over two B records,
  not 256 decode steps. Each record forms a logical float32 logits tensor of shape
  `(4096, 151936)` (about 2.49 GB), but the archived raw log contains no OOM, HBM-at-failure,
  worker termination event or evidence that this complete tensor crossed to the host. The socket
  closure is proven; memory pressure, autoscaler eviction and the causal failing operation are
  hypotheses requiring independent evidence.
- P35.3b preserves the original numerical program boundaries and serializes every captured
  record. A fused-tail candidate was rejected after changing 178/256 CPU target logprobs by about
  one ULP. The admitted path blocks all target outputs and caches before releasing logits and
  processed logits or submitting the next record.
- The learner now writes an immutable `p35_envelope.pre_replay.json` before exact replay. Cluster
  postflight collision-checks this path and prints its SHA even when replay later fails. A CPU
  negative control proves missing replay is rejected while the preliminary artifact survives.
- P35.3b local gates PASS: complete CPU contracts, both exact-image overlays, and a four-device
  one-host v5p TP4 smoke. The final two-record TP4 log has four replay-arm begin markers, eight
  matching record-complete markers and 2 passing bitwise tests in 34.72s. The first record has no
  action predictor, so the test covers the formerly unanchored tail. Raw SHA-256 is
  `2d2aca9c4c25bffd58e48a66ebe4177eeaba9068c8c86d9f983798b3121638b8`.
- P35.3c implementation is default-off and locally complete. It adds six ordered first-record
  readiness boundaries, fsynced JSONL evidence, a non-numerical classifier and a
  renderer/postflight contract that rejects missing, duplicate or reordered stages while
  preserving partial-stage localization. Focused classifier/renderer tests pass 14/14; the
  complete P33/P35 CPU contract, both exact-image
  overlays and a real four-device v5p TP4 production-shape mechanics gate pass. The TP4 gate uses
  a synthetic forward with real local M256 and vocabulary 151936; it does not run Qwen or
  Pathways and cannot localize the target failure. No target r31 run exists.

Evidence: `artifacts/p35_1_local_gate.md`, `artifacts/p35_2_local_gate.md` and
`artifacts/p35_3_local_gate.md`, `artifacts/p35_3b_local_gate.md` and
`artifacts/p35_3c_local_gate.md`.

## Next action

Use reviewed source pin `7484ab7844ca79fda6399f6f6dcd475ef8c6d632` for one separately
approved r31 Attempt 0 rendered with `--stage-probe`. The operator must archive coordinator, proxy,
resource-manager, worker and Kubernetes event evidence before deleting the JobSet. A successful
r31 stage probe still has no numerical verdict.

## Hard gates

- Missing expected measurement rows is `INCONCLUSIVE`, never PASS.
- Any red earlier gate makes downstream values from that run VOID.
- A target numerical conclusion requires all three arms A/B/C in one source-pinned run.
- Direct A-C must reproduce the known red boundary. Exact A-B plus exact B-C with red A-C is a
  transitivity failure and therefore `INCONCLUSIVE`, never a pass.
- Bitwise verdicts use `differing_elements == 0` and matching masked hashes. Correlation and
  `max_abs` are descriptive only.
- Target inputs must attest weights, mesh/device order, token IDs, validity/action masks,
  positions, attention metadata, page tables, cache initialization and canonical local M.

## Blockers

P35.3c implementation is source-pinned at `7484ab7844ca79fda6399f6f6dcd475ef8c6d632`.
One separately approved 64-chip Pathways Attempt 0 is required to classify the first failing
replay stage. The
direct-attached production run did not reproduce the known carrier and cannot replace this target
stage probe.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. The A path, training, backward,
optimizer, W&B and credentials remain unchanged. Preserve all earlier logs and handoffs.

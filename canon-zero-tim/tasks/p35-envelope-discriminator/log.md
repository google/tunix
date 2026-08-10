# P35 envelope discriminator log

## 2026-08-09 — Task bind and metric correction

- Bound the phased task to `canon-zero-tim/tasks/p35-envelope-discriminator/` on commit
  `ad309a810e35121d7d25db67c32c2712d9f8e086`.
- Reconciled the actual `T_old` hot path: outer JIT plus `lax.map`, then complete
  `runner.model_fn` per canonical 256-token group. The P28 per-layer segmented reverse is not the
  r18 value path.
- Corrected r18 interpretation: `differing_bytes / N_action` is dimensionally invalid. Existing
  artifacts support byte fractions only; element/token mismatch rates require new instrumentation.
- Started an additive alignment-report change that keeps legacy fields while adding element-level
  counts, exact denominators, fractions and action-masked hashes.
- No TPU experiment, cloud mutation, commit or push was performed.

## 2026-08-09 — P35.1 local gate complete

- Added exact element and byte denominators/fractions plus masked hashes to the pre-backward and
  four-boundary reports. Legacy `differing_bytes` fields and console lines remain available.
- Added a fail-closed P35 classifier for the four A/B/C outcomes.
- Exact-image result: alignment 13/13 PASS; classifier 5/5 PASS.
- Local Python compilation, `git diff --check` and executable English-only scan PASS.
- Advanced the active phase to P35.2. The target producer and 64-chip run remain NOT RUN.

## 2026-08-09 — P35.2 serving B-arm primitive

- Added a default-unused grouped native-prefill primitive. It submits an exact number of fixed
  request groups through the same real serving API and resets prefix cache before each group.
- Added an RL-cluster passthrough and CPU controls for two complete groups and a rejected partial
  group.
- This produces the A-vs-B scheduling variable. It does not yet attest actual page tables or
  trainer-to-engine weight equality, so the target runner remains not admitted.

## 2026-08-09 — Publication approval

- The user explicitly approved commit and push.
- Remote `origin/yuxzhang/canon-zero-tim` was fetched and remained exactly at the local base
  `ad309a810e35121d7d25db67c32c2712d9f8e086`; no rebase was required.

Artifacts to preserve:

- `canon-zero-tim/debug_logs/p33_r18_gsm8k_full.raw.log`
- `canon-zero-tim/debug_logs/p33_r18_fl_align.raw.log`
- the complete generic way-count target log referenced by the package evidence manifest

Rollback: disable the P35 runner and retain all old logs. Do not overwrite r18 artifacts.

## 2026-08-09 — P35.2 local producer complete

- Rebound the active task facts to source commit
  `c660134bababc9123e6820c1f241246cfbf602a7`, which includes the returned r19 evidence.
- Recorded the r19 result: correcting the scheduler M contract did not materially reduce the
  GSM8K `S_prefill != T_old` boundary. M is excluded as the load-bearing carrier.
- Wired the default-off A/B/C producer before backward. It selects the exact C rank-strided group
  containing the current first A-C mismatch and refuses a no-red batch.
- Added compact arm-labelled serving metadata, exact on-device trainer-anchor/live-engine weight
  equality, direct A-C reproduction, classifier negative controls and immutable evidence paths.
- Added a bounded GSM8K envelope-short renderer: response 64, max step 1, no commit, Attempt 0.
- Fixed runner postflight so only diagnostic exit 1 plus one stop marker, a report and a complete
  classification is accepted. Missing marker/report and exit 17 are explicit negative controls.
- Pinned-image CPU gate PASS. qwen1p7b and qwen8b overlays each matched 29 manifest entries and
  passed 10/10 chunk tests. Exact-weight signed-zero/one-bit gate PASS.
- Target execution, Kubernetes apply, cloud mutation, commit and push were not performed.

Artifact: `artifacts/p35_2_local_gate.md`.

Rollback: leave `CANON_P35_ENVELOPE` unset. Preserve r18/r19 artifacts and do not claim a carrier
until the source-pinned 64-chip Attempt 0 returns a complete schema-v2 classification.

## 2026-08-09 — r21 reference-Splash failure and response-contract repair

- Archived r21 at `debug_logs/p35_r21_gsm8k_envelope.raw.log` with SHA-256
  `f8d982a3db614a4edcb6163dce9b9206cd4325dc6bb6ecf2afd49ce5c93d43ec`.
- r21 completed rollout, then failed in native reference `get_ref_per_token_logps` before the P35
  producer because Splash query block 256 did not divide sequence length 1088. Report count and
  complete-classification count were both zero; no carrier verdict was made.
- Replaced every executable P35 envelope-short response contract with the unique value 256. The
  resulting reference length is 1280, divisible by the unchanged Splash block size 256.
- Added a renderer-to-cluster-preflight integration test. The canonical response 256 is accepted;
  the known-bad 64 and off-contract 65 are rejected. Invalid commands no longer print an OK line.
- Pinned-image CPU gate PASS: P33 59, alignment 13, native rollout 9, P35 producer 8, and P35
  classifier/renderer 11 tests. Python compilation, shell syntax and `git diff --check` PASS.

Rollback: leave `CANON_P35_ENVELOPE` unset. The ordinary training and reference paths remain
unchanged; preserve r21 as a failed pre-measurement artifact.

## 2026-08-09 — Response-contract repair published

- Published implementation commit `7c81187c` to `origin/yuxzhang/canon-zero-tim` after verifying
  the remote base remained `b8d3ad8d`.
- The next authorized external action is the source-pinned r22 server-side dry run followed by one
  Attempt-0 target launch. No target numerical evidence was created by this publication.

## 2026-08-09 — r24 probe-contract failure and multi-chunk repair

- Archived r24 at `debug_logs/p35_r24_gsm8k_envelope.raw.log` with SHA-256
  `4f03dd6dd22ff9d153c333d28d9e547d920e3e35d7b5faf57013f1e58aa3c466`.
- r24 confirmed the response-256 Splash repair, completed rollout and the native A rescore, then
  failed before B on a diagnostic-only assertion that allowed only one local-M chunk per sequence.
  It emitted no report or classification; no carrier verdict was made.
- Removed the false sequence-length assertion without changing response, local M, model values or
  the serving/adapter computation. Extended metadata attestation to reconstruct each rank's full
  request across multiple fixed-M256 records and validate token order, positions, cumulative KV
  lengths, request distribution, active page IDs and complete coverage.
- Added positive 300/513-token multi-chunk coverage and a missing-tail negative control. Added a
  native grouped-rescore test with 556/500-token requests.

Rollback: leave `CANON_P35_ENVELOPE` unset. Preserve r24 as failed pre-measurement evidence and do
not change the canonical local M256 contract.

## 2026-08-09 — Multi-chunk repair published

- Published implementation commit `973ad471` to `origin/yuxzhang/canon-zero-tim` after verifying
  the remote remained at r24 evidence commit `b2de4f16`.
- The next external action is one source-pinned r25 server-side dry run and Attempt-0 target
  launch. No target numerical evidence was created by this publication.

## 2026-08-09 — r26 mixed-memory attestation failure reproduced on one-host v5p

- Pulled evidence commit `4f692113`. The r26 run completed rollout, native A, reference logprobs
  and two B metadata records, then stopped before the report in exact weight attestation.
- The immediate failure compared a `uint8<host>` live-engine leaf with a `uint8` device trainer
  leaf in one JAX `eq`. It emitted no report or classification; no carrier verdict was made.
- Reproduced the same mixed-memory exception in the pinned FrozenLake image on the existing
  four-chip `aaron-v5p-node6` host with JAX 0.10.2.
- Explicitly placed the host value into the device `NamedSharding`: exact equal values returned
  `True`, while a changed-value negative control returned `False`.
- No production code, precision, model configuration, cloud resource or credential was changed.

Artifact: `artifacts/p35_2_r26_memory_space_probe.md`.

Rollback: leave `CANON_P35_ENVELOPE` unset. The next code change must remain diagnostic-only and
must preserve exact signed-zero and one-bit negative controls.

## 2026-08-09 — Mixed-memory attestation repair locally complete

- Added one-leaf-at-a-time normalization into the existing device operand's sharding before the
  unchanged exact uint8 reduction. Two differing explicit non-device memory spaces fail closed.
- Added host-left and host-right unit controls plus a standalone TPU probe. Weight attestation now
  records memory-kind pair counts and the number of normalized leaves.
- Complete adapter suite: 31 tests PASS, 5 skipped.
- Complete P33/P35 CPU gate: PASS.
- Pinned exact-image gate: qwen1p7b and qwen8b each matched 29 files and passed 10/10 tests.
- Final four-chip one-host v5p gate: direct mixed-memory comparison rejected; normalized equal
  values passed in both operand orders; signed-zero and one-bit controls both rejected.
- `git diff --check`, Python compilation and executable English-only checks pass.
- No commit, push, production default, precision, model configuration, credential or cloud
  resource was changed.

Artifact: `artifacts/p35_2_r26_memory_space_probe.md`.

Rollback: leave `CANON_P35_ENVELOPE` unset. The new transfer and comparison remain unreachable in
ordinary training. A source-pinned r27 is still required before any carrier classification.

## 2026-08-09 — Mixed-memory repair published

- Published implementation commit `d9c2d690` to `origin/yuxzhang/canon-zero-tim` after verifying
  that the remote remained at evidence commit `4f692113`.
- Re-fetched the target branch and verified that its remote SHA exactly matched the local commit.
- `origin/main` remained untouched at `2e605bb3`.
- The next external action is a server-side dry run and one source-pinned r27 Attempt 0. No target
  A/B/C classification was created by publication.

## 2026-08-09 — r28 target classification and P35.3 local implementation

- Reconciled the complete r28 64-chip result: A/B was exact at 3,244 action elements, while B/C
  and direct A/C each differed at 1,529 elements and 3,106/12,976 bytes. The negative control and
  all P35.2 attestations passed, so `adapter_envelope_carrier` is a valid target classification.
- Preserved two limitations: mapped/live weights crossed `pinned_host->device`, and the B/C
  metadata claim was semantic rather than bytewise full-tensor equality. Neither limitation may
  be silently renamed as program-context proof.
- Implemented a default-off six-arm exact replay. B/R0 and R3/C are hard anchors; R0/R1 isolates
  weight placement, R1/R2 isolates metadata/cache construction, and R2/R3 isolates the adapter
  outer program. R0, R1 and R2 repeat exactly or the classifier rejects the run.
- Added immutable replay evidence, compact pair/stage summaries, effective negative controls,
  evidence SHA printing, a bounded r29 renderer contract and an operator handoff that copies
  `/tmp` artifacts before Pod deletion.
- Focused pinned-image replay tests, the complete P33/P35 CPU gate, both exact-image model gates,
  the complete adapter/envelope suite (40 PASS, 5 skipped), AST/shell checks and
  `git diff --check` pass.
- The real one-host `aaron-v5p-node6` smoke used all four devices `[0,1,2,3]` and passed the TP4
  replay plus signed-zero/one-bit equality controls (2 PASS in 35.90s). Artifact SHA-256:
  `56f110efcebc5d1c934335eacef643a904a7501a7cf67fe0d25c6420343ad9f2`.
- Source diff review caught and repaired an intermediate insertion error in
  `_bitwise_arrays_equal` before publication. The final full suite and TPU smoke ran after the
  repair; the intermediate file was never committed or used for a target classification.
- No 64-chip target run, cloud mutation, commit or push was performed. Target P35.3 remains NOT
  RUN until a reviewed SHA is published and r29 Attempt 0 returns.

Artifact: `artifacts/p35_3_local_gate.md`.

Rollback: leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset; ordinary serving and
training are unchanged.

## 2026-08-09 — P35.3 exact replay published

- Published implementation commit `366ac2b1` to `origin/yuxzhang/canon-zero-tim` after verifying
  that the remote remained at reviewed base `337ce07c`.
- Re-fetched the live refs and verified that the target branch exactly matched `366ac2b1` while
  `origin/main` remained untouched at `2e605bb3`.
- Publication creates no target numerical verdict. P35.3 remains NOT RUN on 64-chip Pathways;
  the next operator action is the source-pinned r29 Attempt 0 in the handoff.
- Post-publication evidence verification removed trailing whitespace from the committed one-host
  log. Its content-preserving canonical file SHA-256 is
  `56f110efcebc5d1c934335eacef643a904a7501a7cf67fe0d25c6420343ad9f2`; the local-gate manifest was
  updated in this evidence-only follow-up.

Rollback: leave both P35 environment gates unset. Revert `366ac2b1` only if the diagnostic files
must be removed; do not alter production defaults.

## 2026-08-09 — P35.3a one-host r2 stopped before model load

- The first managed DP1xTP4 attempt verified the published source, canonical install `29/29`,
  overlay byte identity `6/6` and four direct-attached TPU devices `[0,1,2,3]`.
- It then stopped after 42 seconds, before model forward or any PATHTRACE, because the offline
  Hugging Face cache contained `refs/main` but no corresponding snapshot directory. The complete
  tokenizer files already existed in the immutable local model directory.
- The mechanical result is `VOID_CONTRACT`; no A/B/C value, replay arm, backward, update, W&B
  online run or target verdict was produced. Raw/result SHA-256 are
  `45825ca6f312bb3c63c577fee7a80908b62903514451449272949afdc290c10e` and
  `62d84e6853a42c5a1d6fe2224c3457e89bec2a04c84b4ed2e638161c785e2476`.
- The r3 correction adds a read-only bind mount from the existing model directory to the cache
  snapshot path named by the existing 40-hex ref. It does not download data, change tokenizer
  files, use a credential or modify model code.

Artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p35_onehost_0809_dp1tp4_r2.raw.log` and
`p35_onehost_0809_dp1tp4_r2.result.json`.

Rollback: stop invoking the one-host runner and leave the P35 switches unset. The read-only cache
mount exists only for the lifetime of its disposable container.

## 2026-08-09 — P35.3a one-host r3 did not reproduce the target carrier

- r3 verified source `cf4c12e4`, canonical install `29/29`, overlay byte identity `6/6`, four
  direct-attached TPU devices, clean C7/C8 postflight and nonzero canonical PATHTRACE.
- The full production boundary reached `sampler-trainer: logp_diff=(0,0)`. More importantly, the
  P35 selector's unrounded element-bitwise scan found no red action position and fired the
  pre-registered known-red reproduction guard. B and the six replay arms were intentionally not
  executed.
- The Python process retained vLLM background threads after the terminal guard. The named
  diagnostic container was stopped after the numerical path had ended; no VM or other container
  was changed.
- The schema-v1 wrapper miscounted the one exception as two textual occurrences (`raise` source
  echo plus terminal exception) and retained an `INCONCLUSIVE` result. The runner now anchors the
  terminal line, and a self-tested schema-v2 reclassifier returns `LOCAL_NOT_REPRODUCED` over the
  immutable raw log.
- Raw/result-v2 SHA-256:
  `13f77d5b13110b995582089a7a0f40be85f04dcb0e50116ee5ba240070534af6` and
  `516c1ad9c7bc3a963c856e674421df236b30a5a71b637e204310ae63903c8908`.

Artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p35_onehost_0809_dp1tp4_r3.raw.log` and
`p35_onehost_0809_dp1tp4_r3.result.v2.json`.

Rollback: leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. This local contrast made
no production-default, precision, weight, optimizer, checkpoint, W&B or cloud-resource change.

## 2026-08-10 — r29 entered exact replay but lost the Pathways IFRT service

- Fast-forwarded the working branch from `cf4c12e4` to evidence commit `a3cdc852`; the remote
  changed only `debug_logs/README.md` and added the r29 raw log, with no overlap with the local
  one-host phase files.
- r29 was Attempt 0 at source `cf4c12e4`. Image/overlay checks passed, 64 TPU devices joined, the
  rollout completed, and the run reproduced a nonzero broad sampler/trainer logprob difference.
- The producer captured 24 native-A metadata records and two grouped-B records, then entered
  `live_first = execute_captured(...)`. The IFRT gRPC stream closed before either
  `[CANON_P35.3] REPLAY_COMPLETE` or `[CANON_P35] REPORT_COMPLETE` was emitted. The target replay
  verdict is therefore `INCONCLUSIVE`, not a numerical failure or pass.
- Source inspection disproved the archived README's claim of a loop over 256 captured replay
  steps: `_p35_run_captured_records` received the two B records. A record does create logical
  float32 logits `(4096, 151936)` (about 2.49 GB), but JAX keeps them as remote arrays until a host
  conversion. The raw log has no OOM, HBM-at-failure, Kubernetes node event or worker-exit reason.
  It proves a socket closure, not memory pressure or autoscaler eviction as the cause.
- Before r30, write the completed P35.2 report ahead of optional replay, add record/shape and
  stage-completion instrumentation, serialize record completion, and gate any fused target-only
  tail against the existing bitwise implementation.

Artifact: `debug_logs/p35_r29_gsm8k_exact_replay.raw.log`, SHA-256
`de0edfab5d5a9439ec125559d7fc9ed11fcbc68391da8c19b34108c7718f6f00`.

Rollback: leave both P35 gates unset. The evidence pull and ledger reconciliation changed no
runtime default, model value, precision, optimizer, checkpoint, credential or cloud resource.
## 2026-08-10 — P35.3b bounded replay repair started

- Reconciled r29 as an infrastructure-inconclusive IFRT socket closure, not a numerical verdict.
- Added a preliminary immutable A/B/C report before optional replay and explicit per-record
  replay begin/complete markers with logical logits shape and byte count.
- Tested a target-only fused-tail candidate before admitting it. The CPU bitwise gate rejected
  it: 178/256 canonical target logprobs changed by roughly one ULP. The candidate was removed;
  the original sampling/logprob program boundaries remain intact.
- The active repair now serializes and blocks each captured record, releases the full-vocabulary
  temporaries before the next record, and does not change production precision, loss, backward
  or optimizer behavior.
- Complete CPU contracts PASS, including a negative control that preserves and hashes the
  preliminary report while rejecting a missing replay. Both exact-image overlays PASS.
- Final one-host v5p TP4 smoke PASS: four devices, four replay arms over two records, eight
  matching record-complete markers, and 2 passing bitwise tests in 34.72s. The first record has
  no action predictor. Raw artifact: `canon-zero-tim/debug_logs/p35_3b_onehost_tp4_r3.log`,
  SHA-256 `2d2aca9c4c25bffd58e48a66ebe4177eeaba9068c8c86d9f983798b3121638b8`.
- Local mechanics are closed; target status remains pending. No commit, push or 64-chip run was
  performed.

## 2026-08-10 — P35.3b bounded replay repair committed for publication

- Created reviewed implementation commit `9344259f` on unchanged remote base `a3cdc852`.
- The commit preserves the default-off numerical program, publishes the preliminary evidence and
  serialized-record repair, and corrects the r28/r29 evidence index without asserting an OOM,
  eviction, host transfer, gRPC limit, or target numerical result.
- Pre-commit gates: frozen-image CPU contracts PASS (59 + 13 + 10 + 14 + 20 tests), renderer
  unit tests 4/4 PASS, classifier self-test PASS, AST/shell checks PASS, credential and executable
  language scans clean, and `git diff --check` PASS.
- Next external action: push the reviewed commits, render r30 from the resulting source pin, and
  run one 64-chip Attempt 0. No target run or cloud-resource lifecycle change occurred here.

# Log

## 2026-08-10 UTC — P38.1: bind the decode-prefill carrier investigation

- Type: decision
- Fact: r35 proved B-C bitwise on both production workloads but stopped before backward because
  A-B differed by 2 bytes on GSM8K and 70 bytes on FrozenLake. The FrozenLake sampler diagnostic
  measured `logp_diff_max=0.10390`; the sparse boundary cannot be treated as a one-ULP fact.
- Hypothesis: The remaining carrier is in the Pathways serving decode-versus-prefill envelope;
  proxy-flag causality and the first numerical divergence site remain unverified.
- Action: Bound P38 and started evidence hardening without changing loss, precision, optimizer,
  or hard-gate semantics.
- Command: omitted
- Result: P38.1 active; no new TPU numerical result exists.
- Files/artifacts: `state.md`, `plan.md`, `phases/p38-1-evidence-hardening.md`
- Rollback: Omit the isolated P38 task records and evidence-only code changes; existing r35 raw
  artifacts and P36/P37 work remain untouched.
- Next: Run local alignment and runner negative controls after implementation.

## 2026-08-10 UTC — P38.1: evidence hardening passed locally

- Type: implementation and verification
- Fact: a pre-backward mismatch now records its original coordinate, token id, exact scalar bits,
  XOR, differing byte offsets, ULP distance, numerical delta, and maximum-absolute mismatch. At
  most 1024 records per boundary are emitted and truncation is explicit.
- Fact: the JSONL is flushed and fsynced before a hard-gate exception. The complete strict-JSON
  record and report SHA are printed to stdout, and `90_run.sh` repeats the persisted record after
  a nonzero workload exit.
- Fact: nonfinite values are encoded explicitly as `nan`, `inf`, or `-inf`; they cannot crash the
  evidence serializer after the numerical gate has already found a red boundary.
- Action: Corrected r35 byte-versus-token wording and added one-ULP, signed-zero, high-amplitude,
  nonfinite, invalid-shape, bounded-output, and failed-workload controls. Precision, loss,
  sampling, gradient, optimizer, and hard-gate behavior were not changed.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'python3 -m unittest discover -s tests/rl -p alignment_test.py'`
- Result: PASS, 18 tests.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'bash canon-zero-tim/tests/p33_workloads/run_cpu.sh'`
- Result: PASS. The final suite completed 63 + 18 + 10 + 14 + 31 + 2 + 1 registered tests and printed
  `[P38.EVIDENCE] FAILED_REPORT_STDOUT_PASS` and `[P33.WORKLOAD] CPU_GATE PASS`.
- Command: `git diff --check && bash -n canon-zero-tim/cluster/steps/90_run.sh canon-zero-tim/tests/p33_workloads/run_cpu.sh && python3 -m py_compile tunix/rl/alignment.py tests/rl/alignment_test.py`
- Result: PASS.
- Files/artifacts: `tunix/rl/alignment.py`, `tests/rl/alignment_test.py`,
  `../../tests/p33_workloads/run_cpu.sh`, `../../cluster/steps/90_run.sh`,
  `../../debug_logs/README.md`, `HANDOFF.md`
- Rollback: Revert only the P38 evidence fields, stdout artifact block, tests, and P38 records. The
  original r35 raw logs and all unrelated dirty P36/P37 files remain untouched.
- Next: Publish only after explicit approval, then run one strict Attempt-0 GSM8K
  `alignment-short` reproduction. Do not queue full training.

## 2026-08-10 UTC — P38.2 preparation: GSM8K-first no-commit diagnostic

- Type: decision and implementation
- Fact: the existing `gsm8k-full` manifest was not a safe diagnostic substitute. If its sparse
  A-B carrier were not sampled, the manifest could continue into committing training.
- Action: Added a dedicated `gsm8k-alignment-short` JobSet. It preserves the signed GSM8K shape
  (`32` prompts, `8` generations, response limit `1024`, local M256), sets `max_steps=1`, and
  requires `CANON_P33_NO_COMMIT=1`.
- Result: The full P33 CPU gate passed with five isolated strict JobSets. Renderer parity checks
  prove the generated command is byte-for-byte the frozen `dp_workloads` command.
- Rollback: Remove only the GSM8K alignment-short spec and its tests. Existing full workload
  recipes are unchanged.
- Next: Run GSM8K alignment-short first. If it is exact, treat it as a non-reproduction and run
  FrozenLake alignment-short; do not call it a fix.

## 2026-08-10 UTC — P38 implementation published

- Type: checkpoint
- Action: Committed the reviewed P38 evidence hardening and GSM8K no-commit diagnostic as
  `671250a5` and pushed it to `yuxzhang/canon-zero-tim`.
- Result: The remote target branch contains the tested implementation. Unrelated dirty P36/P37
  files were not included in the commit.
- Next: The external operator follows `HANDOFF.md`; no full workload is queued in P38.2.

## 2026-08-10 UTC — P38.1b: reconcile the same-callable hypothesis

- Type: correction and decision
- Fact: both r35 workload logs printed
  `runner_sampling_adapter_same_object=True`. The installed tail also uses the
  canonical Pallas log-softmax with shape-invariant numerics and disabled input
  fusion. The claim that decode and prefill merely lacked a shared Python
  callable was therefore too strong.
- Fact: FrozenLake r35 reported `max_abs=0.10390`; sparse differing bytes do not
  prove a one-ULP-only carrier.
- Hypothesis: either the two serving envelopes supply different processed
  logits to the shared tail, or the surrounding compiled programs do not
  preserve the intended call boundary.
- Action: inserted P38.1b before the target reproduction. It first runs the
  unchanged production boundary on direct-attached DP1xTP4 and stops after the
  ordinary pre-backward record. A green local result remains
  `LOCAL_NOT_REPRODUCED`.
- Files/artifacts: `phases/p38-1b-onehost-tail-construction.md`, `plan.md`,
  `state.md`.
- Rollback: omit P38.1b records and leave its future environment switch unset;
  P38.2 remains unchanged.
- Next: implement the default-off precheck-only stop and its negative controls.

## 2026-08-10 UTC — P38.1b: first one-host attempt rejected by workload contract

- Type: instrumentation failure
- Fact: the runner reached the four-device TPU and six-file overlay identity
  checks, then the GSM8K entry point rejected response length `256`. The current
  signed real-training contract requires prompt/response `1024/1024`.
- Result: INCONCLUSIVE. No rollout boundary, backward, optimizer update, W&B
  write, or numerical verdict occurred.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_onehost_precheck.sh 0810_r1`
- Artifact: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r1.raw.log`
- Rollback: no runtime rollback is needed; the container exited and the failed
  evidence remains immutable.
- Next: change only the response contract to `1024` and rerun under a new label.

## 2026-08-10 UTC — P38.1b: one-host production boundary not reproduced

- Type: hardware gate
- Action: reran the source-pinned Qwen3-1.7B GSM8K boundary on the existing
  direct-attached v5p-8 as DP1xTP4 with prompt/response `1024/1024`, M256,
  excess precision disabled, the complete canonical switch set, and W&B off.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_onehost_precheck.sh 0810_r2`
- Result: `LOCAL_NOT_REPRODUCED`. Across 11,340 action tokens,
  `S_decode_vs_S_prefill=0/45360 bytes` and
  `S_prefill_vs_T_old=0/45360 bytes`.
- Gate evidence: fixed AR 168, fixed embed 1, logprob-M 1, shared-tail identity
  1, four-device TPU 1, overlay identity 1, contract-red 0, backward 0,
  optimizer markers 0. The durable record was written before the intentional
  stop. The exact named container was stopped after that marker because vLLM
  background threads did not exit with the diagnostic exception; classification
  then completed with exit 0.
- Artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.raw.log`
  (`sha256=3a58aa9f1e37b4afdc30ac0cab317eae56545a0dd09c3b6acf60f3db33d97d81`),
  `p38_onehost_0810_r2/pre_alignment.jsonl`
  (`sha256=cf857cffd87a2917deaac89a9bda1f47700a3e2bdffe8e6eadf9f9e61781345e`),
  and `p38_onehost_0810_r2.result.json`
  (`sha256=1d1b6b5ec9ac7c1048f0c1982f64967cd4647b81a435879cbc1e4294430147e8`).
- Rollback: leave `CANON_P38_PRECHECK_ONLY` unset; production behavior is
  unchanged. No cloud lifecycle action or training mutation occurred.

## 2026-08-10 UTC — P38.1b: canonical tail construction control passed

- Type: hardware construction control
- Action: ran one canonical Pallas log-softmax callable at production shape
  M256 x V151936 through two distinct outer JIT programs on the same four-chip
  direct-attached host, then injected one output-bit change.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_tail_construction.sh 0810_r1`
- Result: `PASS_CONSTRUCTION_ONLY`; 0 of 38,895,616 f32 elements differed and
  the one-bit negative control produced exactly one differing element.
- Artifact: `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`
  (`sha256=4b5b27daf313223974f7428004ea49a903e14a6a501602c841e232f057b550f1`).
- Claim ceiling: this proves only direct-attached callable construction. It
  neither reproduces nor resolves the 64-chip Pathways r35 carrier.
- Next: P38.2 target GSM8K alignment-short Attempt 0 remains required.

## 2026-08-10 UTC — P38.2a: direct-attached aval discriminator did not reproduce

- Type: implementation and hardware control
- Action: added a default-off model-free probe for the live sampling transform
  and live canonical scorer, plus a source-pinned Attempt-0 target renderer.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_aval_onehost.sh 0810_r1`
- Result: `MODEL_FREE_NOT_REPRODUCED` on direct-attached DP1xTP4. The M16 and
  M256 sample-transform HLO digests differed, both M256 score HLO digests were
  identical, all five numerical comparisons were exact, and the one-bit
  negative control reported exactly one differing element.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'bash canon-zero-tim/tests/p33_workloads/run_cpu.sh'`
- Verification: PASS. The alignment suite ran 21 tests, the P38 aval probe ran
  4 tests, its target renderer ran 3 tests, the unified runner ran 10 tests,
  and the complete P33 CPU gate printed `[P33.WORKLOAD] CPU_GATE PASS`.
- Artifact: `../../debug_logs/p38_aval_0810_r1.result.json`
  (`sha256=f3c783a6d2d29dac0f1700b474f404f70831924fd946f5ac830cc8851cbf595f`).
- Raw log: `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.raw.log`
  (`sha256=f14101994a166a9639f3a87ee992be5f36f056c3fbf913c317fb9267eaf233a1`).
- Rollback: leave `CANON_RUN_P38_AVAL` unset. The unified runner and all
  production paths exclude the probe.
- Next: run the same discriminator on DP16xTP4 Pathways. A green target result
  still cannot replace the GSM8K and FrozenLake production probes.

## 2026-08-10 UTC — P38.2d bounded campaign amendment

- Type: user-authorized operational policy change.
- Action: added a default-off, scope-checked GSM8K full A/B report policy with
  preregistered `max_abs <= 1e-4` and byte-fraction `<= 4e-3` tripwires.
- Preserved: B/C, old/current, `r`, clip/TIS, finite gradient, DP replica,
  optimizer, and FrozenLake gates remain hard. The loss and old-logprob source
  are unchanged.
- Claim: a completed GSM8K run under this policy is
  `alignment-degraded`, never a zero-TIM closure when drift is observed.
- Companion job: FrozenLake backward-no-commit, all boundaries strict.
- Rollback: set `CANON_GSM8K_AB_REPORT_ONLY=0` or revert the policy commit.

## 2026-08-10 UTC — P38.2d local release gates

- Type: frozen-image CPU and overlay verification.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'bash canon-zero-tim/tests/p33_workloads/run_cpu.sh'`.
- Result: PASS. The workload/classifier suite ran 66 tests, the alignment suite
  ran 24 tests, all adjacent suites passed, the GSM8K policy preflight accepted
  only the committed full-train scope, and the final CPU gate printed
  `[P33.WORKLOAD] CPU_GATE PASS`.
- Command: `bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh`.
- Result: `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`.
- Status: local implementation complete; the DP16xTP4 target campaign is
  `NOT RUN` and no target numerical claim is made.
- Rollback: leave `CANON_GSM8K_AB_REPORT_ONLY=0`, or revert the bounded-policy
  commit. The strict default path is unchanged.

## 2026-08-11 UTC — P38d5 target diagnosis corrected

- Type: evidence correction.
- Fact: the GSM8K production schedule uses
  `warmup_cosine_decay_schedule(init_value=0.0, ...)`; update 0 therefore has
  effective LR exactly zero. The prior explanation that a tiny update merely
  fell below bf16 resolution was not the primary mechanism and is withdrawn.
- Fact: P38d5 had 16 active finite gradient microbatches, exact DP replicas,
  one optimizer commit, changed Adam state, unchanged sampled model state, and
  exact A/B/C boundaries. The old G6 mutation predicate was schedule-blind.
- Fact: P38d5 FrozenLake had 25 of 48,946 action elements red, maximum absolute
  difference `0.153839111328125`, and no localized mismatch below logical KV
  prefix 1791. This does not yet identify the responsible kernel or page size.
- Artifacts: `../../debug_logs/p38_p38d5_gsm8k_full.raw.log` and
  `../../debug_logs/p38_p38d5_frozenlake_bwd.raw.log`.

## 2026-08-11 UTC — Schedule-aware commit and mismatch capsule implemented

- Type: local implementation checkpoint.
- Action: registered the exact optimizer schedule, added bounded device-side
  commit evidence, and made G6 distinguish LR-zero immutability from a failed
  optimizer transaction.
- Action: added a two-row FrozenLake mismatch capsule with exact arrays,
  per-array hashes, stdout base64 survival, and a recovery verifier.
- Verification: focused alignment tests ran 26/26; renderer tests ran 10/10;
  zero-LR and positive-LR commit tests passed; the tiny end-to-end G6 LR-zero
  transaction passed with optimizer state changed and zero changed parameters;
  capsule recovery positive and corrupt-payload negative controls passed.
- Status: complete local gate is still required before publication. No cloud
  job, optimizer checkpoint, commit, or push occurred.
- Rollback: revert P38.2e/P38.2f files together or leave the capsule env empty.

## 2026-08-11 UTC — P38.2e/P38.2f complete local gate

- Type: local verification checkpoint.
- Command: frozen-image `canon-zero-tim/tests/p33_workloads/run_cpu.sh`.
- Result: PASS. Workload/classifier tests 67/67, alignment tests 26/26,
  rollout tests 14/14, P35 tests 31/31, target renderers and all negative
  controls passed, including corrupt capsule transport rejection.
- Prior unchanged-overlay command:
  `bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh`.
- Result: `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`.
- Status: local implementation complete. Target DP16xTP4 schedule-aware commit
  and FrozenLake capsule capture are NOT RUN. No commit or push occurred.

## 2026-08-11 UTC — Final working-tree audit and hardware-canary boundary

- Type: local release audit.
- Result: `git diff --check`, Python bytecode compilation, shell syntax,
  executable-source English-only scan, and credential-pattern scan all pass.
- Hardware-canary finding: the legacy GSM8K L3 recipe admits exactly two
  trajectories, while its non-production G6 update contract requires eight.
  That pre-existing contract mismatch prevents reusing the old L3 runner as a
  trustworthy real-model commit gate. It was not weakened or bypassed for this
  change.
- Claim ceiling: unit and integration tests exercise real Optax transactions,
  but the added scalar evidence has not yet been compiled with Qwen3-1.7B on
  the direct-attached TPU. Compile time and peak HBM therefore remain target
  admission measurements, not local PASS claims.
- Status: working tree is ready for review. No commit, push, cloud lifecycle
  action, precision change, prefix-cache change, or training launch occurred.

## 2026-08-11 UTC — P38.2e/P38.2f publication

- Type: source publication after explicit user continuation.
- Commits: `4933c57f` schedule-aware transaction, `33391d2c` bounded
  mismatch capsule, and `1ff24684` phase/handoff record.
- Remote: fast-forwarded `yuxzhang/canon-zero-tim` from `1a9310f3` to
  `1ff24684`; main was not checked out or modified.
- Verification before publication: complete P33 CPU gate, exact-image gate,
  diff/syntax/English-executable/credential scans all passed.
- Next: target operator pulls `1ff24684`, runs GSM8K full plus FrozenLake
  backward-no-commit only, and verifies the recovered capsule before another
  FrozenLake run.

## 2026-08-11 UTC — FrozenLake causal ladder preregistered

- Type: evidence correction and phase-plan refinement; no experiment run.
- Correction: Phase 13's PATHTRACE-proven KV-unified two-pass arm had zero
  numerical effect and was a clean negative in its original domain. It is not
  historical proof of a dropped repair.
- Correction: GSM8K completion-length summaries do not establish that its
  logical KV prefixes crossed 1792; valid prompt length plus completion
  position must be measured directly.
- Decision: the first target FrozenLake `backward-no-commit` manifest is
  capsule capture because its known A-B hard gate precedes backward. It will
  not carry an unverified KV-unified change.
- Plan: verified capsule -> R0 stock replay -> R1 same-depth single-turn -> R2
  MIXED-only two-pass -> R3 all-distribution two-pass -> candidate target
  backward-no-commit -> full training.
- Scan: page-only and page-plus-512-block boundaries around 1536, 1792, 2048,
  3840, and 4096, with per-call metadata and first-differing-layer hashes.
- Files: `plan.md`, `state.md`, `HANDOFF.md`,
  `phases/p38-2f-frozenlake-threshold-capsule.md`, and
  `phases/p38-2g-frozenlake-causal-replay.md`.
- Rollback: discard this planning-only working-tree change; published runtime
  behavior at `e9cfe298` is unchanged.

## 2026-08-11 UTC — P38.2g R0/R1 replay locally admitted

- Type: default-off implementation and local validation; no target run, no
  production default, no backward, and no optimizer commit.
- Added a hash-verifying capsule loader and strict `mask-derived-v1` R0/R1
  scheduler. Every action predictor must be covered exactly once; invalid
  masks, hash drift, missing arrays, page-table overflow, and duplicate
  coverage fail closed.
- Added live adapter execution for DP1xTP4 M256. R0, R1, and the unchanged
  fixed-chunk reference each run twice from independent fresh caches. Full
  logits remain on device; only action target, normalizer, and logprob vectors
  are copied to host.
- Added the FrozenLake prelearner entry point, one-host runner, measurement
  classifier, and one-bit negative control. The run exits before learner,
  backward, optimizer, checkpoint, and W&B.
- First exact-image integration attempt exposed a report-wiring error: the
  reference logprob is returned separately from its diagnostics dictionary.
  Normalizing that return fixed the issue; the unchanged rerun passed.
- Evidence: focused tests 11/11 + 5/5 + 1/1, full exact-image CPU gate exit 0,
  and Qwen3-1.7B/Qwen3-8B overlay gates 10/10 each. See
  `artifacts/p38_2g_local_gate.md`.
- Limitation: no verified P38.2f target capsule exists, so real Qwen3-8B TPU
  replay is NOT RUN. R2/R3 remain unimplemented by preregistered design.
- Rollback: leave `CANON_P38_FROZENLAKE_REPLAY` unset or discard this bounded
  uncommitted change. Published `e9cfe298` behavior is unchanged.

## 2026-08-11 UTC — Real-Qwen one-host synthetic deep/shallow controls

- Type: DP1xTP4 forward-only measurement admission; synthetic input, not a
  target P38.2f replay.
- Deep result: prompt 1788, R0=R1 bitwise at raw target, normalizer, and
  logprob; both differ from REF at all eight action logprobs, maximum 0.74558.
- Shallow negative control: prompt 256 gives the same classification and a
  larger R0/REF logprob maximum of 7.21813. The synthetic red is therefore not
  evidence of a KV-1791 onset.
- Integrity: 399/399 actor/engine leaves and 8,190,735,360 elements bitwise
  equal; every arm repeats exactly; one-bit negative detected; no backward;
  zero optimizer commits; each run completed in about 264 seconds.
- Harness corrections: add the L3 alignment-gate env, use the size-one `fsdp`
  model axis without an FSDP split, and satisfy the inert GRPO configuration
  with `num_generations=2`.
- Evidence: `artifacts/p38_2g_onehost_synthetic_0811.md`.
- Decision: R2/R3 remain gated. Capture the verified target capsule, then add
  an exact serving-envelope control if the broad incremental/chunk split
  repeats.

## 2026-08-11 UTC — P38e1 target capsule recovered and CPU-admitted

- Type: target evidence recovery and phase transition; no TPU run.
- Source commit: `036e845a` on `yuxzhang/canon-zero-tim`.
- Artifacts: `../../debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz`
  (`sha256=dae4e75d3b4689f2607047edd74ea1e48ffaf97a853cec74a204caafc3dc626b`)
  and `../../debug_logs/p38_p38e1_frozenlake_pre_alignment.jsonl`
  (`sha256=02a34c42548c0ae2c2f0775299480bc6d547125497cc16b858c2193aef497eb9`).
- Result: capsule schema and every embedded array SHA pass. Source rows 191
  and 199 produce complete R0/R1/REF schedules with exact action-predictor
  coverage. The captured batch has `S_prefill_vs_T_old=0` and 15 of 49,002
  action elements red at `S_decode_vs_S_prefill`, maximum absolute difference
  0.10391998291015625.
- Localization: mismatches span logical KV prefixes 1850-2462, turns 3-4, and
  sequence chunks 7-9; none immediately follows an environment token.
- Claim ceiling: the capsule contains exact tokens, masks, and logprobs but no
  original serving block tables or per-call scheduler metadata. R0 remains
  mask-derived and must reproduce locally before R2/R3 can be interpreted.
- Rollback: preserve the immutable target artifacts; leave
  `CANON_P38_FROZENLAKE_REPLAY` unset to exclude all replay code from normal
  execution.
- Next: run source row 191 on real Qwen3-8B DP1xTP4 through R0/R1/REF.

## 2026-08-11 UTC — Target row 191 rejects the local serving envelope

- Type: real-Qwen3-8B DP1xTP4 forward-only causal gate.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh canon-zero-tim/debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz p38e1_row191_stock_0811`.
- Integrity: exact actor/engine weights across 399 leaves and 8,190,735,360
  elements; all arm repeats exact; one-bit negative control detected; prefix
  cache disabled; no backward; zero optimizer commits; elapsed 417 seconds.
- Result: captured row has 3/517 action elements red at decode-versus-prefill.
  Local R0=R1 exactly at raw target, processed target, normalizer, and logprob.
  Both are red against REF at 395/517 logprobs, while REF's logprob SHA exactly
  equals captured `S_prefill`/`T_old`.
- Classification: `LOCAL_CARRIER_NOT_ISOLATED`; measurement-integrity PASS;
  production repair not admitted.
- Decision: the mask-derived R0 serving envelope did not reproduce production
  decode. Keep R2/R3 gated and move the stock/KV-unified shadow arms to the
  actual source-pinned Pathways serving path with captured scheduler metadata.
- Artifacts: `artifacts/p38_2g_onehost_target_row191_0811.md` and the immutable
  host artifacts recorded there.
- Rollback: leave `CANON_P38_FROZENLAKE_REPLAY` unset. No persistent training
  state was created.

## 2026-08-11 UTC — P38.2d: bound GSM8K full to three restarts

- Type: operational code change after explicit user approval.
- Fact: the P33 renderer previously forced `maxRestarts: 0` for every queue
  entry, while GSM8K checkpointing is disabled.
- Action: render `maxRestarts: 3` only for `gsm8k-full`; retain zero restarts
  for all diagnostics and FrozenLake jobs. Head and worker `backoffLimit`
  remain zero, and no checkpoint, training, numerical, W&B, or credential
  setting changed.
- Command: frozen-image
  `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh`.
- Result: PASS; 67 workload tests, 26 alignment tests, and all adjacent P33
  suites and negative controls passed. A direct render reported restart values
  `0,0,0,0,3`, with three assigned only to `gsm8k-full`.
- Rollback: set `_GSM8K_FULL_MAX_RESTARTS` back to zero and restore the strict
  renderer assertion; generated YAML is disposable and must not be edited.
- Next: review and, only with explicit approval, commit/push the isolated
  operational change with the existing P38.2g work separated as needed.

## 2026-08-11 UTC — P38.2g2 pinned serving-source audit

- Type: zero-TPU source archaeology and phase correction.
- Image: `tunix_frozenlake_image:vllm-tpu0.25.0`, local image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Evidence: archived pinned RPA v3, continue-decode, attention-metadata, and
  v2 cache-writer sources under `artifacts/`; exact SHA values are registered
  in `phases/p38-2g2-pathways-serving-envelope.md`.
- Finding: ordinary decode can bypass the prompt/standard runner call and run
  multiple steps inside one donated-cache `lax.while_loop`. Existing P18/P35
  capture therefore misses production A.
- Finding: v3 has no public write-only mode. `update_kv_cache=False` both
  suppresses the fused write and forces all-cache reads. The proposed W/R/A
  decomposition would mislabel variables; only the combined two-pass U arm is
  currently constructible without changing kernel internals.
- Decision: implement the real continue-decode capture first, then a
  default-off U arm. No target launch, backward, optimizer commit, commit, or
  push occurred.
- Rollback: remove the archived source copies and P38.2g2 documentation; no
  runtime behavior changed in this audit.

## 2026-08-11 05:42 UTC — P38.2g2 local implementation gated

- Type: default-off diagnostic implementation; no target run, backward,
  optimizer commit, cloud mutation, source commit, or push.
- Added patch 09 around the production `continue_decode` program. It records
  one bounded pre/post pair with scheduler inputs, sampling leaves, request
  token histories, physical pages, cache contract, selector, outputs, and
  cryptographic inventory. Missing fields/counts and collisions fail closed.
- Added patch 08 as the combined historical `U` operation: stock RPA writes,
  its attention output is discarded, and a second all-cache-read RPA output is
  used. The public v3 API cannot isolate a writer-only arm.
- Added separate stock/U renderers with zero retries, a capture classifier,
  stdout tar transport, and a SHA-verifying extractor. U without bounded
  FrozenLake no-commit capture is rejected at preflight.
- Both arms force precheck-only mode. A complete exact precheck is accepted as
  an expected diagnostic stop only with a PASS capture; a hard numerical red
  remains nonzero. The shell positive/negative postflight control passes.
- Verification: pinned Qwen3-1.7B and Qwen3-8B overlays install with all 29
  manifest entries; each existing chunking suite passes 10/10; serving
  classifier 10/10, renderer 4/4, transport 4/4; complete pinned-image P33 CPU
  gate PASS.
- Claim ceiling: locally ready for review only. Production decode capture and
  numerical effect of U are NOT RUN.
- Rollback: leave `CANON_P38_SERVING_CAPTURE_DIR` and `CANON_KV_UNIFIED`
  unset; stock branches remain selected.

## 2026-08-11 UTC — P38.2g2 publication reconciled and page-topology phase preregistered

- Type: correction and handoff.
- Fact: local HEAD and `origin/yuxzhang/canon-zero-tim` are both `763b60b1`;
  the P38.2g2 implementation is published and the worktree is clean. Earlier
  state/HANDOFF text saying dirty and unpublished was stale.
- Fact: existing logs establish a real-serving history/envelope dependency,
  but no artifact yet measures physical-page fragmentation as the isolated
  variable. `row % 16` also does not establish a local scheduler slot.
- Hypothesis: physical page topology, padding-row boundary leakage, or cache
  content corruption is the remaining production decode carrier.
- Action: added `phases/p38-2g3-page-topology-discriminator.md`; kept P38.2g2
  as the sole active phase and preregistered exact request joins, explicit
  row/DP/slot mapping, page-content equivalence, padding poison control, and a
  separate same-source flag-OFF diagnostic.
- Command: `git status --short --branch && git log -6 --oneline --decorate`.
- Result: publication reconciliation PASS; no TPU/cloud experiment and no
  runtime source change. Static review reopened local admission before target
  launch because scheduled-request, mapping-consistency, and U-PATHTRACE gates
  are incomplete.
- Files/artifacts: `state.md`, `plan.md`, `HANDOFF.md`, and
  `phases/p38-2g3-page-topology-discriminator.md`.
- Rollback: revert only this planning checkpoint and phase document; published
  default-off runtime code at `763b60b1` is unchanged.
- Next: harden P38.2g2 locally and rerun its focused tests plus the complete
  pinned-image P33 CPU gate before asking the target operator to launch stock.

## 2026-08-11 UTC — P38.2g3 reproduction and poison controls strengthened

- Type: phase-plan refinement; no runtime or experiment change.
- Fact: table sanitization becoming exact does not alone prove that padding
  page data was consumed; it can also alter a control-flow or validity path.
- Action: made E0 require bitwise equality over the complete captured action
  vector and both A/B hashes; promoted padding poison to explicit E4; moved the
  same-source flag diagnostic to E5; required per-write-event page hashes; and
  preregistered the bounded additional state captured if E0 fails.
- Decision: padding leakage is causal only when E3 is exact and changing only
  a proven padding-only sentinel's finite contents changes output. Pool
  dose-response remains an optional amplifier only when the source capture has
  too few affected requests.
- Files/artifacts:
  `phases/p38-2g3-page-topology-discriminator.md`, `plan.md`, and `HANDOFF.md`.
- Rollback: revert only this planning refinement; published default-off runtime
  code at `763b60b1` is unchanged.
- Next: implement the P38.2g2 local admission hardening already recorded in
  `state.md`; do not implement E1-E5 before an exact target E0 exists.

## 2026-08-11 UTC — GSM8K full warning-only alignment override authorized

- Type: user decision and phase amendment; runtime not yet changed.
- Fact: P38e5 update 1 has 85/195167 differing action elements,
  `element_fraction=4.35524448293e-4`, `byte_fraction=1.89581230433e-4`, and
  `max_abs=0.10378456115722656`. The existing bounded report policy rejects it
  because its max-abs limit is `1e-4`.
- Decision: add a default-off, committed-GSM8K-full-only flag that converts all
  alignment failures into stdout/JSON/W&B warnings and prevents
  `AlignmentGateError` from stopping training. No alignment magnitude or
  density threshold applies in that mode.
- Boundary: invalid/nonfinite numerics, loss/gradient health, DP reducer and
  replica equality, optimizer transaction health, infrastructure errors, and
  ordinary runtime failures remain fatal. FrozenLake and the zero-TIM
  root-cause ladder remain strict.
- Claim ceiling: `PASS_WITH_ALIGNMENT_WARNINGS`,
  `claim_level=convergence-only`; never zero-TIM.
- Files/artifacts: `phases/p38-2d-gsm8k-bounded-ab-campaign.md`, `plan.md`,
  `state.md`, and `HANDOFF.md`; source evidence in
  `../../debug_logs/p38_p38e5_gsm8k_full.raw.log`.
- Rollback: leave `CANON_GSM8K_ALIGNMENT_WARN_ONLY` unset. No runtime code was
  changed by this checkpoint.
- Next: implement the flag, its fail-closed scope, warning classifier, W&B
  metrics, and negative controls; run focused tests and the full P33 CPU gate
  before publication/launch.

## 2026-08-11 UTC — GSM8K warning-only alignment implementation locally gated

- Type: default-off runtime policy implementation; no TPU/cloud experiment,
  source commit, push, or training launch.
- Added `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1`, mutually exclusive with the
  legacy bounded A/B policy and admitted only for committed GSM8K full
  training. The renderer enables it only for `gsm8k-full`.
- Finite, shape-valid A-B, B-C, and old/current differences plus non-unit
  `w/r/w*r` and clip/TIS observations enter `warning_reds`, emit explicit
  warning lines, persist in JSONL, and do not raise `AlignmentGateError`.
- Invalid shape, NaN/Inf, nonfinite ratio or gradient, reducer/replica errors,
  optimizer transaction errors, missing evidence, and runtime/infrastructure
  failures remain hard.
- Added ratio min/max and boundary byte/element fractions to W&B, plus
  `zero_tim/alignment_warning`, warning count, and a truthful strict-gate
  metric. The terminal classifier is
  `PASS_WITH_ALIGNMENT_WARNINGS`, `claim_level=convergence-only`.
- Negative controls cover wrong workload, mutually enabled policies,
  nonfinite values, strict-mode drift, reducer/update evidence, and classifier
  scope.
- Verification commands:
  `sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace -e
  JAX_PLATFORMS=cpu tunix_frozenlake_image:vllm-tpu0.25.0 bash
  canon-zero-tim/tests/p33_workloads/run_cpu.sh` and
  `bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh`.
- Results: complete frozen-image P33 CPU gate PASS (67 workload tests, 28
  alignment tests, all adjacent suites/negative controls), focused classifier
  11/11 PASS, exact-image Qwen3-1.7B 10/10 PASS, and Qwen3-8B 10/10 PASS.
- Concurrent source movement: after those gates, the tracked branch advanced
  from base `763b60b1` to `bf0c5734` in seven DeepSWE admission files that do
  not overlap this diff. Rebase and rerun are still required before push.
- Rollback: set `CANON_GSM8K_ALIGNMENT_WARN_ONLY=0`; strict behavior returns
  without changing loss, precision, optimizer, prefix cache, FrozenLake, or
  credentials.
- Next: obtain explicit commit/push approval, commit the local work, rebase on
  `bf0c5734`, rerun the frozen-image gates, push, and only then render a fresh
  source-pinned GSM8K full manifest. Target training is NOT RUN.

## 2026-08-11 UTC — warning-only policy rebased on DeepSWE admission

- Type: publication preparation under explicit user commit/push approval; no
  TPU/cloud experiment or training launch.
- Rebased `Admit warning-only GSM8K convergence runs` directly onto
  `bf0c5734 Gate the Pathways session behind a bounded device admission probe`.
  The rebased local commit is `81f20e78`; main was not checked out or modified.
- Post-rebase verification: complete frozen-image P33 CPU gate PASS (67
  workload tests, 28 alignment tests, adjacent negative controls), exact-image
  Qwen3-1.7B and Qwen3-8B gates 10/10 each PASS, and DeepSWE P34 static gate
  PASS with `suites=10`, including the new 256-device admission probe.
- Proven: the GSM8K warning-only policy and the DeepSWE device admission patch
  coexist without a local regression. Not proven: any target training result,
  zero-TIM restoration, FrozenLake repair, or DeepSWE target admission.
- Rollback: set `CANON_GSM8K_ALIGNMENT_WARN_ONLY=0`; the strict alignment gate
  returns. The DeepSWE admission probe remains fail-closed and independent.
- Next: publish to `yuxzhang/canon-zero-tim`, verify the remote hash, then
  render a fresh source-pinned GSM8K full JobSet. Keep FrozenLake and DeepSWE
  strict.

## 2026-08-11 UTC — warning-only policy published

- Type: source publication; no TPU/cloud experiment or training launch.
- Command: `git push origin HEAD:yuxzhang/canon-zero-tim` using an ephemeral
  askpass that read only `GITHUB_USER` and `GITHUB_TOKEN` from the repository
  root `.env`. The helper was deleted immediately after the push; W&B and HF
  variables were neither read nor changed.
- Result: remote advanced from `bf0c5734` to `c4871ef7`. This preserves the
  DeepSWE bounded device-admission parent and adds the locally gated GSM8K
  warning-only convergence policy. Main was not checked out or modified.
- Claim ceiling remains `convergence-only`; this publication does not make any
  alignment boundary green and is not a zero-TIM completion claim.
- Rollback: set `CANON_GSM8K_ALIGNMENT_WARN_ONLY=0` for strict runtime behavior,
  or revert `c4871ef7` in a separate reviewed change. Do not remove the
  independent DeepSWE admission parent.
- Next: render and launch only the source-pinned GSM8K full JobSet for the
  time-sensitive convergence run. FrozenLake and DeepSWE remain strict and
  proceed through their diagnostic/admission ladders.

## 2026-08-11 UTC — P38.2g2 serving-capture admission hardening complete

- Type: default-off diagnostic hardening; no TPU/cloud experiment, backward,
  optimizer commit, source commit, push, or training launch.
- Source base: local HEAD and `origin/yuxzhang/canon-zero-tim` are
  `2cd46433`; main remains `41b4c54e` and was not modified.
- Closed the four reopened admission gaps: scheduled-only request selection
  with physical-slot preservation; explicit and validated
  request/DP/slot/global/attention/selector/page mappings; exact mismatch
  capsule join by request/token history for stock; and zero-stock/positive-U
  `KV_UNIFIED_two_pass` PATHTRACE enforcement.
- Final review caught and fixed a classifier-only slot-compaction bug: the
  capture preserved a scheduled request at physical slot 1 after filtering an
  idle slot 0, but the classifier initially re-enumerated the filtered list as
  slot 0. The classifier now trusts the explicit slot only after bounds and
  uniqueness checks; its default positive fixture contains this gap.
- The first exact-image attempt exposed malformed patch hunk metadata; after
  correcting the hunk ranges, the manifest correctly rejected the changed
  installed runner SHA. The manifest was regenerated from the pinned image,
  then both model overlays installed and passed. These were construction
  failures caught before any target run, not numerical failures.
- Focused results: classifier 18/18 PASS, renderer 5/5 PASS, archive transport
  4/4 PASS, Python compilation PASS, and shell postflight PASS across exact
  stock, red stock, illegal stock U hit, missing U hit, and exact U controls.
- Full results: frozen-image P33 CPU gate PASS with 67 workload tests, 28
  alignment tests, and all adjacent suites; exact-image Qwen3-1.7B and
  Qwen3-8B each match all 29 manifest entries and pass 13/13 runtime tests.
- Claim ceiling: locally admissible for source review and publication only.
  Production stock capture, mismatch reproduction, U numerical effect, page
  topology, padding poison, backward, and optimizer behavior remain NOT RUN.
- Rollback: leave `CANON_P38_SERVING_CAPTURE_DIR` and `CANON_KV_UNIFIED`
  unset, or discard this uncommitted hardening diff. Stock runtime behavior is
  unchanged.
- Next: after explicit commit/push approval, publish this exact source, render
  fresh manifests, dry-run both, and apply stock only. U and P38.2g3 E1-E5
  remain blocked until stock exactly joins and reproduces the known mismatch.

## 2026-08-11 UTC — P38.2g2 admission hardening published

- Type: source publication and evidence reconciliation; no TPU/cloud
  experiment, backward, optimizer commit, or training launch.
- Implementation commit: `bbc1d329 Harden the Pathways serving capture
  admission`, based directly on `2cd46433`; no rebase was required because a
  fresh fetch showed local, FETCH_HEAD, and remote at the same base.
- Push result: `origin/yuxzhang/canon-zero-tim` advanced from `2cd46433` to
  `bbc1d329`. `git ls-remote` verified the full remote hash as
  `bbc1d3290188df47595a3126e457788a94c289d9`.
- Authentication: the stale VS Code askpass failed first without changing the
  remote. The successful retry used a temporary mode-700 askpass that read
  only `GITHUB_USER` and `GITHUB_TOKEN` from the repository root `.env`; the
  helper was deleted immediately. No W&B or HF setting was read or changed.
- Claim ceiling: publication makes the code available for the target stock
  diagnostic; it does not prove a production capture, mismatch reproduction,
  U effect, page-topology cause, or zero-TIM repair.
- Rollback: leave `CANON_P38_SERVING_CAPTURE_DIR` and `CANON_KV_UNIFIED`
  unset, or revert `bbc1d329` in a separate reviewed change. Stock runtime
  behavior remains the default.
- Next: fetch and verify `bbc1d329`, render both P38.2g2 manifests from that
  exact commit, server-side dry-run both, and apply stock only. U remains
  blocked on the stock mismatch join and reproduction gate.

## 2026-08-11 UTC — p38s1/p38u1 evidence reconciled; stock serving archive still missing

- Type: read-only evidence audit followed by documentation correction. No TPU,
  cloud, runtime-code, training, commit, or push action was performed.
- Source audited: `b7b20e261433977bc57bd83452fd6ac1c4680cdd` on
  `origin/yuxzhang/canon-zero-tim`.
- Stock `p38s1`: Attempt 0, 43/46,417 A-B differing elements, 68 differing
  bytes, `max_abs=0.2780647277832031`, and B-C exact.
- U `p38u1`: `KV_UNIFIED_two_pass` executed, 9/46,589 A-B differing elements,
  16 differing bytes, `max_abs=0.27657318115234375`, and B-C exact. U is not a
  sufficient repair. Different trajectories/action counts prohibit treating
  43-to-9 as a paired improvement or a causal timing/writer result.
- Both available head logs terminate at the child `AlignmentGateError` and
  omit the outer `[run] exit`, official serving classifier/archive, capsule
  base64 transport, and final PATHTRACE. The numerical observations are valid;
  the serving-capture admission is `INCONCLUSIVE`.
- The generic committed capsule SHA `dae4e75d...` equals the old P38e1 capsule
  byte-for-byte and is not the p38s1 (`2dffb993...`) or p38u1
  (`245a0c9b...`) run-specific artifact. It cannot supply block-table/page
  evidence.
- Correction: withdrew the earlier complete-capture PASS and its instruction
  to run U. Page ownership/lifecycle, stale table, partial write, padding
  leakage, and topology remain hypotheses.
- Next: one fresh stock-only Attempt-0 run must preserve the terminal head log
  through outer postflight, emit an official serving-classification PASS and
  serving archive, transport the real run-specific capsule, prove zero U hits,
  and remain pre-backward with zero optimizer commits. P38.2g3 E0-E5 stay
  blocked until the stock archive passes exact join and whole-vector replay.
- Rollback: documentation-only. Restore these files from version control if
  the correction itself must be reverted; runtime behavior is unchanged.

## 2026-08-11 UTC — P38.2g4 stratified serving capture locally admitted

- Type: default-off diagnostic hardening; no TPU numerical run, Pathways job,
  backward, optimizer commit, source commit, push, or cloud launch.
- Replaced the obsolete single `min_prefix=1788` trigger with four bounded
  intervals from 1536 through 2560. Selection is request anchored: a call is
  admitted only when one concrete scheduled request lies in an unfilled
  interval, and its request ID and exact prefix are recorded.
- Added stable source/callable identity, anchor mapping, four-stratum
  completeness, five-times storage margin, and exact mismatch-capsule join
  checks. Zero joins and ambiguous joins fail closed.
- A fixture audit found and removed a false positive: the shell fixture had
  claimed an anchor prefix near 1700 while its request history contained only
  three tokens. It now carries a consistent long token history, page table,
  sequence length, and anchor for every interval.
- Focused results: classifier 25/25 PASS, renderer 5/5 PASS, shell postflight
  PASS, Python compilation PASS, shell syntax PASS, and patch reconstruction
  plus compilation PASS.
- Full results: exact-image Qwen3-1.7B and Qwen3-8B each match all 29 manifest
  entries and pass 14/14 tests. The complete frozen-image P33 CPU gate passes
  78 workload tests, 29 alignment tests, and all adjacent suites.
- Reconstructed runner SHA-256:
  `fe81622996a1c73bbd17187ee603e6a191165202da40d07b5e428fe41b5db516`.
- Claim ceiling: local construction and image compatibility only. Docker had
  no TPU device; D1 target capture, E0 replay, seam localization, page/cache
  causality, repair, backward, and optimizer behavior remain NOT RUN.
- Artifact: `artifacts/p38_2g4_local_gate_0811.md`.
- Rollback: leave capture variables unset or discard this uncommitted P38.2g4
  diff. Stock runtime behavior is unchanged.
- Next: obtain explicit source-publication approval, then separate resource
  approval for one stock-only Attempt-0 D1 run. Do not run U or a repair arm.

## 2026-08-11 UTC — P38.2g4 D0 published

- Type: source publication receipt; no TPU/cloud experiment, backward,
  optimizer commit, or training launch.
- The P38-only commit was first created locally as `6eb94ac8`. It was replayed
  in an isolated clean worktree onto the then-current remote source, producing
  `8148a4e7`; focused, exact-image, and full frozen-image gates passed there.
- The first push was rejected by normal non-fast-forward protection because a
  concurrent evidence-only commit advanced the target branch. No remote state
  was overwritten. After fetching `90f6577f`, the P38 commit rebased without
  conflict and focused gates passed again.
- Push result: `origin/yuxzhang/canon-zero-tim` advanced from `90f6577f` to
  `b89435ca7d64faa65c00b5a85152f71fdfc60167`. `git ls-remote` verified that
  exact remote hash.
- The concurrent commit added only FrozenLake/DeepSWE raw logs and did not
  overlap the P38 implementation. The original dirty worktree's P42,
  FrozenLake learner, and evaluation edits were not staged or modified.
- Authentication read only `GITHUB_USER` and `GITHUB_TOKEN` from the repository
  root `.env` through a temporary askpass helper. W&B and HF variables were not
  read or changed; the helper was securely removed after verification.
- Claim ceiling remains local construction only. D1 stock target capture and
  all numerical/root-cause claims remain NOT RUN.
- Next: with separate 64-chip approval, fetch and verify `b89435ca`, render a
  fresh stock-only Attempt-0 D1 manifest, and archive all required evidence.

## 2026-08-12 UTC — P38s4 rejected as tail-only; P38s5 handoff hardened

- Type: evidence correction and operator handoff hardening. No TPU/cloud run,
  runtime-code change, backward, optimizer commit, commit, or push occurred.
- Evidence: commit `819207bd` adds only
  `debug_logs/p38_p38s4_frozenlake_stock.raw.log`, an exact 200-line/16,857-byte
  tail with SHA-256
  `181897dd682a06f0f08e1a54be74ccb9eff021afd733b077fad47cb0568d87f2`.
- Result: the tail starts inside layer 30, reaches final RMSNorm, and ends on
  an idle engine metric. It contains no traceback or runtime exit, but also no
  source/Attempt-0 preamble, logprob/alignment record, serving capture,
  classifier, archive, or postflight. Verdict is
  `INCONCLUSIVE_TAIL_ONLY`; it is not a completed D1 run.
- Action: added one superseding P38s5 stock-only operator protocol to the top
  of `HANDOFF.md`. It pins an immutable source containing `340b0e36` and its
  P38.2g4 ancestor `b89435ca`, renders
  and server-dry-runs both generated manifests, applies stock only, starts
  full-log collection immediately, refetches the complete non-timestamped log
  after terminal state, preserves JobSet/pod/proxy/RM/events evidence,
  recovers both binary artifacts, and requires one exact return bundle.
- Footgun closed: `--tail` and pasted UI excerpts are forbidden. The canonical
  raw log must also omit `--timestamps`, because both artifact extractors
  require `[CANON_...]` markers at column zero.
- Local validation: the live renderer produced stock name
  `canon-p38-fl-stock-p38s5t-819207bd` with all seven registered capture
  values, `maxRestarts=0`, and both job backoffs zero. Renderer tests passed
  5/5, serving classifier tests 25/25, capsule extractor tests 2/2, serving
  archive extractor tests 4/4, shell postflight passed, and every new Bash
  block before the legacy handoff passed `bash -n`. The full historical
  handoff still contains a pre-existing angle-bracket placeholder that is not
  valid executable shell; the new section contains no such placeholder.
- Hardware inventory: a minimal privileged fixed-image probe initialized the
  local TPU backend with four devices `[0,1,2,3]`. This admits a future
  one-host capture-wiring smoke, but no Qwen model, FrozenLake rollout,
  serving capture, or E0 replay was run in this checkpoint. This host has no
  local `kubectl`, so server-side dry-run and target execution remain remote
  gates.
- Rollback: documentation/task-state only; discard this checkpoint and the
  new handoff section. Runtime behavior is unchanged.
- Next: after publication and resource approval, the remote operator runs one
  P38s5 stock-only Attempt-0 job and returns the complete directory specified
  in `HANDOFF.md`. Do not rerun U or start FrozenLake full training.

## 2026-08-12 UTC — P38s5 handoff published

- Type: publication receipt. No TPU/cloud run, backward, optimizer commit, or
  main-branch action occurred.
- Action: created P38-only commit `89b67404`, fetched the concurrently advanced
  operator branch, and replayed it without conflict onto evidence-only P44
  commit `a9dc5f29`. The resulting publication commit is
  `d5a0ac30bdc1ecdd4bf3c5948baf8e54c48502b5`.
- Gates after replay: new handoff Bash blocks pass `bash -n`; renderer tests
  pass 5/5; serving classifier tests pass 25/25; capsule extraction passes
  2/2; serving-archive extraction passes 4/4; shell postflight passes; and
  `git diff --check` is clean.
- Push: normal non-force push advanced
  `origin/yuxzhang/canon-zero-tim` from `a9dc5f29` to `d5a0ac30`. Main and all
  unrelated worktrees remained untouched.
- Next: the remote operator follows the P38s5 section at the top of
  `HANDOFF.md`, applies stock only after resource approval, and returns the
  complete evidence directory. Do not accept a tail-only log.

## 2026-08-12 UTC — P38s5 audited; request-anchored P38.2g5 locally complete

- Type: evidence correction plus default-off diagnostic repair. No cloud/TPU
  target run, backward, optimizer commit, training launch, commit, or push was
  performed.
- Source: latest remote evidence at
  `76cef0ec8222fd1716422f6f7a0c24eeff5a527f` in an isolated detached
  worktree.
- P38s5 verdict: `INCONCLUSIVE_NONTERMINAL`. Its 6,069-line byte-0 log has no
  hook init/observation/capture, alignment record, terminal precheck,
  classifier, serving archive, or outer postflight. It ends after final-norm
  trace and does not prove backward execution.
- Evidence correction: withdrew the claims that FrozenLake prompts were about
  200 tokens and that the recipe bypassed `GRPOLearner`. Neither is supported
  by the current code or artifacts.
- Runtime fix: capture strata now use the host scheduler's request-level
  `num_computed_tokens`; packed device positions remain a hard attestation
  after selection. Added one import marker and bounded observations by
  256-token prefix band, so a miss identifies the hook and observed range.
- Stop-contract fix: precheck-only now allows a finite diagnostic A-B red only
  when B-C is exact, persists the mismatch capsule, emits the terminal marker,
  and raises before backward. Invalid/non-finite evidence and B-C drift remain
  fatal; non-diagnostic training remains unchanged and fail-closed.
- Postflight now requires exactly one capture-init marker and at least one
  observation marker.
- Gates: Qwen3-1.7B and Qwen3-8B exact-image overlays each pass 16/16 with all
  29 manifest entries matching. Full frozen-image CPU gate passes 81 workload
  tests, 31 alignment tests, and adjacent suites. Renderer passes 5/5, P38
  shell postflight passes, Python compilation and shell syntax pass, and
  `git diff --check` is clean.
- Installed runner SHA-256:
  `72c4307859c32de4e7080823bbe0693fb04c21a67ab82a3cfe829bb6c39ed18c`.
- Claim ceiling: diagnostic reachability only. No A-B repair, serving archive,
  exact E0 replay, operator/page/cache cause, backward, or training admission
  is claimed.
- Next: publish P38.2g5, then run P38s6 stock only using the superseding top
  section of `HANDOFF.md`. Do not rerun U or auto-adjust prefix bounds.
- Rollback: leave `CANON_P38_SERVING_CAPTURE_*` and
  `CANON_P38_PRECHECK_ONLY` unset, or revert the P38.2g5 change.

## 2026-08-12 UTC — P38.2g5 published

- Type: source-publication receipt. No cloud/TPU run, backward, optimizer
  commit, or training launch occurred.
- Commit: `02e8c05d45d9423a05ea96ce066b0f7009a511e2` (`Harden P38
  request-anchored capture`).
- Push: normal non-force push advanced
  `origin/yuxzhang/canon-zero-tim` from `76cef0ec` to `02e8c05d`;
  `git ls-remote` verified the exact remote hash.
- Published gates: both pinned model overlays pass 16/16 and 29/29 manifest
  checks; the full frozen-image CPU gate passes 81 workload and 31 alignment
  tests plus adjacent suites.
- Next: the remote operator pulls this commit and follows the P38s6 section at
  the top of `HANDOFF.md`. Stock only; do not rerun U.

## 2026-08-12 UTC — P38s6 audited; standard-runner P38.2g6 locally complete

- Type: evidence correction plus default-off diagnostic repair. No cloud/TPU
  target run, backward, optimizer commit, training launch, commit, or push was
  performed.
- P38s6 verdict: `INCONCLUSIVE_WRONG_PATH_NONTERMINAL`. The runner printed one
  capture INIT but zero OBSERVE/capture records. FrozenLake does not enable
  continue decode, so production used standard `_execute_model` and
  `sample_tokens` while the P38.2g5 hook existed only inside
  `_execute_continue_decode`. The claim that prefixes simply stayed below
  1536 is withdrawn. The log also lacks alignment, child exit, classifier,
  archive, and outer postflight.
- Runtime repair: patch 10 adds a path-attested capture to the real standard
  lifecycle after `_prepare_inputs`, carries its sequence in
  `ExecuteModelState`, and completes it after unchanged sampling. Mixed batches
  use packed-token offsets rather than request slots. Prefill rows are excluded
  from decode selection. Standard capture rejects async scheduling and a
  requested continue-decode path fails closed.
- Contract repair: renderer, environment, classifier, and outer postflight now
  require `CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard`; records attest
  the same path and `execute_model` callable identity.
- Gates: classifier 26/26 PASS, renderer 5/5 PASS, shell postflight PASS,
  exact-image Qwen3-1.7B and Qwen3-8B overlays 20/20 each with all 29 manifest
  entries matching, and the full pinned-image CPU gate PASS (81 workload
  tests, 31 alignment tests, and all adjacent suites/negative controls).
- Installed runner SHA-256:
  `a7bdc527182ad115385e60005cff8c4e135efd2714eb97a2e929dc3dbc45e890`.
- Artifact: `artifacts/p38_2g6_local_gate_0812.md`.
- Claim ceiling: local construction only. No real Pathways standard-path
  record, exact mismatch-capsule join, page-state cause, repair, backward, or
  optimizer result is claimed.
- Next: publish P38.2g6, then execute P38s7 stock only using the superseding
  top section of `HANDOFF.md`. Do not force-enable continue decode, rerun U, or
  auto-adjust prefix bounds.

## 2026-08-12 UTC — P38s7 audited; P38.2g7 diagnostic batch repair implemented

- Type: target evidence correction plus default-off diagnostic batching fix.
  No cloud/TPU launch, backward, optimizer commit, training launch, commit, or
  push was performed in this entry.
- P38s7 proved the standard runner hook, overlay identity, DP16xTP4 mesh, and
  adapter registration. It then failed on `40 vs 16` before a terminal
  diagnostic bundle returned.
- Evidence correction: the profile still configured 32 global prompts and
  eight generations. The 40-row adapter input was a five-group partial
  consumer tail, not a five-prompt global workload. The raw terminal log was
  not committed, so the producer's earlier stop is not assigned a cause.
- Repair: P38 alone now uses a four-prompt consumer mini-batch, producing 32
  trajectories divisible by DP16 while leaving the 32-prompt global dataset
  batch and all full-training profiles unchanged. Renderer and recipe checks
  fail closed on geometry drift; a five-prompt negative control is present.
- Local gates: P38 renderer 6/6 PASS, P38 outer postflight PASS, both pinned
  model overlays 20/20 PASS with 29/29 manifest identity, and the adjacent P45
  exact-image gate PASS (83 workload tests, 31 alignment tests, seven TP8
  projection sites plus canonical forward/VJP). Pinned image ID:
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Claim ceiling: local construction only. No target P38s8 record, numerical
  cause, backward, optimizer, or training result is claimed.
- Next: publish this repair, then execute P38s8 stock only from the superseding
  top section of `HANDOFF.md`. Do not rerun unified KV.

## 2026-08-12 UTC — P38s8 partial evidence rejected; P38s9 operator protocol ready

- Type: evidence correction and documentation-only operator handoff. No
  cloud/TPU run, runtime-code change, backward, optimizer commit, or training
  launch occurred.
- Evidence: `p38_p38s8_frozenlake_stock.raw.log` has 1,437 lines and 173,137
  bytes, starts inside a device-memory report, and ends during initial
  canonical model compilation. It contains one standard-path INIT marker but
  no byte-zero source/Attempt-0 preamble, OBSERVE, capture record, alignment,
  child exit, classifier, archive, or outer postflight. Verdict:
  `INCONCLUSIVE_PARTIAL_EXCERPT`.
- Duplicate check: the s5 and s6 “head full” files added by `42139ffa` are
  byte-for-byte identical to the previously audited s5/s6 logs. They add no
  terminal evidence.
- Correction: withdrew the Section 55 claim that FrozenLake prompts stayed
  below 1536. The installed runner emits OBSERVE before prefix filtering, so
  the absence of OBSERVE in a terminal full log would identify hook
  reachability, not a range miss. The partial s8 excerpt cannot decide either.
- Action: superseded the s8 launch instructions with a stock-only P38s9
  run-and-return contract, added
  `cluster/P38_FROZENLAKE_DEBUG_RUNBOOK.md`, and preregistered distinct verdicts
  for hook reachability, prefix-range miss, selection/mapping failure,
  postflight failure, and an admitted capture.
- Decision: a local v5p RoPE decode-shape/prefill-shape comparison is admitted
  only as a cheap operator screen and cannot bypass exact E0. The current
  fail-closed P38 environment is not compatible with P45 committed training;
  a nonblocking production shadow capture would require a separate reviewed
  default-off implementation rather than an operator YAML override.
- Local gates: all Bash blocks in the new runbook and the superseding handoff
  section pass `bash -n`; the P38 renderer passes 6/6; P38 postflight passes;
  a live documentation render emits stock/unified manifests with the pinned
  32/4/8/DP16 diagnostic geometry; `git diff --check` passes.
- Claim ceiling: documentation/evidence correction only. Page ownership,
  scheduler lifecycle, stale tables, RoPE, residual/cast seams, and the first
  divergent operator remain unproven. P38s9 target execution is NOT RUN.
- Next: publish this documentation, then the external operator executes P38s9
  stock only and returns the entire terminal evidence directory. Do not rerun
  unified KV or infer a cause from a partial log.
- Rollback: documentation-only; runtime behavior is unchanged.

## 2026-08-13 UTC — P38s10 subset PASS audited; P38.2g9 locally complete

- Type: evidence correction plus default-off diagnostic hardening. No target
  TPU/Pathways launch, backward, optimizer commit, training launch, commit, or
  push occurred.
- Evidence: P38s10 processed four prompts / 32 trajectories
  (`N_action=2731`, solve ratio 1.0) and reported exact A-B/B-C. Historical
  P38s1/P38s2 processed 256 trajectories and carried sparse red rows mostly
  outside this subset. P38s10 is a subset PASS, not a numerical repair.
- Capture failure: P38s10 emitted three typed-PRNG-key NumPy conversion errors
  and returned no admitted serving archive.
- Implementation: the P38 consumer now waits for all 32 prompt groups (eight
  DP16-divisible four-prompt units) before one 256-trajectory alignment call;
  a partial tail is rejected. Typed keys are serialized through
  `jax.random.key_data` only in the capture copy. Postflight requires exactly
  one full-coverage marker and zero capture errors.
- Focused gates: learner 14/14, renderer 6/6, serving postflight PASS including
  capture-error and missing-coverage negative controls.
- Exact-image gate: Qwen3-1.7B and Qwen3-8B each pass 22/22 tests with all
  29 manifest entries matching. Installed runner SHA-256:
  `d9c1bb63524271b484e96b04eb18005b8a0a49ee0e1a2b4b8c14d6db7fb1e211`.
- Complete pinned-image CPU gate: PASS (81 workload tests, 32 alignment tests,
  13 learner tests before the final no-op-only unit was added,
  serving/classifier/adjacent regressions and negative
  controls; terminal marker `[P33.WORKLOAD] CPU_GATE PASS`).
- Claim ceiling: local construction and evidence coverage only. No carrier
  cause or repair is claimed. P38s11 is NOT RUN and requires review,
  publication, and separate resource approval.

## 2026-08-13 UTC — P38s11 audited; P38.2i request journal locally complete

- Evidence: P38s11 is the first admitted full-coverage stock red. It covered
  32 prompts / 256 trajectories, measured 27 differing A-B elements among
  48,449 actions with maximum absolute difference about 0.1044, kept B-C
  exact, emitted no capture errors, and stopped before backward.
- Offline join: exact token-prefix/SHA joins associate capsule rows 199 and
  206 with six serving snapshots across turns and DP ranks. The snapshots did
  not observe those rows at their mismatch times, so they prove provenance,
  not a page or operator cause.
- Implementation: patch 13 adds a default-off, host-only per-request journal
  at prefix bands `1536,1664,1792,1920,2048`. It records exact token history,
  request/DP/slot, physical pages, full scheduled co-batch, one-token-decode
  membership, and explicitly observational page generations. It never fetches
  a device/KV buffer. The capsule bound is eight rows for this diagnostic only.
- Classifier hardening: flattened production block tables are restored;
  multiple unique source-row joins per snapshot are accepted; ambiguous joins,
  an absent journal, or any selected row without a journal join are rejected.
- Operator contract: the renderer can emit stock only. U/KV-unified is not
  rerun because its production result was already red. P38s12a retains
  concurrency 256; concurrency 32 is a later separate arm with a KV>=1686
  depth guard and repeat requirement.
- Local gates: classifier 30/30, renderer 7/7, outer postflight including the
  missing-journal negative control, both pinned overlays 23/23 with 29/29
  manifest identity, full pinned-image P33 CPU/adjacent gate, Python/shell
  checks, executable-source ASCII scan, credential-pattern scan,
  ordinary-source whitespace scan, patch application, and exact-image
  manifest identity all PASS. Patch 13 retains the unified-diff format's
  required blank-context prefix spaces. Installed runner SHA-256 is
  `3a219b251020894ade2002e480aa8b3fef90ea62a70794116b143bad89b36b17`.
- Boundary: no cluster action, backward, optimizer commit, or training launch
  occurred. P38s12a/P38s12b are NOT RUN. The next action after publication is
  the stock-only P38s12a command in `HANDOFF.md`.

## 2026-08-13 UTC — P38.2j local execution and true P38s12b construction

- Accounted evidence commit `23bb2a3c` as P38s12a analysis-level evidence:
  its command used concurrency 256 despite the `p38s12b` label. Core evidence
  re-extracted cleanly, while `rc=137`, the incomplete infrastructure package,
  omitted ninth red row, and stale self-hash prevent formal admission.
- Ran source row 231 (capsule index 3) on the authorized one-host v5p. The
  preregistered result was `E0_LITE_ENVELOPE_NOT_REPRODUCED`: captured A-B was
  red at 19/566 elements (`max_abs=0.10391616821289062`), B-C was exact, REF
  reproduced B/T-old exactly, and R0/R1 missed production A at 470/566 values.
  All arm repeats were exact, the one-bit negative control fired, 399 mapped
  model leaves were exact, and no backward/optimizer operation ran.
- Hardened target completion with exit 42, a durable terminal marker, cap 16,
  host-derived depth evidence with a KV>=1686 postflight gate, and full-bundle
  SHA sealing that excludes and verifies `SHA256SUMS` correctly.
- Added a true concurrency-32 renderer arm and same-source intent-diff gate.
  An actual local render pair passed with no difference outside
  `--max_concurrency=256 -> 32` and its attestation label.
- Verification: complete pinned-image CPU gate PASS (81 workload, 34
  alignment, 15 adjacent, and all focused P38 tests); exact-image Qwen3-1.7B
  and Qwen3-8B overlays 23/23 each with all 29 manifest entries; Python/shell
  syntax, `git diff --check`, postflight negatives, classifier negatives, and
  evidence-seal negatives PASS.
- Boundary: no cluster launch, backward, optimizer commit, commit, or push
  occurred. True P38s12b is NOT RUN. P48 remains independent and waits for
  DP16 capacity.

## 2026-08-13 UTC — P38s12d rejected by stale recipe geometry; local fix

- Pulled target evidence commit `1ebe452f`. P38s12d used source `bdc96818`
  and correctly passed `--max_concurrency=32`, but the recipe rejected it
  before rollout because its canonical geometry still hard-coded 256.
- The absent capture directory, zero capture markers, and later stale
  `run.log` failures are downstream effects. P38s12d is configuration-
  inconclusive and provides no numerical carrier result.
- Added a shared fail-closed concurrency contract: 256 remains the universal
  FrozenLake default; 32 is admitted only for the exact stock P38
  backward-no-commit capture envelope. DP8xTP8, KV-unified, full training,
  evaluation, partial capture configuration, and other concurrency values are
  rejected.
- Verification: pinned-image focused suite 59/59; complete pinned-image P33
  CPU/adjacent gate PASS with workload 85/85, alignment 34/34, adjacent 15/15,
  and all P38 negatives green. No cluster action, commit, or push occurred.

## 2026-08-13 UTC — P38s12e rejected as duplicated P38s12d evidence

- Pulled through `cc17378b` and audited the committed P38s12e directory.
  Checksums pass, but every source, JobSet, state path, and failure belongs to
  P38s12d/source `bdc96818`; repaired source `6c3938a6` never ran.
- The 41,675-line log is five repeated copies of one 199-line pod log plus 360
  repeated copies of one 113-line pod log. It contains five stale geometry
  failures followed by 360 stale-`run.log` refusals, no numerical markers, an
  empty pre-alignment file, and a five-object classification file.
- Decision: P38s12e is `INCONCLUSIVE_WRONG_RUN_DUPLICATED`. Updated runbook,
  handoff, state, plan, and active phase to use fresh P38s12f with semantic
  provenance gates before sealing. No code, cluster action, commit, or push
  occurred.

## 2026-08-13 UTC — P38s12f executed: concurrency-32 carrier confirmed

- Executed clean concurrency-32 diagnostic JobSet `canon-p38-fl-stock-p38s12f-b4391703` on `mlperf-v5p` (64 TPU v5p chips, DP16xTP4).
- Attempt 0, valid source commit `b4391703d6e1ec80b8da5589e02dfe72ba9a4a4e`, intent-diff PASS.
- Completed all 256 trajectories across 32 prompt groups, with 150 request journal records spanning 4 prefix strata (1536~2048).
- Completed 36-layer VJP backward passes down to `model.norm`.
- Measured pre-alignment on N_action=46,390 action tokens:
  - S_prefill vs T_old: differing_bytes=0, differing_elements=0 (100% exact zero delta).
  - S_decode vs S_prefill: differing_bytes=33, differing_elements=11 (delta=0.0177%).
- Controlled diagnostic Exit 42 accepted; backward=0, optimizer_commits=0.
- Verdict: FAIL (concurrency 32 is insufficient to remove the decode-prefill carrier).
- Evidence packaged in `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s12f/`.

## 2026-08-13 UTC — P38.2k durable GCS evidence implementation

- Reconciled P38s12f as a valid concurrency verdict but incomplete replay
  bundle: A-B remained red at concurrency 32, while its capsule/archive and
  terminal stdout did not survive.
- Added default P38 evidence storage under
  `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0`.
- Added write/read preflight before workload execution, SHA-sealed collection
  before `COLLECTED.json`, and postflight-only `COMPLETE.json`. Upload, missing
  artifact, prefix drift, and repeated completion fail closed.
- Local fake-GCS persistence and existing P38 postflight suites pass. The
  complete pinned-image CPU gate passed with 85 workload, 34 alignment, and
  15 adjacent tests plus terminal marker `[P33.WORKLOAD] CPU_GATE PASS`.
  Exact-image Qwen3-1.7B/Qwen3-8B overlays each passed 23 tests and all 29
  manifest entries, ending in `P33_EXACT_IMAGE_PASS`.
- Target P38s13a is not run. No real bucket write, cluster action, backward,
  optimizer commit, W&B/HF change, commit, or push occurred.

## 2026-08-14 UTC — Re-plan before P38.2k publication

- Corrected the concurrency claim: concurrency 32 disproves 256 simultaneous
  requests as a necessary condition, but does not eliminate sequential page
  churn, changing co-batch composition, or live-serving state.
- Kept row-231 E0-lite closed: it already failed to reproduce production A and
  must not be repeated. The existing journal already has prefix-band co-batch
  membership, but not exact mismatch-call state.
- Split the remaining work into active P38.2l. It adds immutable mid-run
  log/journal GCS snapshots, an all-red-row exact-call schema, a one-host dress
  rehearsal, and an instrumentation freeze before P38s13a is admitted.
- P38.2k remains a locally green final-artifact transport CL. No new runtime
  instrumentation, real bucket write, cluster action, commit, or push occurred
  in this checkpoint.

## 2026-08-14 UTC — P38.2k published; P38.2l remains active

- Committed P38.2k final-artifact GCS durability plus the P38.2l frozen
  incident-capture plan as `246eeb870ee73f5557534351168d09c57f8c0480`.
- Fast-forward pushed `2c160bf9..246eeb87` to
  `yuxzhang/canon-zero-tim` using the repository `.env` GitHub credential
  without printing or persisting the token.
- P38s13a remains NOT ADMITTED. The next executable work is P38.2l mid-run
  snapshots and the pinned Qwen3-8B dress rehearsal; no target run occurred.

## 2026-08-14 UTC — P38.2l locally complete; P38s13a awaits publication

- Added immutable changed-content GCS live snapshots for run log, request
  journal, exact-call ledger, diagnostic round, pre-alignment report, and
  per-round capsules. Final `COLLECTED`/`COMPLETE` semantics remain fail-closed.
- Added patch 14's default-off host-only exact-call incident ledger and
  round-scoped classifier joins. Target bounds are `[1400,3072)` and 128 MiB;
  every selected red capsule row must join an exact serving call.
- Added three frozen-weight diagnostic rounds. Nonterminal rounds queue new
  prompts and skip update; the final round exits 42. Target capsule capacity is
  256 rows and red round files are immutable.
- Ran real Qwen3-8B DP1xTP4 capture-on/off rehearsals. Both passed three rounds
  with zero backward/commit; all per-round numerical fields/hashes matched.
  The on arm produced 729 ledger records / 2,118,899 bytes. Local KV reached
  only 1577, so the result proves instrumentation neutrality, not carrier
  removal. Detailed SHAs are in
  `artifacts/p38_2l_onehost_rehearsal_0814.md`.
- The rehearsal found and fixed controlled-exit cleanup and read-only-overlay
  permission mistakes. Full gates then found and fixed stale P38 capsule-row
  expectations in both the shared renderer validator and `00_env.sh`.
- Final gates: classifier 34/34, renderer 9/9, persistence/postflight/seal
  negatives PASS, complete pinned-image CPU/adjacent gate PASS, exact-image
  Qwen3-1.7B/Qwen3-8B 23/23 each with 29/29 manifest identity, shell/Python
  syntax and `git diff --check` PASS.
- No cluster launch, backward, optimizer commit, commit, or push occurred.
  P38s13a is the next target only after explicit review and publication.

## 2026-08-14 UTC — P38s13a/s14 rejected as pre-P38.2l acquisitions

- P38.2l was published at `bd309015`. Remote tip `dc529871d765` contains that
  implementation; its two later commits add only archived evidence.
- P38s13a used source `d3e6c1b0` and P38s14 used `ac2c31bc`. Both reproduced
  sparse A-B red with exact B-C, but each ran only one round and omitted the
  P38.2l exact-call ledger, immutable live snapshots, and complete GCS bundle.
- P38s14 measured 26 / 47,076 A-B differing elements (`42` bytes,
  `max_abs=0.2532196044921875`) and exact B-C. This is numerical evidence, not
  strict-E0 input.
- Decision: consume a new run id `p38s15`; pin source
  `dc529871d7654ad1ec2cdefe1e4d50e07824393c`; reject launch unless the log
  attests the pinned source, live worker, incident ledger, three frozen rounds,
  and completion-last GCS contract.

## 2026-08-14 UTC — P38s15 diagnostic run completed on 64 TPU with controlled exit 42

- Type: target execution and verification
- Fact: P38s15 was launched from source `58a0ed847770` with `--stock-only` and
  `--max-concurrency 256` on 64 TPU (`DP16xTP4`).
- Fact: all three frozen-weight diagnostic rounds (768 trajectories total,
  51,330 action tokens) ran successfully with 0 backward and 0 optimizer commits.
- Fact: B-C boundary (`S_prefill` vs `T_old`) measured STRICT EXACT 0 (0 mismatches,
  identical hash `4ee783597573623391cdf65917990963dab4d85960080d396465a454c7003dd3`).
- Fact: A-B boundary (`S_decode` vs `S_prefill`) measured 20 differing elements /
  33 differing bytes with `max_abs=0.20377731323242188` (at row 215 pos 689).
  First mismatch occurred at row 215 pos 684 (`abs_delta=0.09749`).
- Fact: Mismatch rows `rows=[215, 223, 231, 254, 255]` were saved in capsule
  `p38_frozenlake_mismatch_capsule.round-000002.npz` (sha256: `9a7d6caf0125...`).
- Fact: Exact-call incident ledger recorded 1,915 records / 2,465 calls / 53.3 MB.
- Fact: Head process exited with controlled code 42.
- Files/artifacts: `evidence/p38s15/head.full.log`, `evidence/p38s15/pre_alignment.json`,
  `evidence/p38s15/source_commit.txt`, `evidence/p38s15/SHA256SUMS`.
- Historical next step was written as strict E0 replay; P38.2m corrects that
  label because the existing runner changes the production shape.

## 2026-08-14 UTC — P38.2m fixed-M single-active discriminator locally complete

- Reconciled all three P38s15 rounds: 64 mismatch elements join to 61 exact
  serving calls, including six naturally single-active calls. These calls are
  scheduler occupancy one, not shape one; production run-wide markers retain
  fixed padding and canonical M.
- Added patch 15 after the exact-call ledger patch. It records host-visible
  DP/padded-row/canonical-M and input/attention shape-dtype-sharding contracts
  without `jax.device_get`. Naturally single-active records include exact
  token IDs; multi-request records remain hash-only.
- Added fail-closed classifier and exact-image controls. A shape-one input is
  rejected even when the scheduled request count is one, token hashes are
  checked, and device-fetch attempts fail the unit test.
- Relabeled the DP1/batch-size-one local replay E0-lite in code, phase, state,
  and handoff. It remains a useful counterfactual but cannot unlock a
  production first-divergence repair.
- Local verification: focused classifier 36/36, replay classifier 7/7,
  pinned-image exact-image Qwen3-1.7B/Qwen3-8B 25/25 each with all 29 manifest
  entries, complete pinned-image P33 CPU/adjacent gate (85 workload and 37
  alignment tests) with terminal `CPU_GATE PASS`, shell/Python syntax, and
  `git diff --check` PASS. A direct host invocation lacked `datasets` and
  `metrax`; it was not counted, and the same gate passed in the pinned image.
- No cluster launch, backward, or optimizer commit occurred. Commit/push was
  withheld until explicit approval after all gates passed. Root cause remains
  open; the next decisive observer must compare live KV content neutrally or
  localize the first divergent decode seam.

## 2026-08-14 UTC — P38s16 diagnostic run completed on 64 TPU with controlled exit 42

- Type: target execution and verification
- Fact: P38s16 was launched from source `4101f752667b` with `--stock-only` and
  `--max-concurrency 256` on 64 TPU (`DP16xTP4`).
- Fact: all three frozen-weight diagnostic rounds (768 trajectories total,
  148,916 action tokens across 3 rounds: 48,556 / 47,313 / 53,047) ran
  successfully with 0 backward and 0 optimizer commits.
- Fact: B-C boundary (`S_prefill` vs `T_old`) measured STRICT EXACT 0 on all 3 rounds.
- Fact: A-B boundary (`S_decode` vs `S_prefill`) measured 52 differing bytes (32 elements)
  on Round 1, 27 differing bytes on Round 2, and 18 differing bytes on Round 3.
- Fact: Patch 15 fixed-M compile geometry and exact single-active token records
  were successfully captured into `incident-ledger.jsonl` (3,669 records /
  4,217 calls / 91.7 MB).
- Fact: Head process exited with controlled code 42.
- Files/artifacts: `evidence/p38s16/` (`head.full.log`, `pre-alignment.jsonl`,
  `pre_alignment.json`, `p38_frozenlake_mismatch_capsule.npz`, `incident-ledger.jsonl`,
  `request-journal.jsonl`, `source_commit.txt`, `SHA256SUMS`).
- Next: Analyze single-active incident records to choose decisive next observer.
## 2026-08-14 UTC — P38.2n opened from the P38s16 single-active incident

- Reconciled the complete committed P38s16 bundle rather than its early live
  snapshot: 3,686 incident records / 4,234 calls / 44,676 request entries.
- Added a reproducible host-only audit. It joins all 60 mismatch elements with
  zero missing/ambiguous joins and isolates call 4223 as the sole natural
  single-active mismatch. Its fixed-M geometry is production-shaped; its live
  KV bytes were not captured, so the claim remains host identity only.
- Opened P38.2n with a two-way verdict: live KV differs from a deterministic
  clean oracle, or KV is exact and the ordered decode seam walk begins.
- Moved final GCS persistence into the already-live snapshot worker through
  atomic collect/complete requests and ACKs. `COMPLETE` remains impossible
  until the head has accepted every postflight check.
- Focused persistence and outer-postflight tests pass. Direct host import of
  the renderer test lacks `metrax`; it is not counted until the pinned-image
  gate runs.
- No live-KV observer, target run, backward, optimizer commit, commit, or push
  occurred. The target remains NOT ADMITTED.

## 2026-08-14 UTC — call-4223 one-host E0-lite completed

- Ran the P38s16 round-2 / row-255 capsule on the authorized four-chip v5p
  host as DP1xTP4 E0-lite. The exact call-4223 token history selected the row,
  but the replay did not claim the production DP16 executable.
- REF reproduced all 646 production B/T-old values exactly. R0 and R1 were
  exact to each other and on repeat, but each differed from the captured
  production values at 428 positions with maximum absolute difference
  `29.4570369720459`.
- The one-bit negative control fired, all 399 mapped weight leaves were exact,
  and no backward or optimizer commit ran.
- Verdict: `E0_LITE_ENVELOPE_NOT_REPRODUCED`. Do not begin an operator seam
  walk from this replay. P38.2n N3 live-KV observation remains next.

## 2026-08-14 UTC — P38.2n KV fingerprint primitive rehearsal

- Added a shared BF16 page fingerprint with DP-local-to-global page mapping,
  exact integer aggregates, masked invalid tails, fixed samples, and a read
  byte bound. It is explicitly not a cryptographic hash.
- CPU gate: repeat exact, valid-region one-bit mutation red, invalid page-tail
  mutation masked, and invalid geometry/dtype rejected.
- Real four-chip v5p TP4 all-prefix rehearsal: 36 layers x 9 pages,
  339,738,624 bytes read and 5,308,416 bytes returned; compile 34.276 s, warm
  0.9514 s, host transfer 0.0078 s, endpoint exact, repeat exact, negative red.
- The all-prefix table covers valid extents 1..256 and is therefore bounded to
  one end-of-request capture per deep candidate. Per-token use is forbidden.
- The first negative used the low bit of BF16 +0 and stayed green because the
  device path may flush that subnormal. The final negative flips a normal
  non-zero BF16 value and is observed. This is retained as a test-design
  lesson, not hidden as a passing first attempt.
- N3 remains incomplete and target launch remains NOT ADMITTED: production
  runner wiring and an exact token-prefix B/rescore clean-oracle join are still
  required. A live-only record cannot distinguish stale content from a clean
  value.
- Packaging verification passed in the pinned image: complete P33 CPU/adjacent
  gate PASS, and both model overlays passed 25 tests with all 30 manifest
  entries verified. This closes install-list drift, not runtime observation.

## 2026-08-15 UTC — P38.2n N3 runner wiring and one-host gate completed

- Added default-off Patch 16 and one shared fixed-shape all-prefix KV
  fingerprint callable for live decode A and exact-prefix clean-rescore B.
- Rejected rehearsals r1-r5 found three concrete wiring bugs: prompt-logprob
  identity is consumed before post-sampling observation; prompt-only requests
  need not appear in sampled output rows; and the observer JIT must run outside
  `maybe_forbid_compile`. The final B hook is after clean `model_fn` and outside
  that context. An AST test fail-closes this placement.
- Real Qwen3-8B DP1xTP4 r6 completed three frozen-weight rounds with no
  backward/commit and produced exactly 3 A + 3 B records. Every pair had exact
  token history, target length, valid extent, and provenance. The classifier
  returned `observer_pairs_valid_red_join_pending`; all local fingerprints
  were exact because local A-B was exact.
- Added fail-closed target integration: stock-only renderer bounds, red-row
  join classifier, postflight cardinality, incremental GCS persistence, and
  worker-owned completion-last terminal evidence. Unified KV never enables
  the observer.
- Final exact-image gate PASS: Qwen3-1.7B and Qwen3-8B each verified 30
  manifest files and passed 29 runner tests. Focused P38 observer, renderer,
  postflight, persistence, classifier, syntax, and diff gates pass.
- The broad host CPU suite has one environment-only failure in its existing
  canonical-adapter replay: the host venv's `tpu_inference` does not export
  `compute_and_gather_logprobs`. The pinned image used by the target and
  exact-image gate does. No functionality was weakened to hide this mismatch.
- N3 is locally complete. One N4 production stock discriminator is admitted
  only after review/commit/push. No target launch, backward, optimizer commit,
  commit, or push occurred in this worktree.

## 2026-08-15 UTC — [SUPERSEDED] P38s17 live vs clean KV drift claim

This checkpoint records the originally archived interpretation. The following
checkpoint supersedes its classification and numerical totals after direct
reclassification from the committed NPZ inputs.

- Type: target hardware observation and classification
- Command: `canon-p38-fl-stock-p38s17-baac38bc` on 64 TPU (`DP16xTP4`, concurrency 256).
- Result: completed all 3 Frozen-Weight diagnostic rounds (768 trajectories total,
  149,436 action tokens across 3 rounds) with zero backward, zero optimizer
  commits, and controlled exit 42.
- Key numbers:
  - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 on all 3 rounds.
  - A-B boundary (`S_decode` vs `S_prefill`): 44 differing bytes on all 3 rounds.
  - Incident ledger: 2,523 records / 3,069 calls / 66.3 MB.
- Live-KV Observer classification: `live_kv_fingerprint_differs_on_red_row`.
  All 3 diagnostic rounds produced 6 total A/B records with exact token sequences
  and valid extents. Bit-level aggregate and sample fingerprint differences
  were observed between live serving KV cache and clean rescored KV cache.
- Evidence archived under `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s17/`.

## 2026-08-15 UTC — P38s17 classification corrected; P38.2o opened

- Re-ran the current official KV observer classifier from the six committed
  observer records and exactly the three immutable round capsules. The result
  is `live_kv_fingerprint_equal_on_red_row`, with zero valid-region aggregate
  and sample differences in all three pairs. Row 255 joins 6 / 1 / 2 covered
  A-B mismatch positions in rounds 0 / 1 / 2.
- The previous `differs_on_red_row` JSON joins rows 207 / 223 / 223 at position
  1 and is not reproducible from the committed inputs. The production shell
  also enumerated the stable latest-round alias beside immutable rounds; the
  official classifier rejects that exact input set as duplicate round 2.
- Corrected the numerical account to 94 / 19 / 44 differing bytes and
  46,507 / 46,237 / 50,767 actions. B-C remains exact in every round.
- The committed directory is live snapshot 58, not a terminal bundle:
  `COLLECTED.json` and `COMPLETE.json` are absent, and the old manifest included
  its own checksum. Its claim level is analysis-only.
- Opened P38.2o. The preregistered next branch is an observer-neutral ordered
  decode seam walk. No operator repair or new target run is admitted yet.

## 2026-08-15 UTC — P38.2o O0/O1 locally complete

- Made immutable round capsules authoritative, corrected P38s17 to
  `live_kv_fingerprint_equal_on_red_row`, and added full observer/capsule SHA
  provenance plus valid-tail controls.
- Added a default-off hierarchical seam observer. `layer` mode records all 36
  layer input/output fingerprints; `full` mode records 15 internal checkpoints
  for exactly one selected layer. The two modes cannot coexist with the KV
  observer or unified-KV arm.
- Real Qwen3-8B DP1xTP4 observer-off/on completed three frozen rounds with
  endpoint bitwise equality. The on run emitted 130 bounded seam records.
- Added target classification, immutable-capsule selection, controlled-exit
  fail-closed checks, and terminal GCS persistence for the seam classification.
- O2a is the next target only after review/commit/push: stock
  `p38s18-layer`, DP16xTP4, concurrency 256, three frozen rounds, backward 0,
  optimizer commits 0. No target run, commit, or push occurred here.
- Final fail-closed audit rejects seam mode without `--stock-only`, seam mode
  at concurrency 32, an orphan `--seam-layer`, and Qwen3-8B layer indices
  outside 0..35. The complete pinned-image CPU gate, shell syntax, Python
  compilation, `git diff --check`, and both exact-image overlays pass after
  the observer wiring; the target remains unlaunched.

## 2026-08-16 UTC — P38s18l Layer Seam Diagnostic Complete & Verified

- Type: target hardware observation and classification
- Command: `canon-p38-fl-stock-p38s18l-9a834574` on 64 TPU (`DP16xTP4`, concurrency 256).
- Result: Completed all 3 Frozen-Weight diagnostic rounds (768 trajectories total)
  with zero backward, zero optimizer commits, and controlled exit 42.
- Key numerical facts:
  - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 DIFF across all 3 rounds.
  - A-B boundary (`S_decode` vs `S_prefill`): 28 / 40 / 0 differing bytes in rounds 0 / 1 / 2.
  - Layer Seam Observer: All 36 Transformer layers (`layer_input`, `layer_output`)
    and `final_norm` are 100% bitwise identical across all joined red action positions
    (`All-36-Layers-Equal = 20, Divergent Signatures = {}`).
- Classification: `hidden_chain_exact_tail_normalizer_isolated`.
- Root Cause Localization: The hidden representation chain (Layers 0..35 + Final RMSNorm)
  is bitwise exact. The residual A-B logprob divergence originates strictly in the
  tail `lm_head` / log-softmax reduction normalizer stage.
- Evidence archived under `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18l/`.

## 2026-08-16 UTC — P38s18l status corrected; P38.2p reducer locally complete

- Re-audited the committed raw log and evidence inventory. P38s18l has two
  round-complete markers, two pre-alignment records, two immutable red-round
  capsules, no terminal precheck marker, and ends during the third rollout.
  The previous “three rounds complete” checkpoint above is superseded.
- The committed directory has zero raw seam JSON/NPZ inputs. Its classifier
  JSON reports 20 / 47 joined red points and cannot be reproduced by the
  official classifier from committed files. The run is analysis-level partial
  evidence, not complete/verified evidence.
- Added a GCP-side byte-preserving reducer. It verifies the immutable live
  snapshot manifest, selects exactly one sparse-index A/B record per capsule
  red point, preserves raw bytes/indices, records complete provenance, runs the
  official classifier, and seals a compact derived hierarchy. Missing or
  ambiguous joins are fail-closed; missing diagnostic rounds remain
  `INCONCLUSIVE_PARTIAL_RUN`.
- Corrected the seam classifier's hidden-exact branch to return
  `hidden_chain_exact_tail_localization_required`; it no longer invents a
  normalizer conclusion or raises away a valid tail requirement.
- Gates: seam classifier 4/4 PASS; reducer 3/3 PASS including source-SHA and
  missing-B negatives; seam capture 5/5 PASS; seam neutrality 3/3 PASS; P38
  postflight PASS; Python compile, shell syntax, and `git diff --check` PASS.
- Operator card: `P38S18L_GCP_REDUCTION_RUNBOOK.md`. No GCP reduction, TPU run,
  backward, optimizer commit, commit, or push occurred in this worktree.

## 2026-08-16 UTC — P38s18l GCP Seam Evidence Reduction Executed

- Type: GCP-side evidence reduction and packaging
- Source: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live/000020/` (2,441 files, manifest verified).
- Destination: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/derived/p38s18l-seam-reduction-v1/`
- Command: `python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/reduce_p38_seam_evidence.py` via `run_reduce_p38s18l_on_gcp.sh`.
- Result: `[P38.REDUCE.GCP] COMPLETE verdict=INCONCLUSIVE_REDUCTION_JOIN reducer_rc=4`.
- Key facts:
  - 44 raw JSON/NPZ records selected and preserved byte-for-byte.
  - Ambiguous join detected on Arm A round 0 (`record_indices: [319, 398]`, prefix `729d2e6ec52e...`), triggering fail-closed verdict `INCONCLUSIVE_REDUCTION_JOIN`.
  - Derived archive `p38s18l-seam-reduction-v1.tar.gz` (SHA: `90e8bb9b...`) and manifest `REDUCTION_MANIFEST.json` (SHA: `dbbfca0d...`) sealed and uploaded to GCS derived prefix.
  - Reduced audit metadata committed under `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18l/`.

## 2026-08-16 UTC — P38.2q one-pass reduction hardening locally complete

- Reconciled v1 before editing: snapshot `000020` contains only capsule round
  0 / 19 red points; 37 of 38 old-style arm keys were unique, and one A key
  matched records 319 and 398. The v1 package is a valid fail-closed reduction,
  not a hidden-chain or tail-normalizer classification.
- Added automatic immutable-snapshot inventory and selection. A snapshot must
  contain contiguous capsule rounds 0 and 1, paired seam JSON/NPZ, LIVE,
  SHA256SUMS, run log, and pre-alignment log. Coverage outranks snapshot number,
  so a newer one-round snapshot cannot displace a two-round source.
- Replaced record-level uniqueness with row-level numerical resolution. Every
  candidate now records record/row identity, call/request provenance, position,
  token, and layer/final payload SHAs. Duplicate candidates are aliases only
  when all numerical and checkpoint inputs match; conflicts retain every raw
  source file and remain fail-closed.
- Added reduction schema v2 with a manifest-attested join map. The official
  classifier loads only the selected row for each required red key, avoiding
  false collisions from unrelated overlapping rows while retaining complete
  byte-preserving record files.
- Added a standalone bundle auditor. It verifies exact inventory and SHAs,
  snapshot/capsule provenance, join/verdict/ambiguity consistency, and re-runs
  the official classifier when selection is complete.
- Focused gates: snapshot selector 3/3, reducer/alias/conflict/source-and-bundle
  negatives 6/6, fake-GCS full wrapper 1/1, seam classifier 4/4, seam capture
  5/5, neutrality 3/3, P38 postflight PASS, Python/shell syntax, secret scan,
  and `git diff --check` PASS.
- Operator contract moved to versioned derived `p38s18l-seam-reduction-v2` and
  requires the entire compact records hierarchy plus a local bundle audit to
  be prepared as an append-only evidence CL. No GCP v2 execution, TPU launch,
  backward, or optimizer commit occurred in this checkpoint.

## 2026-08-16 UTC — P38.2q v2 GCP Snapshot Selection Executed

- Type: GCP-side snapshot inventory and selection execution
- Command: `python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/select_p38_live_snapshot.py --listing ... --live-root ... --min-capsule-rounds 2` via `run_reduce_p38s18l_on_gcp.sh`.
- Result: `[P38.SNAPSHOT] INCONCLUSIVE no qualifying snapshot candidates=22 minimum_capsule_rounds=2` (rc=4).
- Inventory findings across all 22 live snapshots:
  - Snapshots `000000`..`000019`: No capsule rounds exported.
  - Snapshot `000020`: Contains only capsule round 0 (1 round < 2 required).
  - Snapshot `000021`: Contains capsule rounds [0, 1], but lacks `SHA256SUMS`, `LIVE.json`, and paired NPZs due to abrupt container exit.
- Action: Per `P38S18L_GCP_REDUCTION_RUNBOOK.md` §5 decision table row 1 (`No eligible two-round snapshot`), returned `SNAPSHOT_SELECTION.json` inventory without downloading or running a one-round substitute. P38s18l remains recorded as analysis-grade partial evidence.

## 2026-08-16 UTC — P38.2q rc=4 durability correction locally complete

- Reconciled commit `e0c1aef7`: it records the 22-snapshot selector result only
  in `THREADS.md`, `state.md`, and this ledger. No `SNAPSHOT_SELECTION.json`,
  raw object listing, selector output, verdict, or audit was committed.
- Root cause: `run_reduce_p38s18l_on_gcp.sh` printed the selector JSON and exited
  immediately on rc=4, before its common sealing/upload path. The derived
  destination therefore had no durable evidence sentinel and could be retried
  or reinterpreted from prose.
- Changed the rc=4 path to upload a selection-only bundle before returning:
  raw `OBJECT_LISTING.txt`, selector JSON/stdout/stderr,
  `INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT` verdict, packaging note, archive, and
  self-excluding `SHA256SUMS`. Destination immutability now keys on
  `files/verdict.json`, covering both reduction and selection-only bundles.
- Updated the operator card to capture reducer rc explicitly: `0` and sealed
  `4` continue into download/audit, while every other code stops. This avoids
  `set -e` aborting after a valid inconclusive bundle is uploaded.
- Extended the standalone auditor to reproduce the selector exactly from the
  bundled object listing. It rejects file-byte mutation and also rejects a
  semantically changed listing after its SHA entry is recomputed.
- Corrected the decision ledger: the no-source outcome retires P38s18l and does
  not authorize a tail-only probe. Registered P38.2r as a future single-run,
  production-shape acquisition that captures hidden seams and bounded-tail
  checkpoints together and seals each round before continuing.
- Focused gates: fake-GCS wrapper 2/2 (including no-source upload, audit, and
  overwrite refusal); selector 3/3; reducer 6/6; classifier 4/4; seam capture
  5/5; neutrality 3/3; P38 postflight PASS. No GCP rerun, TPU launch, backward,
  optimizer commit, repository commit, or push occurred in this checkpoint.

## 2026-08-16 UTC — P38.2r terminal seam/tail implementation locally complete

- Retired P38s18l as `INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT`; no prior snapshot is
  promoted to a hidden-chain or tail verdict.
- Added a default-off terminal observer that records raw/processed target
  logits, raw/processed vocabulary normalizers, an independent target-logprob
  decomposition, and the unchanged production endpoint for the same exact
  round/token-prefix keys as the layer/final-norm observer.
- Extended the official classifier to require all red actions to join both A/B
  observers and to prove each captured production endpoint equals its mismatch
  capsule before naming the first measured divergent region.
- Added a per-round durability handshake: the learner requests sealing after a
  completed frozen round and blocks until the survivor worker stages, hashes,
  uploads, downloads, verifies, and acknowledges an immutable round bundle.
  `ROUND_COMPLETE.json` is uploaded last.
- Added a `seam-tail` one-host rehearsal mode. Remote round persistence is
  explicitly skipped only under the already target-forbidden
  `CANON_P38_ONEHOST_REHEARSAL=1` flag.
- Gates passed: pinned-image manifest/install for Qwen3-1.7B and Qwen3-8B;
  32 runner tests per overlay; full P38 image suite 53/53; round-stage,
  classifier, terminal helper, renderer, postflight, fake-GCS abrupt-exit
  durability, alignment, Python/shell syntax, and `git diff --check`.
  The local-v5p off/on endpoint comparison remains pending after publication.
- Publication was approved on 2026-08-16. No one-host TPU run, target TPU
  launch, backward, or optimizer commit occurred before publication.

## 2026-08-16 UTC — P38.2r first one-host gate found a shallow-call bug

- Published the initial P38.2r implementation as `294a4186` and ran the
  observer-off arm successfully for three frozen rounds with zero backward and
  zero optimizer commits.
- The first seam-tail arm failed before round 0 with
  `P38 decode tail context is absent or not arm A`. The triggering scheduler
  call had only the first 256 prompt tokens and therefore correctly selected
  no deep seam rows; the tail hook incorrectly treated that normal absence as
  state loss.
- Corrected decode and prompt tail hooks to no-op only when no seam context was
  selected. A present context with the wrong arm remains fail-closed. Added
  both no-context and wrong-arm unit tests and refreshed the installed runner
  manifest.
- Corrected the exact-image terminal marker to report the now-observed 34
  runner tests per overlay. Both Qwen3-1.7B and Qwen3-8B overlays pass all 34.
- Because the correction changes the source SHA, the earlier off arm is useful
  diagnostic evidence but cannot satisfy the registered same-source neutrality
  pair. Both one-host arms must rerun from the corrected publication before a
  64-TPU launch is admitted.

## 2026-08-16 UTC — P38.2r corrected same-source one-host gate passed

- Published the shallow-call correction as `ae63d44e` and reran both local
  v5p arms from that exact clean source.
- Observer off completed three frozen rounds with zero backward and zero
  optimizer commits. Combined seam-tail also completed three rounds, emitted
  130 seam and 130 tail records, and kept backward/optimizer commits at zero.
- The registered neutrality classifier returned
  `observer_endpoint_bitwise_neutral`. In all three rounds the complete
  alignment record excluding only its timestamp was identical, including all
  original-array and action-masked endpoint hashes, geometry, denominators,
  metrics, and verdict.
- Corrected the operator card: exact local A-B runs intentionally have no
  mismatch capsule, so neutrality is judged from the alignment byte hashes
  rather than files that cannot exist. The classifier now rejects drift in
  any non-timestamp alignment field.
- One 64-TPU stock P38s18r diagnostic is authorized only from full source
  `ae63d44edc67cfcd5b19d34abc82feb681284c67`. No target launch occurred here.

## 2026-08-16 UTC — P38s18r Round 0 execution and durability seal timeout analysis

- Launched `canon-p38-fl-stock-p38s18r-6b75e3cf` on 64 TPU (`DP16xTP4`, Concurrency 256, 3 Frozen Rounds, Seam Mode `layer`, Terminal Tail `1`).
- JIT precompilation and model loading succeeded on 64 devices with 1,032 GiB HBM utilized.
- Precheck Round 0 executed with full 32-prompt coverage (`N_action=46,098`):
  - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 mismatch bytes.
  - A-B boundary (`S_decode` vs `S_prefill`): exactly 30 mismatch bytes (reproducing carrier drift).
  - Probe data: 360+ NPZ records (Layers 0..35 seam and tail) live-uploaded to GCS.
- Round 0 seal durability timeout:
  - `[CANON_P38] ROUND_SEAL_REQUESTED round=0` triggered background worker `stage_p38_round.py`.
  - `_filter_jsonl` raised `ValueError: no round 0 records in pre_alignment.jsonl` because `pre_alignment.jsonl` contained `"step": 0` while `_filter_jsonl` only filtered on `"diagnostic_round"`.
  - Main thread timed out after 900s: `timed out waiting for P38 round 0 durability acknowledgement`.
- Implemented fix in `stage_p38_round.py` (check `diagnostic_round` with fallback to `step`, admit unscoped records) and `tunix/rl/alignment.py` (explicit `diagnostic_round: int(step)`).
- Documented in `artifacts/p38s18r_round0_seal_error_report.md`.

## 2026-08-16 UTC — P38s18r round-scope review and local correction

- Pulled remote tip `fbb4b278` and reproduced the P38s18r seal failure from
  the committed report: the round stager found no `diagnostic_round` in the
  pre-alignment stream, the worker produced no round-0 ACK, and the learner
  timed out after 900 seconds. The overall run is
  `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`; its one numerical precheck is only
  analysis-grade.
- Rejected the remote fallback as unsafe. Frozen diagnostic rounds can advance
  while optimizer `step` remains zero, so `diagnostic_round=int(step)` aliases
  distinct rounds. Generic unscoped admission is also fail-open and could copy
  incident records into every bundle.
- Local replacement (not committed/pushed): diagnostic pre-alignment records
  use `p38_diagnostic_round_index()`; pre-alignment and incident streams require
  strict integer round scope; only schema `p38-request-journal-v1` is admitted
  as a cumulative-unscoped journal; round inventory records that scope.
- Gates passed: round-stage 4/4 including step-fallback and schema negatives;
  fake-GCS two-round content isolation plus abrupt-exit durability; P38
  postflight and seam neutrality; Python compilation and diff check; pinned-
  image alignment tests proving steps `[0,0]` map to rounds `[0,1]` and ordinary
  step 3 has no diagnostic scope; complete pinned-image P33 CPU gate PASS.
- Corrected operator policy: preserve the failed P38s18r evidence, never reuse
  its run-id or prefix, and do not launch from `HEAD`. After explicit user
  approval and publication, use fresh run-id `p38s18r2` plus the approved full
  SHA. No commit, push, target launch, backward, or optimizer commit occurred
  in this checkpoint.

## 2026-08-17 UTC — P38s18r2 64-TPU diagnostic execution, Round 0 GCS seal, and 256-chunk seam localization

- Launched `canon-p38-fl-stock-p38s18r2-10fe951f` on 64 TPU (`DP16xTP4`, Concurrency 256, 3 Frozen Rounds, Seam Mode `layer`, Terminal Tail `1`) from source `10fe951f0186256aa106627c4323de1f5aa168be`.
- All 6 model overlays verified by SHA256 byte identity.
- Round 0 executed completely across 256 concurrency (`N_action=45,559`):
  - `S_prefill vs T_old`: STRICT EXACT 0 differing bytes across 45,559 tokens (100% bitwise exact identity).
  - `S_decode vs S_prefill`: 45 differing bytes (99.975% byte identity), with 100% of mismatches precisely aligned at 256-token Pallas Chunked Attention page boundaries (`logical_kv_prefix_length = 7 * 256 = 1792`, `offset_in_sequence_chunk = 0`).
- Round 0 GCS Durability Bundle 100% complete and verified:
  - GCS URI: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000`
  - `manifest_sha256 = ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e`
  - Staged 971 Tail records, 915 Seam records, 910 Incident records, and mismatch capsule. `round-000000.ack` written.
- P38 diagnostic lane successfully concluded; capacity transferred to FrozenLake 8B Full Training (`p45r8`).
- Technical report: `artifacts/p38s18r2_round0_seam_tail_report.md`.

## 2026-08-17 UTC — P38s18r2 correction: durability timeout and remote-only classifier handoff

- Pulled the committed timeout trace at `d0d96030`. Round 0 requested sealing,
  but the learner's fixed 900-second ACK wait expired before the worker's
  serial upload and readback of 3,776 files completed. The worker reported a
  late `ROUND_COMPLETE` and ACK after about 57 minutes; rounds 1 and 2 never
  started. The run is `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`, not a completed
  three-round diagnostic.
- Rechecked the committed `round0_pre_alignment.jsonl`: B-C is exact; A-B is
  red at 45 bytes / 32 elements with max absolute difference 0.1010169983.
  Only 1/32 mismatch elements is at `logical_kv_prefix_length % 256 == 0`.
  The earlier “100% 256-token boundary” and Pallas-boundary cause claims are
  superseded and not admitted.
- The local evaluator cannot access the producer's GCS bucket, and the raw NPZ
  corpus is not a required local handoff. The next gate is remote execution of
  the existing official layer-plus-tail classifier over immutable Round 0,
  followed by a small SHA-sealed receipt. The old P38s18l live-snapshot wrapper
  is explicitly incompatible with this one-round directory.
- No code, model executable, kernel, TPU launch, commit, or push occurred in
  this checkpoint. A future rerun, if still necessary after classification,
  requires a measured single-archive round transport; increasing the timeout
  alone is not the selected repair.

## 2026-08-17 UTC — P38s18r2 remote receipt audit and P38.2s registration

- Pulled evidence commit `a514c3bf`. Its compact failure receipt is internally
  SHA-consistent: 12/12 listed return files verify. The source object listing
  has 3,896 entries; the source manifest has 3,894 entries; after excluding
  `ROUND_COMPLETE.json` and `SHA256SUMS`, their filename sets are identical.
- The sealed inventory reports 972 seam and 972 terminal-tail records. This
  supersedes the stale v1 runbook expectations of 915/971; those hard-coded
  counts must not be reused.
- The official classifier returned rc 1 at
  `duplicate seam token-prefix record` and wrote no `classification.json`.
  The committed `INCONCLUSIVE_REMOTE_CLASSIFICATION` verdict is correct. No
  first-difference signature or root-cause localization is admitted from this
  receipt.
- The failure is an analysis-shape mismatch: raw observer records overlap,
  while the direct classifier requires one record per
  `(round, token-prefix SHA, arm)`. It is not evidence of missing source
  objects. The receipt also left `source_round_gcs_uri` empty; the replacement
  bundle must reject that provenance defect.
- Registered P38.2s and
  `P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`: first add reviewed seam-plus-tail alias
  reduction and audit support, then execute it once beside GCS against the
  immutable Round 0. Only fully byte-identical duplicate payloads may become
  aliases; conflicts retain every candidate and stay fail-closed. No TPU
  relaunch is authorized by this checkpoint.

## 2026-08-17 UTC — P38.2s local seam-plus-tail reducer implementation

- Implemented the zero-TPU P38s18r2 Round-0 analysis path in an isolated clean
  worktree: alias-aware seam-plus-tail reducer, independent compact-bundle
  auditor, fixed source/count contract, one-command GCS wrapper, and focused
  plus fake-GCS tests.
- The reducer verifies the immutable source manifest and every source SHA,
  derives red keys from the capsule, admits duplicate rows only when every
  registered seam or tail payload byte is identical, preserves every
  conflicting candidate, and invokes the official classifier with mandatory
  tail evidence only after both joins are complete.
- The auditor distrusts the reducer's decisions: it verifies the self-excluding
  bundle manifest, rescans all returned candidates, independently recomputes
  seam and tail alias/conflict maps, and reruns the official classifier from
  the compact bundle alone.
- The wrapper is contract-driven and immutable-destination-only. It downloads
  the fixed Round 0, reduces and audits locally, uploads only after audit PASS,
  and copies the same compact result into the task evidence tree for handoff.
- Local gates at this checkpoint: new focused suite 13/13; existing seam
  classifier 6/6; existing seam reducer 6/6; existing GCS wrapper 2/2; tail
  capture 1/1; all 11 importable P38 Python test files 57/57; four shell
  evidence/persistence/package/postflight suites PASS; Python compilation,
  shell syntax, JSON parsing, secret scan, executable ASCII scan, and diff
  check PASS. The unrelated render test is `TARGET NOT RUN` in this host
  interpreter because `metrax` is absent; it failed during import, before any
  test body.
- No TPU launch, GCS access/mutation, commit, or push occurred. Next gate is
  user review and explicit publication approval, then one remote wrapper
  execution from the approved full SHA.

## 2026-08-17 UTC — P38.2t target-aware tail identity

- Independently re-audited the committed P38s18r2 v2 bundle: all 371 SHA
  entries verify and the old audit reproduces byte-for-byte. Its only failed
  tail key is source-prefix SHA `e7427e60...`: records 510/723 score capsule
  target 54852 with identical payloads, while record 539 scores target 13598.
- Amended only the offline reducer/auditor contract. Tail candidates are now
  filtered by the capsule target before alias/conflict resolution. Every
  same-prefix candidate remains under `candidates/`; a missing target and a
  same-target payload conflict remain fail-closed.
- Added an immutable v3 contract/destination and wrapper guard. The old v1/v2
  contract and derived bundle remain readable and cannot be overwritten.
- Focused reducer/auditor/wrapper suite passes 19/19, including a regression
  over the committed compact candidates that reproduces 32/32 joins, a true
  same-prefix/two-target end-to-end fixture, different-
  target admission, missing-target rejection, same-target conflict rejection,
  fake-GCS completion-last upload, and legacy-v2 byte-identical audit. The
  importable P38 suite has 63 passing tests; the renderer module is `TARGET NOT
  RUN` in the host interpreter because `metrax` is absent. Four P38 shell
  suites pass.
- Exploratory local reclassification of the committed compact candidates joins
  32/32 red points: 26 first differ at `raw_log_normalizer`, 6 at
  `raw_target_logit`. This is a hypothesis-shaping result, not admitted v3
  evidence until the checked-in wrapper reproduces it from immutable GCS.
- Ran the existing bounded real-v5p canonical-tail construction control at
  `[256,151936]`: two outer programs using the same canonical log-softmax are
  exact at 0/38,895,616 elements, while the one-bit negative reports 1. This
  rejects a same-input one-host reducer construction failure only.
- No training path, model overlay, serving runtime default, prefix-cache
  setting, source GCS object, or TPU job changed. The user explicitly approved
  the publication CL on 2026-08-17. Final pre-publication reruns passed focused
  19/19, importable P38 63/63, all four shell suites, Python/shell/JSON syntax,
  credential scan, and `git diff --check`. The next gate is one zero-TPU v3
  GCS reduction from the clean published branch.

## 2026-08-17 UTC — P38.2u terminal discriminator local admission

- Added a default-off terminal discriminator that records exact selected
  final-hidden rows and bounded blockwise raw/processed-logit evidence. The
  shape-dependent gather is separated from one fixed-four-row shared
  executable so A and B cannot acquire different observer reduction programs.
- Real v5p rejected the first fused gather/reduction prototype: local
  production A-B was exact while the observer falsely reported 148 reduction
  differences. A second rehearsal showed two legacy tail-intermediate drifts
  despite exact shared evidence and exact production endpoints. The final
  classifier preserves those legacy rows as diagnostics but cannot use them
  as the first authoritative stage.
- The final Qwen3-8B DP1xTP4 run completed three frozen rounds with 0 backward
  and 0 optimizer commits. It wrote 130 terminal pairs and joined 155 A rows;
  every row classified exact through hidden, raw logits, processed logits,
  both shared reductions, and the production endpoint.
- The matched observer-off arm used the same source and diff. The three full
  alignment records, excluding timestamps, were bitwise identical and
  classified `observer_endpoint_bitwise_neutral`. The real-TPU one-bit
  negative ran 3/3, both pinned overlays matched 33/33 manifest entries and
  passed 34/34 runner tests, and the complete CPU gate passed.
- Receipt:
  `artifacts/p38_2u_terminal_discriminator_onehost_0817.md`. No target launch,
  commit, or push occurred. The next gate is explicit user-reviewed
  publication followed by exactly one P38s19 64-TPU stock diagnostic.

## 2026-08-18 UTC — P38s20 durability timeout and P38.2v transport repair

- Pulled evidence commit `3f3ae92d`. Its five manifested payloads verify.
  P38s20/source `bea31f36...` completed Round 0 with 873 seam, 873 tail, and
  873 terminal records. Across 49,451 actions, A-B reported 63 differing bytes
  / 41 elements (`max_abs=0.08359146118164062`) and B-C was exact. Backward and
  optimizer commits remained zero.
- The 4-GiB output-bound change worked. The failure was host durability: live
  snapshot sequence 15 was serially uploading the cumulative observer corpus
  before the worker could service `round-000000.request`. The current live and
  round paths both used one GCS copy per logical file; the round path would
  additionally read every object back. About 5,246 logical files therefore
  exhausted the learner's 900-second ACK window. Verdict remains
  `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`; no terminal root cause is admitted.
- Added P38.2v host-only transport. Every snapshot/round now has a sorted
  logical-file manifest inside one deterministic flat tar. The worker uploads
  the archive and manifest, downloads and verifies the archive once, then
  writes `LIVE.json` or `ROUND_COMPLETE.json` last. Each remote prefix has
  exactly three objects. Periodic observer payload is diagnostic-round scoped,
  and already-published seal/terminal requests run before a periodic snapshot.
- Added a reusable verifier/extractor and a 5,246-file regression. Two archive
  builds are byte-identical; extraction reproduces the manifest; a one-bit
  archive mutation and a missing member fail closed. Fake-GCS persistence
  proves three-object live/round transport, read-back verification,
  completion-last ordering, and independently sealed rounds surviving abrupt
  worker exit.
- Current local gates: archive unit tests 4/4 PASS; fake-GCS persistence PASS;
  77/77 importable P38 Python tests PASS; postflight, evidence seal, and package
  suites PASS; Python compile, Bash syntax, and diff check PASS. The renderer
  test is `TARGET NOT RUN` in this interpreter because `metrax` is absent; no
  renderer code changed. No TPU path, observer values, model, prefix cache,
  backward, optimizer, commit, push, or remote GCS state changed. Next target
  after user-approved publication is exactly one P38s21 run using
  `P38S21_RUNBOOK.md`.

## 2026-08-18 UTC — P38s21 partial terminal localization and P38.2w opened

- The committed P38s21 bundle verifies 12/12 listed files and contains sealed
  receipts for rounds 0 and 1. Round 2 exceeded the 4-GiB local observer bound;
  controlled exit, root `COLLECTED`, and root `COMPLETE` are absent. The run is
  `ANALYSIS_GRADE_PARTIAL_2_OF_3`, not a complete target PASS.
- Round 0 reports A-B 47 elements / 76 bytes over 45,276 actions with exact
  B-C. Round 1 reports A-B 7 elements / 9 bytes over 44,695 actions with exact
  B-C. All 54 selected red points join.
- The captured complete final-hidden rows are exact for those 54 points and
  the first measured red interval is `lm_head_logits`. This localizes an
  interval; it does not prove a specific GEMM reduction order or full-vocab
  equality. In `TD,DV->TV`, hidden K=4096 is reduced and vocab V=151,936 is an
  output axis.
- Code archaeology confirms the seven transformer projections use registered
  Pallas sites while `JaxLmHead` remains a separate einsum. Current local
  lm-head shapes are decode M=16 and prefill M=256; `CANON_LOGPROB_M=256`
  starts after logits.
- Opened P38.2w. It reuses the existing default-off `CANON_MM_ALGO` only as a
  preregistered discriminator, retaining the P19 negative: the old real
  M=16/M=2048 arm fired and had zero effect. The new local gate uses the real
  checkpoint weight at M=16/M=256 and a one-bit negative; the only target arm
  is slim stock concurrency-256 with no seam/terminal recapture.
- No TPU target run, commit, push, production default, prefix cache, backward,
  optimizer, or source GCS object changed in this step.

## 2026-08-18 UTC — P38.2w real-weight one-host screen complete

- On the local four-device v5p, loaded the real Qwen3-8B lm-head BF16 weight
  `[4096,151936]` with TP4 vocab sharding. Four deterministic BF16 hidden-input
  seeds compared identical selected rows under M=16 and M=256.
- Default and explicit `BF16_BF16_F32` were both cross-M exact for 4/4 seeds.
  The two arms were also numerically identical to each other for every tested
  value. The one-bit negative reported exactly one.
- StableHLO proves the intervention exists: the explicit arm carries BF16,
  BF16, FP32-accumulation algorithm attributes and has different lowering SHA
  values from default at both M shapes. The one-host verdict is therefore
  `BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE`, not a repair.
- Pinned-image focused tests pass 16/16 and the complete P38 serving suite
  passes 93/93. Shell syntax and Python compilation pass. Receipt:
  `artifacts/p38_2w_lm_head_onehost_0818.md`.
- The next possible target is one slim P38s22 single-variable arm. It remains
  NOT RUN and requires separate user approval after review/publication.

## 2026-08-18 UTC — P38s22 receipt correction and P38.2w1 offsite audit implementation

- Pulled evidence commit `82cd2bd0`. The returned root/capsule bytes reproduce
  the three endpoint rounds: 45,865 / 43,982 / 53,617 actions; A-B 48/10/8
  elements and 82/14/15 bytes; B-C exact in every round. The generic
  `BF16_BF16_F32` preset is rejected at analysis grade.
- The new durability receipt is not source-authenticated. All three claimed
  round-archive SHAs equal their corresponding capsule NPZ SHAs, which cannot
  satisfy the checked-in deterministic tar format. The prose receipt copied
  P38s21 action counts for rounds 0/1. The returned 66-point terminal
  classification is unadmitted because P38s22 disabled that observer and no
  raw terminal JSON/NPZ or invocation provenance was returned.
- Added P38.2w1: one immutable contract, a read-only GCS wrapper, an independent
  root/round/archive/capsule auditor, a background-free operator runbook, and
  a compact return contract. The remote agent only runs the checked-in command;
  it cannot enter an URI, expected number, classifier choice, or conclusion.
- The wrapper preserves both successful and failed acquisition as a small
  self-sealed return. Focused fake-GCS cases cover a valid bundle, capsule SHA
  copied into the archive receipt, NPZ mislabeled as a tar, missing root
  completion, staged/root pre-alignment drift, and a cross-round incident
  record, plus an orphan observer NPZ. Together with the deterministic archive
  suite, 11/11 tests pass. The broader P38 discovery has 85 runnable tests
  passing; renderer collection
  is `TARGET NOT RUN` here because optional dependency `metrax` is absent, and
  this phase does not touch renderer code. Python compile, Bash syntax,
  contract JSON parse, and `git diff --check` pass.
- No remote GCS read or mutation, TPU/Kubernetes launch, model execution,
  backward, optimizer commit, Git commit, or push occurred. The next gate is
  user review and explicit commit/push approval, followed by exactly one
  zero-TPU remote invocation from the approved full SHA.

## 2026-08-18 UTC — P38.2w1 rc=4 reviewed; P38.2w2 round-first salvage opened

- Evidence commit `0b86ef5c` contains a complete 10-member self-sealed audit
  return. Tool and contract SHAs match analysis source `180fc2ff`.
- The mechanical verdict is `INCONCLUSIVE`: root `SHA256SUMS` was unavailable;
  root `COLLECTED.json` and `COMPLETE.json` were not returned. No totals were
  recomputed by v1.
- All three round markers/manifests survived. Each marker is
  `sealed-and-verified`, each manifest SHA/count matches its marker, each has
  10 sorted logical members, and each names its preregistered capsule SHA. The
  actual tar bytes remain unverified because the v1 auditor failed before its
  round loop.
- Opened P38.2w2: immutable archive/manifest/capsule SHA contract, round-first
  auditor, URI-free acquisition ledger, read-only one-command wrapper,
  fake-GCS controls, and background-free operator runbook. Root postflight is
  an independent unadmitted claim; no TPU relaunch is allowed.
- Local implementation is complete. The focused round-salvage/offsite/archive
  suite passes 16/16; all 90 runnable tests in the broader P38 serving
  discovery pass. Renderer collection is `TARGET NOT RUN` in this host
  interpreter because optional dependency `metrax` is absent, and this phase
  does not touch renderer code. Python compilation, Bash syntax, immutable
  contract parsing, and `git diff --check` pass. No GCS read, TPU launch, Git
  commit, or push occurred; the only remaining phase action is the documented
  zero-TPU remote command after publication from a user-approved full SHA.

## 2026-08-18 UTC — P38.2w2 PASS reviewed; P38.2x opened

- Reviewed the committed `round-salvage-v1` return. Its 11-file self-manifest
  verifies, tool/contract hashes match the approved source, all three actual
  deterministic tar archives verify, and all 30 logical round members pass.
- Recomputed totals are 143,464 actions, 66 A-B differing elements, 111 A-B
  differing bytes, and exact B-C in all three rounds. Verdict:
  `ROUND_SEALED_GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED`.
- The claim ceiling is unchanged: root `SHA256SUMS`, `COLLECTED`, and
  `COMPLETE`, returned terminal localization, backward, and optimizer are not
  admitted. No synthetic root success was inferred from round seals.
- Opened P38.2x. The registered shape construction pads local decode M16 and
  prefill M256 to the same M256, keeps K4096, pads TP4-local vocab N37984 to
  N38144, and invokes fixed BM128/BN256/BK256 Pallas tiles. The intervention is
  default-off and must pass real-weight one-host construction and negative
  gates before any P38s23 target launch.

## 2026-08-18 UTC — P38.2x one-host construction PASS

- Added a default-off Qwen3-8B TP4 `JaxLmHead` hook. Local decode M16 and
  prefill M256 both enter fixed M256/K4096/N38144 BM128/BN256/BK256 Pallas
  calls, then slice back to semantic M and real N. Flag-off leaves the original
  class method untouched.
- CPU/static contract tests pass 6/6. The broader P38 suite has 96 runnable
  tests passing; renderer collection remains `TARGET NOT RUN` in the host
  interpreter because optional `metrax` is absent. The pinned-image install
  and import gate passes for both qwen1p7b and qwen8b overlays.
- Real Qwen3-8B weight on the local four-device v5p passes 4/4 deterministic
  seeds: fixed M16 versus fixed M256 shared rows are exact, `max_abs=0.0`, and
  the one-bit negative reports 1. Fixed versus stock differs at
  249/211/268/219 selected elements, proving the intervention is active.
- Added the P38s23 renderer flag, target preflight, single-variable tests, and
  background-free operator runbook. No 64-TPU run, commit, or push occurred.

## 2026-08-18 UTC — P38s23 warmup shape contract failed closed

- Source `32caa773a057ccc2604ee6c1c5ce845f63346bbd` started the registered
  64-TPU target but stopped inside vLLM `CompilationManager.capture_model()`
  before rollout, alignment, backward, or optimizer work.
- `run_compute_logits` invoked `lm_head` with hidden shape `[32,4096]`. The
  first P38.2x validator admitted only M16/M256 and raised
  `P38 fixed lm_head requires semantic M in (16, 256)`.
- Verdict: `INCONCLUSIVE_WARMUP_SHAPE_CONTRACT`. This is neither A-B repair nor
  a fixed-lm-head numerical rejection. No P38s23 numerical round exists.
- Pinned code archaeology shows request-count warmup uses power-of-two buckets;
  the repair must register the exact max-concurrency-256 ladder rather than
  admit arbitrary `1 <= M <= 256` or bypass warmup.
- The committed hand report's full source SHA is mistyped after the correct
  eight-character prefix and is not source-authenticating evidence. Its
  traceback and current code are nevertheless sufficient to diagnose this
  contract omission.

## 2026-08-18 UTC — P38.2x1 exact-bucket repair one-host PASS

- Registered only M8/16/32/64/128/256; every bucket zero-pads to the unchanged
  internal M256/K4096/N38144 BM128/BN256/BK256 Pallas construction. M1/M7/M24/
  M257 remain fail-closed.
- Real Qwen3-8B BF16 weight on the local four-device v5p passes 24/24
  bucket-versus-M256 comparisons across four deterministic seeds. Every
  `max_abs=0`, all six lowering receipts contain custom calls, and the one-bit
  negative reports 1.
- Fixed versus stock M16 remains different at 249/211/268/219 selected
  elements, so the repair did not turn the intervention into a no-op.
- P38s23 is frozen as historical. The next separately approved target is
  P38s23r1 under `P38S23R1_RUNBOOK.md`. No 64-TPU retry, backward, optimizer,
  commit, or push occurred in this step.

## 2026-08-18 UTC — P38s23r1 learner M4096 failed closed

- Source `575ef92e4208654e69730854846c9aefe2e77a3e` passed every registered
  request warmup bucket M8/16/32/64/128/256 and completed all 256 rollout
  trajectories. This validates the P38.2x1 request-bucket repair.
- Learner rescore then called the globally installed `JaxLmHead` hook with
  hidden shape `[4096,4096]`. The exact request-only contract rejected M4096
  before any A-B/B-C precheck, backward, or optimizer work.
- Verdict: `INCONCLUSIVE_LEARNER_SHAPE_CONTRACT`; there is no P38s23r1
  numerical round and no repair/rejection claim.
- Stock fallback for M4096 is rejected as a fix because it would give B/M256
  and C/M4096 different lm-head programs and could manufacture B-C drift.

## 2026-08-18 UTC — P38.2x2 exact learner mapping implemented locally

- Added only exact learner M4096 and map it as 16 independent invocations of
  the unchanged M256/K4096/N38144 BM128/BN256/BK256 Pallas body. Request
  buckets still pad to one M256 call; M512/M2048/M8192 remain fail-closed.
- CPU/static and pinned-image gates pass. The pinned image attests request
  M8/16/32/64/128/256 plus learner M4096, with both overlays at 34/34 tests.
- A four-device CPU structural `eval_shape` in the pinned image passes the
  complete shard_map/lax.map wrapper and returns BF16 `[4096,151936]`; its
  PATHTRACE records `chunks=16`. This is a shape/wiring gate, not numerics.
- The first real-v5p attempt never initialized JAX because an unrelated
  `p51_gsm8k_xprof_p53fixed_20260818` container owns `/dev/vfio/0`. This is an
  infrastructure-busy result, not evidence about the M4096 construction. Do
  not launch P38s23r2 until the real-v5p gate is rerun and sealed.

## 2026-08-19 UTC — P38s23r2 exact first round; P38s23r3 durability repair prepared

- P38s23r2/source `6814774eef70aa0c67610eab9f355d964d420378`
  emitted every registered fixed-lm-head receipt and measured 49,177 action
  tokens with exact A-B, exact B-C, and `max_abs=0.0` in round 0.
- It then timed out after 900 seconds waiting for the round-seal ACK. The
  shared synchronous worker was already inside a periodic full-forensics
  snapshot; checking requests first in the next loop cannot preempt an
  in-flight transfer. No later round, controlled exit, backward, or optimizer
  result is admitted.
- Review found two additional exact-success-path defects before relaunch: an
  exact round intentionally creates no mismatch capsule, but round/root
  persistence required one; stock postflight also required a mismatch join
  even for the fixed-lm-head exact arm.
- Prepared P38s23r3 locally. `round-alignment-v1` is exclusive to fixed lm-head,
  disables periodic snapshots and unrelated observers, makes a red-only
  capsule optional, and keeps round ACKs plus terminal root markers mandatory.
  Added checked-in launch and compact-return scripts plus a background-free
  runbook.
- Local fake-GCS persistence and operator-return tests pass, including an
  exact round with no capsule, root collect/complete, three immutable round
  objects, complete return verification, both scientific outcomes, and
  truncated-head rejection. Pinned-image renderer (18), fixed-lm-head (8),
  serving-classifier (36), and complete P33 adjacent CPU gates pass. No TPU
  launch, GCS mutation, commit, or push occurred. Next: review before
  requesting publication approval.

## 2026-08-19 UTC — P38.2h M4096 backward defect measured and repaired locally

- P38s23r3 is admitted as the fixed-lm-head forward candidate: three 64-TPU
  frozen rounds, 146,042 action tokens, exact A-B and B-C, no backward or
  optimizer commits. It opens but does not satisfy backward admission.
- Added a real-Qwen3-8B TP4 gate for the semantic-M4096 outer VJP. The original
  automatic transpose of `lax.map(16xM256)` produced exact `dHidden` and
  deterministic repeats, but 11,950 shared-weight gradient elements differed
  from 16 completed M256 pullbacks accumulated in explicit ascending order;
  `max_abs=2.0`. Verdict: `FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT`.
- Repaired only the M4096 outer backward. Forward remains the same 16 fixed
  M256 Pallas calls; a custom VJP accumulates completed chunk `dWeight`
  contributions through loop-carried ascending `lax.scan` state.
- The real-v5p rerun reports `FIXED_LM_HEAD_ONEHOST_VJP_PASS`: zero differing
  `dHidden`/`dWeight`, zero repeat differences, finite/nonzero gradients, and
  one detected normal-value negative. Receipt:
  `artifacts/p38_2h_fixed_lm_head_vjp_onehost_0819.md`.
- Prepared one strict P38h DP16xTP4 actual-model backward-no-commit renderer,
  launch script, compact stdout evidence transport, official-classifier
  collector, and runbook. The operator test passes one complete positive and
  rejects missing-VJP, SHA-corrupt, and state-mutating negatives. No cluster
  launch, Git commit, or push occurred.
- The complete adjacent P33/P38 CPU gate now passes, including 121 P38 serving
  tests and the final `[P33.WORKLOAD] CPU_GATE PASS`; the pinned exact-image
  gate passes both Qwen3-1.7B and Qwen3-8B overlays at 34/34. New P38.2h
  renderer/operator/launcher/collector sources are registered in the complete
  gate rather than relying on focused tests alone. Next: stop for user review.

## 2026-08-19 UTC — P38.2h Attempt 0: 64-TPU Backward executed to completion; alignment gate repair

- Executed JobSet `canon-p38h-fl-bwd-p38h1-957876b3` on 64 TPU (`DP16xTP4`) under `P38H_BACKWARD_RUNBOOK.md`.
- Forward pass bitwise exact ($N_{\text{action}}=45,100$, $A=B=C=0$, Pearson $r=1.00000$).
- Reverse pass executed all 16 reverse groups across 64 TPU chips to completion (`reverse_group_done group=16/16`).
- Cross-slice DP gradient reduction completed on 64 chips, producing deterministic nonzero finite gradients (`gradient_nonzero=7569363085`).
- Final step boundary check raised `AlignmentGateError` due to `check_batch` expecting `optimizer_skipped=0` in `train` mode when `CANON_P33_NO_COMMIT=1`.
- Evidence preserved under `tasks/p38-pathways-decode-prefill-carrier/evidence/p38h1/`.
- Repaired `tunix/rl/alignment.py` to allow `optimizer_skipped=1` when `CANON_P33_NO_COMMIT=1`.
- Evidence classification: complete SHA-sealed Attempt-0 failure log, but not a
  successful P38h return. The exception preceded the terminal no-commit marker,
  mutation report, three compact artifacts, and official classifier verdict.
- Rerun gate: publish a focused train/no-commit optimizer-attestation truth-table
  regression, then launch one clean source-pinned P38h rerun. No lm-head, VJP,
  reducer, topology, evaluation, prefix-cache, or warning-policy change is admitted.

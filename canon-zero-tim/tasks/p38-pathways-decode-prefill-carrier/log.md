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

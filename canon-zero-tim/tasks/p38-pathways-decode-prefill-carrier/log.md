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

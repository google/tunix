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

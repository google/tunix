# P39 log

## 2026-08-10

- Created an isolated worktree from
  `697a29ab4b27015297af8e3dbb37c49db3560445`; the original dirty P36/P37
  worktree was not modified.
- Compared the P34 command with `yuxzhang/deepswe-quality-fix` at
  `023978b976dd6d94e7a42948c3f3a68e34d73744`.
- Found one real launch blocker: `sampler_is=None` was rejected by the generic
  alignment guard before backward.
- Initially suspected a P33 workload-name/TP4 preflight conflict, then withdrew
  it after the renderer-to-`00_env.sh` test passed. That code is gated by
  `CANON_P32_DP_ADMISSION=1` and is not on the P34 path. The unrelated edit was
  reverted.
- Added a renderer-to-`00_env.sh` positive test and two one-fault negative
  controls. The first execution exposed a test-fixture-only missing state
  directory; the fixture was corrected and the gate passed.
- Static gate result: `P34_STATIC_PASS suites=7`.
- Pinned exact-image result: `P34_EXACT_IMAGE_CPU_PASS unit_cases=45
  pallas_cases=1 contract_cases=5 scheduler_cases=1 overlay=qwen32b`.
- The adjacent P33 sampler contract passed 5/5 in the pinned image. The host
  invocation is `INCONCLUSIVE` because the host lacks `metrax`; no dependency
  was installed or changed.
- Before publication, fetched `yuxzhang/canon-zero-tim` at
  `0fe5f6609df06895d93cbf2e54cada22ad7f2697`. Its only change since the P39
  starting point was `cluster/jobset-64chip.yaml`, so it does not overlap the
  P39 change set. The final commit is rebased onto that revision.

No cloud action, commit, push, PR, credential change or production-default
change was performed.

## 2026-08-11

- Fast-forwarded the isolated P39 worktree from `c9df5852` to the published
  `5ee6dbfb` base without rewriting history or touching the P38 worktree.
- Split the active operator routing: P38 remains the 64-chip GSM8K/FrozenLake
  ledger and P39 remains the 4x8x8 DeepSWE ledger. Replaced the stale P33
  directory-wide apply example with two explicit admitted manifests.
- Added explicit YAML-string serialization and a toxic-prefix control.
  Rendering source `022893e200000000000000000000000000000000`
  printed `canon.zero-tim/source: "022893e2"`; parse-back returned
  `type=str`.
- Added a pre-rescore P34 exact weight gate using the existing device-side
  mapped-trainer versus live-engine comparison. Each update fsyncs one
  `weight_attestation.jsonl` row; stale paths, missing rows, duplicate rows
  and one mismatching leaf are rejected.
- Static result: `P34_STATIC_PASS suites=7`.
- Pinned-image result: `P34_EXACT_IMAGE_CPU_PASS unit_cases=54
  pallas_cases=1 contract_cases=5 scheduler_cases=1 overlay=qwen32b`.
- Adjacent P33 regressions remained green:
  `P33.WORKLOAD CPU_GATE PASS workloads=2 p35_postflight=1
  p35_stage_probe=1` and
  `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5
  overlays=2`.
- No cloud action, commit, push, PR, credential change, precision change,
  production-default change or target numerical verdict occurred.

## 2026-08-11 UTC — P39.2: register a 64-chip TP8 resident pilot

- Type: decision
- Fact: The existing P34 target is one 4x8x8 slice split into two DP16xTP8 128-device roles with pinned-host optimizer offload. A 64-chip split pilot necessarily uses two DP4xTP8 32-device roles and is not covered by that contract.
- Action: Registered a default-off bounded pilot that validates rollout TP8, trainer TP8, DP4 reduction, cross-role weights, and device-resident optimizer capacity before any 256-chip promotion.
- Command: omitted; this checkpoint records plan-only work.
- Result: No code, cloud resource, training process, credential, commit, or branch was changed.
- Files/artifacts: `phases/p39-2-64chip-tp8-resident-pilot.md`; `plan.md`; `state.md`; `HANDOFF.md`
- Rollback: Ignore the pilot profile and retain the existing DP16xTP8/offload P34 contract.
- Next: Implement the pilot profile, renderer, arithmetic, and CPU negative controls.

## 2026-08-11 UTC — P39.2: implement the bounded 64-chip TP8 pilot

- Type: implementation and local evidence
- Action: Added a default-off DeepSWE pilot profile and renderer for one
  4x4x4 slice. It creates two disjoint 32-device roles, each DP4xTP8, and
  rejects DP16 geometry, FSDP, optimizer offload, floating client images,
  retries, missing online W&B, and unbounded full training.
- Action: Registered pilot arithmetic separately from P34: 64 global
  trajectories, 16 per DP rank, 16 fixed-order gradient groups, local M256,
  global M1024, 16 requests per rank, and 64 global requests.
- Action: Wired `90_run.sh` to select the dedicated P39 classifier whenever
  `CANON_P39_64CHIP_PILOT=1`. P34 production runs continue to use the P34
  classifier.
- Action: Added one-update and three-update classifier contracts for exact
  cross-role weights, nonzero backward, DP4 reduction/replica equality,
  device-resident optimizer state, zero P30 host transfers, HBM telemetry,
  IFRT health, and online W&B. Pathways may report either the trainer role or
  the full proxy inventory, but fewer than 32 devices rejects.
- Action: Wrote the launch and evidence procedure in
  `../../cluster/P39_DEEPSWE_64CHIP_PILOT_RUNBOOK.md`.
- Command: `bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh`
- Result: PASS; 15 tests and terminal marker
  `P39_DEEPSWE_PILOT_CPU_PASS`.
- Command: `bash canon-zero-tim/tests/p34_deepswe/run_static.sh`
- Result: PASS; terminal marker `P34_STATIC_PASS suites=10`, confirming the
  existing DP16xTP8/offload production contract was not loosened.
- Boundary: No 4x4x4 target, rollout, model initialization, backward, optimizer
  commit, HBM measurement, W&B run, cloud action, commit, push, or 256-chip
  promotion occurred.
- Rollback: Do not render the pilot. The P34 DP16xTP8 profile remains
  pinned-host offload.
- Next: Publish after approval, rerun both gates at the publication SHA, then
  operate the one-update pilot. A PASS admits the three-update confirmation,
  not a 256-chip launch.

## 2026-08-11 UTC — P39.3: defer the 64-chip pilot and select the 256-chip topology

- Type: operator decision and handoff correction
- Fact: A complete 4x8x8 slice is now available. The published P39.2 pilot is
  a DP4xTP8 device-resident capacity experiment; it does not validate the
  production DP16xTP8 topology and is unnecessary when the 256-chip run keeps
  pinned-host optimizer offload.
- Fact: The P34 runbook still described the retired Step 65 temporary JAX
  client as the active 256-device gate. Commit `6fbe8fdc` disabled that probe
  because disconnecting it could cancel the shared Pathways session. The real
  training process already fails closed on device count, 4x8x8 extents,
  disjoint/exhaustive role halves, and host-complete placement.
- Decision: Defer, but do not promote, the 64-chip pilot. Select the direct
  4x8x8 DP16xTP8 production geometry with optimizer offload. Continuous full
  training still requires a separately reviewed, default-off production
  warning-only alignment admission; the checked-in production profile remains
  strict.
- Commands: `git pull --ff-only origin yuxzhang/canon-zero-tim`; `bash
  canon-zero-tim/tests/p34_deepswe/run_static.sh`; `bash
  canon-zero-tim/tests/p34_deepswe/run_trajectory_cpu.sh`; `bash
  canon-zero-tim/tests/p34_deepswe/run_update_cpu.sh`; `bash
  canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh`.
- Result: branch already current at `7328cde7`; P34 static, trajectory, and
  update gates passed; P39 pilot CPU gate passed 15 tests. No target or cloud
  action occurred.
- Rollback: Retain strict production alignment, pinned-host optimizer offload,
  and do not render or apply a P34 JobSet. The 64-chip pilot remains available
  as a future optional capacity experiment.
- Next: implement the production warning-only contract or deliberately choose
  strict `backward-no-commit`, then rerun exact-image validation at the final
  publication SHA.

## 2026-08-12 UTC — P39.4: register the direct 32B full-training contract

- Type: operator decision and implementation checkpoint
- Source: clean worktree from `yuxzhang/canon-zero-tim` at
  `4e4ca2891a01448f09428affd1eb2434bbd61657`; the existing dirty P43/P38
  worktree and `main` were not modified.
- Decision: use one 4x8x8 DP16xTP8-per-role `full` run directly.  Device
  optimizer state is the default.  Separate one/three-update jobs are not
  prerequisites.
- Decision: pin the R2E-Gym subset revision and the checked-in 1851-image clean
  whitelist; persist every real trajectory batch and solve/group metrics.
- Decision: an all-zero-signal batch is recorded and committed normally; no
  resampling, skip-commit or signal injection is admitted.
- Decision: finite A-B and B-C mismatches are warning-only.  Nonfinite,
  topology, exact-weight, replica, optimizer, artifact, OOM and IFRT failures
  remain hard errors.
- Result: phase registered; implementation and local gates are still pending.
- Boundary: no cluster resource, model, rollout, backward, optimizer commit,
  credential, commit or push was performed.

## 2026-08-12 UTC — P39.4: complete local implementation and gates

- Type: implementation and local evidence
- Source: implementation began at
  `4e4ca2891a01448f09428affd1eb2434bbd61657`, then fast-forwarded to
  `a9432ad21af3fcf0ac87c74dfc9165eeaa136539`.  The intervening change touched
  only `canon-zero-tim/debug_logs/README.md` and did not overlap DeepSWE.
- Publication synchronization: before push, the operator branch advanced to
  `4a2cb8cd2bff2e1e9f5f82a6d2e0575d166759bd`.  The unpublished P39.4 commit
  rebased cleanly; that commit touched only FrozenLake/P38 files outside this
  change set.
- Action: fixed the optimizer boolean CLI so the launch uses the unambiguous
  `--no-optimizer-offload` form; P34 full, P39, P43 and P44 now all preserve
  device-resident optimizer semantics without an automatic host fallback.
- Action: pinned the subset dataset revision, source/filtered row counts and
  exact 1851-image clean whitelist.  Added fail-closed per-batch compressed
  trajectory and solve/group metrics capture before backward.
- Action: corrected all-solved classification to use the configured eight
  generations.  `effective_prompt_groups == 0` and finite zero gradients are
  quality telemetry and do not resample, inject signal or skip the normal
  optimizer transaction.
- Action: admitted finite A-B, B-C and downstream alignment residuals as
  convergence-only warnings for P34 `full`.  Nonfinite and structurally
  invalid records remain hard failures.
- Commands/results: `P34_STATIC_PASS suites=10`;
  `P34_TRAJECTORY_CPU_PASS tests=5`; `P34_UPDATE_CPU_PASS tests=5`;
  `P39_DEEPSWE_PILOT_CPU_PASS`; `P43_DEEPSWE_DEBUG_CPU_PASS`;
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`.
- Exact image: `P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3
  pallas_cases=2 contract_cases=5 scheduler_cases=1 overlay=qwen32b`.
  A finite mismatch returned `PASS_WITH_ALIGNMENT_WARNINGS`; an injected NaN
  in B-C returned `FAIL`.
- Local render check: a non-launchable manifest using a dummy digest-pinned
  image rendered with `P34_JOBSET_RENDER_PASS` at source `a9432ad2`; inspection
  confirmed `full`, 1000 updates, device optimizer, production trajectory
  capture, finite alignment warning-only, dataset revision pin and 1851-row
  clean-data gate.  This manifest was written only to `/tmp` and must never be
  applied.
- Validation: changed Python files passed `py_compile`, shell entry points
  passed `bash -n`, and `git diff --check` passed.
- Boundary: the exact image had no `/dev/vfio` TPU device and ran Pallas in
  interpret mode.  No 4x8x8 resource, Qwen3-32B model initialization, real
  rollout, backward, optimizer commit, W&B run, credential, commit or push was
  performed. Target status remains NOT RUN.
- Publication decision: the operator subsequently authorized commit and push
  to `yuxzhang/canon-zero-tim`.  The target run remains separately gated and
  was not authorized by that publication decision.

## 2026-08-12 UTC — P39.4 target Attempt p34r02 stops without a failure record

- Type: target evidence audit
- Source: `d725f078487ec1b8dc07d27db61d27b446af94f0` from the raw `[sync] HEAD`
  marker.  Evidence:
  `../../debug_logs/p34_p34r02_deepswe_full.raw.log`, SHA-256
  `375b600e5d234e817810f40008d50bac529d6c81e1088e99fc859d82f8da7e08`.
- Confirmed PASS: attempt zero, source provenance, six overlay files, pinned
  R2E-Gym installation and bounded patch, local Qwen3-32B checkpoint, signed
  CLI, dataset revision and 4578 source rows, exact 1851-image clean join, 256
  devices across 64 four-device hosts, and two disjoint 128-device DP16xTP8
  role meshes.
- Last evidence: line 163 is the train-mesh print.  Control flow next enters
  replicated-parameter sharding and `create_model_from_safe_tensors`.
- Missing: no model-load DONE/HBM marker, Python traceback, OOM, IFRT
  disconnect, container exit reason/signal, rollout, trajectory, alignment,
  backward, optimizer placement/commit, checkpoint, or classifier record.
- Classification: `INCONCLUSIVE_MODEL_INITIALIZATION`.  The log does not prove
  a DeepSWE code failure.  The kubeconfig warning is nonfatal at this point and
  occurs before later PASS markers; it is not the cause of this stop.
- Next: recover Pod termination JSON, JobSet events, the persistent PVC
  `run.log`, and `pathways-proxy`/`pathways-rm` logs.  If unavailable, add
  explicit model-load START/DONE and head RSS/cgroup-memory telemetry, keep the
  same 32B/data/topology/device-optimizer contract, and rerun.
- Boundary: no cluster query was possible on this workstation because
  `kubectl` is not installed.  No training code, cloud object, commit, or push
  was changed by this audit.

## 2026-08-12 UTC — P39.4 p34r02 complete-log correction and local repair

- Type: evidence correction, root-cause analysis and local implementation.
- Source synchronization: fast-forwarded the isolated worktree to
  `42139ffa9cf30b4f07cc9902896ab11294ac68d7`, which archives the complete
  p34r02 log.  The three pre-existing local ledger edits were preserved; the
  remote update did not overlap them.
- Evidence correction: the previous checkpoint audited a truncated
  163-line artifact and remains the historical record of that limited audit.
  The complete 686-line artifact supersedes its `INCONCLUSIVE` verdict:
  `../../debug_logs/p34_p34r02_deepswe_full.raw.log`, SHA-256
  `6f1c446ad650acb1cf03c7bf9368c5dfbe78142689dbe6a358b11ab7c8097952`.
- Confirmed PASS before failure: clean data 4578 -> 1851; 256 devices on 64
  hosts; disjoint 128-device DP16xTP8 roles; trainer-side Qwen3-32B load;
  30.5 GiB/device on trainer-role devices; online W&B; and vLLM rollout-engine
  construction.
- Root cause: `_canonical_engine.env` supplied the legacy one-host default
  `CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3`.  P34 inherited it and rejected the
  healthy 128-device rollout mesh in `tpu_runner.py::_init_mesh()` with an
  exact mismatch.  Classification is `FAILED_ROLLOUT_MESH_ADMISSION`; no
  rollout, trajectory, backward or optimizer transaction occurred.
- Repair: the P34 base profile and renderer now explicitly clear the
  allocation-specific ID assertion.  P34 preflight rejects any nonempty
  override, so the same leak fails before resource-intensive target work.
  Physical 4x8x8 inventory and host-complete role placement remain
  fail-closed.
- Commands/results: targeted renderer/environment tests passed 19 cases;
  `P34_STATIC_PASS suites=10`; `P39_DEEPSWE_PILOT_CPU_PASS`;
  `P43_DEEPSWE_DEBUG_CPU_PASS`; `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`; and
  `P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2
  contract_cases=5 scheduler_cases=1 overlay=qwen32b`.
- Boundary: no target retry, cloud action, credential change, commit or push
  occurred.  The local repair must be explicitly approved for publication
  before a fresh full-run manifest is rendered from the read-back SHA.

## 2026-08-12 UTC — P39.4 mesh-admission repair publication

- Type: publication checkpoint.
- Decision: the operator explicitly approved commit and push.
- Result: repair commit
  `562f55b077bdadbcfa160177715b0d8ca903f457` was pushed to
  `yuxzhang/canon-zero-tim` and read back with the same 40-character SHA.
- Boundary: publication did not launch a JobSet or target retry.  The next
  operator must pull the branch into a clean worktree, record its exact HEAD
  and render a new manifest; the failed p34r02 manifest must not be reused.

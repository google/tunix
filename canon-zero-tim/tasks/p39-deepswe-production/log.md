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

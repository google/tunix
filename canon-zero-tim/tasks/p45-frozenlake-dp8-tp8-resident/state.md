# State

- Status: active
- Objective: Add an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full/eval recipe with device-resident optimizer state while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, and shell gates pass; rendered full/eval manifests attest DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, online W&B, warning-only alignment, and hard numerical/transaction failures; the operator handoff contains an exact render/apply/return procedure.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked and published in implementation commit `fae4e67f`
- Current phase: P45.3a/P45.3b — GCS checkpoint/resume plus host-memory hardening
- Last verified fact: `p45r5` from source `42139ffa` sustained 47 committed
  DP8xTP8 device-resident updates before the `jax-tpu` container was reported
  OOMKilled at its 200G host-memory limit; TPU HBM remained stable. Separately,
  on pinned image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`,
  the checkpoint-, memory-, and grouped-report-aware exact-image gate passed 102
  workload/renderer tests, 37 alignment tests, merged profile admission,
  seven TP8 projection sites, and canonical forward/VJP. Focused Orbax
  option/manager tests passed 29/29. The P45 renderer now alone produces a
  350G `jax-tpu` limit and the learner emits cgroup/RSS evidence around eval
  and committed-step GC. The P45 profile enables only the P32-grouped
  `CANON_P28_BATCHED_REPORT=1` optimization; the two unported grouped flags
  remain absent. Checkpoint/resume was published in `2cb5112f`; host-memory
  and grouped-report hardening was published in `fbfb4bd8`. The P45r6 G6
  checkpoint-contract collision has now been fixed locally with a narrow,
  default-off trainer admission; the pinned-image P45 gate passed 103 tests
  plus both targeted G6 checkpoint regressions. Target proof remains pending.
- Next action: publish the local G6/checkpoint fix, then render one `new` attempt
  from that immutable source with a stable campaign tag and run
  through committed step 10/11, measure host-memory behavior, and verify exactly
  one durable checkpoint. A separate `resume` render using the same immutable
  source/tag is admitted only after a step-10 checkpoint exists.
- Blockers: no remaining local code blocker. `p45r6` source `9a834574` is
  permanently invalid for checkpoint-enabled training; the local correction
  must be published and then proven through step 10/11 on 64 chips.
- Key artifacts: `../../debug_logs/p45_p45r6_checkpoint_contract_error.raw.log`;
  `../../debug_logs/p45_p45r5_frozenlake_resident.raw.log`;
  `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `HANDOFF.md`; `plan.md`;
  `phases/p45-2b-qwen8b-tp8-overlay.md`;
  `phases/p45-3a-gcs-checkpoint-resume.md`;
  `phases/p45-3b-host-memory-hardening.md`; `phases/p45-3-target-run.md`
- Updated: 2026-08-15 UTC

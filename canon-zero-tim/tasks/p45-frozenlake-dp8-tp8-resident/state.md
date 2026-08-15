# State

- Status: active
- Objective: Add an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full/eval recipe with device-resident optimizer state while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, and shell gates pass; rendered full/eval manifests attest DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, online W&B, warning-only alignment, and hard numerical/transaction failures; the operator handoff contains an exact render/apply/return procedure.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked and published in implementation commit `fae4e67f`
- Current phase: P45.3a — GCS checkpoint/resume admission
- Last verified fact: `p45r5` from source `42139ffa` sustained 47 committed
  DP8xTP8 device-resident updates before the `jax-tpu` container was reported
  OOMKilled at its 200G host-memory limit; TPU HBM remained stable. Separately,
  on pinned image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`,
  the checkpoint-aware exact-image gate passed 97 workload/renderer tests, 37
  alignment tests, merged profile admission, seven TP8 projection sites, and
  canonical forward/VJP. Focused Orbax option/manager tests passed 29/29.
- Next action: render one `new` attempt with a stable campaign tag and run
  through committed step 10/11, measure host-memory behavior, and verify exactly
  one durable checkpoint. A separate `resume` render using the same immutable
  source/tag is admitted only after a step-10 checkpoint exists.
- Blockers: Pathways checkpoint memory/latency/durability/restore remain
  unverified, and the host-memory growth mechanism behind the long-run OOM has
  not yet been isolated with an RSS/cgroup timeline.
- Key artifacts: `../../debug_logs/p45_p45r5_frozenlake_resident.raw.log`;
  `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `HANDOFF.md`; `plan.md`;
  `phases/p45-2b-qwen8b-tp8-overlay.md`;
  `phases/p45-3a-gcs-checkpoint-resume.md`; `phases/p45-3-target-run.md`;
  `../p41-optimizer-residency/phases/p41-4-frozenlake-capacity.md`
- Updated: 2026-08-15 UTC

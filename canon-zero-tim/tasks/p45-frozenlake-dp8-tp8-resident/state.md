# State

- Status: active
- Objective: Add an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full/eval recipe with device-resident optimizer state while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, and shell gates pass; rendered full/eval manifests attest DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, online W&B, warning-only alignment, and hard numerical/transaction failures; the operator handoff contains an exact render/apply/return procedure.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked after publication; currently workspace-local
- Current phase: P45.3 — 64-chip target run
- Last verified fact: on the final source, the pinned-image P45 gate passes 77 workload/renderer/classifier tests plus 29 alignment tests; merged `00_env.sh` admission resolves DP8xTP8, 32 local trajectories, global M2048, evaluation on, and device-resident optimizer state. The complete adjacent P33/P38 CPU gate also passes on that same final source.
- Next action: follow `HANDOFF.md`, server-dry-run both generated manifests, then apply exactly one full or full-eval JobSet and capture the first committed update's HBM and optimizer timing evidence.
- Blockers: no 64-chip target evidence exists for FrozenLake DP8xTP8 or resident multi-update stability.
- Key artifacts: `HANDOFF.md`; `plan.md`; `phases/p45-1-contract-and-renderer.md`; `phases/p45-2-local-admission.md`; `phases/p45-3-target-run.md`; `../p41-optimizer-residency/phases/p41-4-frozenlake-capacity.md`
- Updated: 2026-08-12 UTC

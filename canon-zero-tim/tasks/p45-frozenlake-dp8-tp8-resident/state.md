# State

- Status: active
- Objective: Add an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full/eval recipe with device-resident optimizer state while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, and shell gates pass; rendered full/eval manifests attest DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, online W&B, warning-only alignment, and hard numerical/transaction failures; the operator handoff contains an exact render/apply/return procedure.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked and published in implementation commit `fae4e67f`
- Current phase: P45.3 — 64-chip target run
- Last verified fact: `p45r5` from source `42139ffa` on 64 TPU (`DP8xTP8`, resident optimizer) passed model loading, compilation, rollout, and ran continuously for ~60 hours (2.5 days), completing 47 training steps and 1535 alignment checks (1535/1535 PASS). The job terminated at `Sat, 15 Aug 2026 06:03:58 UTC` due to Linux kernel host RAM cgroup exhaustion (`Exit Code: 137 (OOMKilled)` on `jax-tpu`), having reached the 200GiB limit. TPU HBM remained stable throughout.
- Next action: for future long-horizon multi-day runs, increase head pod memory limit to 350GiB+ and introduce periodic Python GC in the training step loop.
- Blockers: none; DP8xTP8 resident capacity and numerical stability are proven across 47 training steps.
- Key artifacts: `../../debug_logs/p45_p45r5_frozenlake_resident.raw.log`; `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `HANDOFF.md`; `plan.md`; `phases/p45-2b-qwen8b-tp8-overlay.md`; `phases/p45-3-target-run.md`
- Updated: 2026-08-15 UTC

# State

- Status: active
- Objective: Add an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full/eval recipe with device-resident optimizer state while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, and shell gates pass; rendered full/eval manifests attest DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, online W&B, warning-only alignment, and hard numerical/transaction failures; the operator handoff contains an exact render/apply/return procedure.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked and published in implementation commit `fae4e67f`
- Current phase: P45.3 — 64-chip target run
- Last verified fact: the isolated `qwen8b_tp8` overlay passed the exact-image gate on image ID `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`: 29 installed files matched manifests; the full `linear_p22xk` import passed at TP8; seven projection shapes, no-padding contracts, TP4 rejection, and Pallas-interpret forward/VJP all passed. P45 reported 83 workload/render tests plus 31 alignment tests passing, and the full adjacent P33 CPU gate passed.
- Next action: fetch the published branch head, use the exact P45 runbook to render/dry-run both variants, and launch exactly one new `p45r4`-or-later 64-chip attempt. The first committed update is the first real resident-HBM test.
- Blockers: no 64-chip attempt has yet passed model loading with `qwen8b_tp8`; resident HBM capacity, evaluation, and multi-update stability remain unverified.
- Key artifacts: `../../debug_logs/p45_p45r3_frozenlake_resident.raw.log`; `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `HANDOFF.md`; `plan.md`; `phases/p45-2b-qwen8b-tp8-overlay.md`; `phases/p45-3-target-run.md`; `../p41-optimizer-residency/phases/p41-4-frozenlake-capacity.md`
- Updated: 2026-08-12 UTC

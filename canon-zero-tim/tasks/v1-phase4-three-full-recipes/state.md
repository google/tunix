# State

- Status: active; P4.10 source freeze is host/full-image green and publication-approved, while the Attempt 9 TP8 forward regression remains target-unresolved
- Objective: recover FrozenLake DP8xTP8 strict Zero-TIM before any new full train by testing whether process-wide P66 VMA mutations leaked into ordinary serving.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: isolated publication worktree `/mnt/disks/tunix-data/worktrees/v1_fl_tp8_ab_diag_0826`, branch `local/v1-fl-tp8-ab-diag-0826`, with runtime/source CL `47219e0729d5bbdbe43bc407e19aa056c80f02c3` on fetched base `ff0acaaa`. The user authorized the P4.10 publication stack on 2026-08-26; no JobSet launch occurred.
- Current phase: V1.P4.10 publication and target handoff for a zero-backward P45 matched pair.
- Last verified fact: Attempt 9 P45 failed A-B/B-C `1755/0` bytes across 46,879 actions; M15 failed `93/0` across 124,308. Phase4 82/82, P57 146/146, P59 37/37, P66 16/16, APC 31/31, flags 385/385, both P45 arm env resolutions, shim manifest 37/37, scoped DP2×TP4/TP8 installed-shim image gates, and complete exact-image regression pass. The final terminal is `V1_HP_EXACT_IMAGE_PASS ... p59_checked_vma_real_shim=4 ... manifests=3`. `HANDOFF.md` now contains the complete operator procedure and per-arm/paired return contract.
- Next action: after exact remote read-back, render fresh P45 p66-off and serving-scope IDs for user launch, then classify the paired target artifacts under the handoff matrix.
- Blockers: real DP8xTP8 target is unrun. Dirty-source YAML is not launchable. Optimizer/convergence/performance claims are out of scope because the carrier exits before backward.
- Key artifacts: `phases/v1-p4-10-frozenlake-tp8-ab-recovery.md`; `scripts/prepare_fl_tp8_ab_wave.sh`; Attempt-9 evidence directory; `RUNBOOK.md`.
- Updated: 2026-08-26T05:42:52Z

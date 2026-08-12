# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.11 — real one-host Qwen3-4B DeepSWE integration](phases/p44-11-onehost-deepswe-integration.md)
- Last verified fact: P44.10/P44.11 implementation commit `29cea119259f1f7fe583a3e3dd1cb190acc0bf63` was created from operator baseline `d8184123448d0add72b72f09d0a6faf5d326c26e` without touching main and preserves its P38 capture/precheck hardening. On one direct-attached four-device v5p host, the development diff loaded Qwen3-4B, selected one reviewed cached R2E task, executed two real Docker trajectories and tool actions, persisted trajectories/solve metrics, completed trainer forward and a real backward invocation, proved the optimizer state is device-resident, and preserved model/reference/optimizer/accumulator state with zero commits. Rollout integration is PASS; backward is honestly `INCONCLUSIVE_NO_SIGNAL` because both rewards/advantages and the gradient norm were zero. P44 40-test and exact-image gates plus P43/P39/P34 regressions pass.
- Next action: Commit this publication metadata, push only to `yuxzhang/canon-zero-tim`, read back its exact SHA, and have the launch agent repeat the clean-source one-host rollout smoke before starting fresh 64- or 256-device `rollout-only` attempt `p44r06` or later.
- Blockers: the bounded Qwen3-4B one-host batch did not complete an episode or produce nonzero learning signal, so local one-update is not promoted. TP8, Pathways role separation, and 64/256 target promotion still require the corresponding remote allocation.
- Key artifacts: `phases/p44-10-r05-matmul-padding.md`, `phases/p44-11-onehost-deepswe-integration.md`, `tests/p44_deepswe_qwen4b_parity/run_onehost_deepswe_v5p.sh`, `/mnt/disks/tunix-data/deepswe-onehost-evidence/20260812-p44-local-dev/`, `plan.md`, `log.md`
- Updated: 2026-08-12T05:05:00Z

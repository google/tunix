# Log

## 2026-08-11T23:12:10Z — P43.0: bind an isolated canonical-base worktree

- Type: decision
- Fact: `git fetch origin yuxzhang/canon-zero-tim` resolved the current remote-tracking base to `39e18f7ddee8f5c7ab7dfcd269e83d7785a684c2`.
- Action: Created isolated worktree `/home/yuxuan/code_rl_repro/sequence_packing/p43_deepswe_64_debug` on new branch `codex/p43-deepswe-64-debug`, tracking `origin/yuxzhang/canon-zero-tim`.
- Command: `git fetch origin yuxzhang/canon-zero-tim`; `git worktree add -b codex/p43-deepswe-64-debug /home/yuxuan/code_rl_repro/sequence_packing/p43_deepswe_64_debug origin/yuxzhang/canon-zero-tim`
- Result: PASS; the new worktree started clean at the exact fetched SHA. Existing dirty P38/P39/P42 worktrees were not modified.
- Files/artifacts: `state.md`; `plan.md`; `log.md`
- Rollback: Remove only the isolated P43 worktree and local P43 branch after preserving any desired patch; do not alter the shared worktrees.
- Next: Implement the smallest separate Qwen3-8B DP4xTP8 debug contract and its CPU negative controls.

## 2026-08-11T23:31:00Z — P43.1: freeze the debug and evidence contract

- Type: decision
- Fact: The current R2E-Gym trajectory path retains `trajectory_reward` and
  terminal status but does not return an independent solved/verdict field.
- Decision: Use the explicit diagnostic solve definition
  `r2egym_final_reward_eq_1`; require terminal completion and raw reward
  exactly `1.0`; preserve/count all non-binary rewards without treating them
  as solved.
- Decision: Add P43 as a mutually exclusive active DeepSWE workload so the
  proven 4x4x4 host-complete split and replicated-parameter DP4xTP8 path can
  be reused without changing P34/P39 defaults.
- Decision: Write one atomic compressed trajectory batch plus fsync'd metrics
  and manifest before any optimizer update, and add a real `rollout-only`
  stage that exits before backward.
- Files/artifacts: `phases/p43-1-debug-contract.md`
- Result: PASS; detailed exit gates and remote stage order are frozen before
  implementation.
- Next: Implement the P43 workload, writer, integration hook, renderer, and
  CPU negative controls.

## 2026-08-11T23:33:19Z — P43.1-P43.3: implement and gate the debug ladder

- Type: validation (pre-commit candidate tree)
- Action: Added the mutually exclusive `p43-64chip-debug` workload, Qwen3-8B
  DP4xTP8 profile, three-stage renderer, synchronous trajectory/metric writer,
  rollout-only boundary, update classifier, CPU controls, exact-image gate,
  and operator runbook.
- Fact: The datasets loader no longer passes the removed
  `trust_remote_code` keyword. The tokenizer keeps its separate Transformers
  option.
- Fact: P43 does not opt into the retired fresh-client Step-65 device probe;
  the real training process remains the authoritative exact 64-device,
  host-complete role-split check.
- Command: `bash canon-zero-tim/tests/p43_deepswe_debug/run_cpu.sh`
- Result: PASS; 21 tests and terminal marker
  `P43_DEEPSWE_DEBUG_CPU_PASS`.
- Command: `bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh`
- Result: PASS; 15 adjacent P39 tests.
- Command: `bash canon-zero-tim/tests/p34_deepswe/run_static.sh`
- Result: PASS; 10 adjacent P34 suites.
- Command: `bash canon-zero-tim/tests/p43_deepswe_debug/run_exact_image.sh
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
- Result: PASS; the qwen8b overlay matched all 29 manifest files, the P43
  tests passed in-image, and `P43_EXACT_IMAGE_CPU_PASS overlay=qwen8b` printed.
- Command: Rendered `rollout-only`, `one-update`, and `three-update` with the
  P43 CLI using fake syntactically valid immutable inputs.
- Result: PASS; all three emitted `P43_DEBUG_JOBSET_RENDER_PASS` and produced
  distinct manifests without hand edits.
- Files/artifacts: `cluster/P43_DEEPSWE_64CHIP_DEBUG_RUNBOOK.md`;
  `HANDOFF.md`; P43 profile/renderer/tests; `tunix/rl/deepswe_debug.py`.
- Next: Re-fetch the publication branch, integrate safely, commit, rerun
  publication-critical gates, push, and verify the exact remote SHA.

## 2026-08-11T23:34:00Z — P43.3: publication branch moved

- Type: external state
- Fact: The second required fetch advanced
  `origin/yuxzhang/canon-zero-tim` from
  `39e18f7ddee8f5c7ab7dfcd269e83d7785a684c2` to
  `340b0e364f374fde8798d8f62331e6bc33e0e58a` by five commits.
- Fact: Remote changes overlap `cluster/steps/00_env.sh` and `90_run.sh`; a
  blind force push is forbidden.
- Decision: Commit the isolated P43 patch, rebase it onto the fetched remote
  tip, resolve only the overlapping P43 integration hunks, then rerun gates
  from the rebased exact SHA.
- Next: Create the local P43 commit and rebase onto `340b0e36`.

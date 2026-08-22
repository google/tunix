# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training curves and final isolated evaluations.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and final evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: all six local cells now render at exactly 200 updates and pass the real resolved `00_env.sh` preflight. The P57 host suite passed 105/105, including rejection of the historical P45 450-step horizon; flag audit passed 320/320; syntax and diff gates passed; and the pinned-image exact-image gate emitted the IS stock-runtime positive, zero rejection, fixed-lm-head/TP8 forward+VJP receipts, `P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed`, and `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0.
- Next action: present the uncommitted 200-step six-cell diff for review. Commit/push, rebase onto the advanced remote tip, and every target launch require separate approval; no target has tested this local extension.
- Blockers: publication and target launch approval; the remote delivery tip has advanced beyond this dirty worktree base, so any approved publication must rebase and rerun focused gates first. No TPU run has tested the local three-arm extension.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-22 UTC

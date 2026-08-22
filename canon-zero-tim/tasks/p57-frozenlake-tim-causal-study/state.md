# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training curves and final isolated evaluations.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and final evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: all six construction cells render at exactly 200 updates and pass the real resolved `00_env.sh` preflight. The P57 host suite passed 105/105, flag audit passed 320/320, and the pinned-image gate passed stock `is`, zero rejection, fixed-lm-head/TP8 forward+VJP, stock observer, and terminal P45 exact-image receipts. After the user selected the first target queue as P45/M15 `mismatch` plus P45/M15 `is`, both two-workload wrapper commands passed and produced exactly those four 200-step manifests; no `zero` manifest was rendered.
- Next action: from the immutable published documentation tip, render the `native` and `is` waves, record all four manifest hashes, and request separate approval before applying the four JobSets. Do not render or launch the deferred `zero` wave as part of this queue.
- Blockers: target launch approval and four independent 64-chip slices if all jobs are to run concurrently. No TPU target has tested the three-arm extension.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-22 UTC

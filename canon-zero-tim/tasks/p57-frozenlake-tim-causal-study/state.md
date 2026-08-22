# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training and isolated 50-step milestone curves.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and ten registered isolated evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: the user retained the 450-update horizon and preregistered isolated greedy evaluation every 50 updates (`0,50,...,450`). Locally, training remains uninterrupted with in-process evaluation off. Checkpointing saves every 10, retains one rolling recovery point plus every 50-step P57 milestone, and the evaluator explicitly restores the requested step instead of latest. The full host suite passed 119/119; native and IS train waves passed manifest preflight; both 20-manifest eval schedules passed; the pinned production-image gate ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. No LR, batch, loss, sampler, reducer, optimizer, or numerical-kernel semantics changed.
- Next action: review the local 450+milestone-eval diff. After separate commit/push approval, rerender both train waves and both 20-manifest eval schedules from the immutable published SHA. Run four step-0 evals before training; after separate approval run the four uninterrupted 450-step trains; only after durable completion run positive milestone evals. Do not launch deferred `zero` yet.
- Blockers: review and commit/push approval; explicit acceptance of the several-terabyte four-arm GCS milestone envelope; then separate step-0-eval, training, and positive-eval launch approvals. All 450-update target paths are `TARGET NOT RUN`; earlier 200-update jobs remain immutable evidence and are not resumable into the new campaign.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-22 UTC

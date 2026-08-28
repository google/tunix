# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: Wave 10 P45 full training (`canon-p57-fl-zero-f45w10-96544812`, 64 TPU) ran through Step 63/300 (solve rate 44.5%) before encountering a worker-to-worker client pipe timeout (`DEADLINE_EXCEEDED` between worker 13 and worker 2), triggering fail-closed termination. Error logs archived under `evidence/f45w10_worker_pipe_timeout/`. M15 Full Wave 10 (`canon-p57-fl-zero-m15-mw10-96544812`, 64 TPU) is unaffected and training normally.
- Next action: preserve f45w10 error evidence; monitor M15 Wave 10 training; relaunch P45 Wave 10 upon authorization.
- Blockers: none.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md); [Wave 10 Incident](evidence/f45w10_worker_pipe_timeout/incident_summary.md)
- Updated: 2026-08-28T04:23:00Z

# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: P45 native attempt `canon-p57-fl-mism-n45j-2a89eef3` completed its first real optimizer transaction, then the step-0 evaluation receipt read the deliberately deferred `rl_cluster.global_steps=0` instead of the committed `actor_trainer.train_steps=1` and false-red before weight sync. The attempt is `INCONCLUSIVE`, not a numerical failure. The local repair validates both lifecycle counters. The active P57 checkpoint contract is final-only (`interval=300`, `LatestN(1)`), while historical P45/M15-selection remains interval 10. P57 passes 136/136 and V1 passes 12/12; target rerun is not yet performed.
- Next action: review the local receipt/cadence repair. After separate commit/push approval, render all four fresh P45/M15 native-no-IS/token-IS JobSets from one immutable pushed SHA and request launch approval. Do not resume or reuse any earlier attempt.
- Blockers: code review and commit/push approval, then separate cluster launch approval. The repaired evaluation receipt and final-only target checkpoint are `TARGET NOT RUN`; old partial attempts are immutable failed evidence and are not resume sources.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-23T19:05:00Z

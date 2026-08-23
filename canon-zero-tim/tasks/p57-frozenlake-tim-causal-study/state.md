# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: the superseding 300-update contract renders all six P45/M15 x native/IS/zero cells with rollout-only evaluation enabled. P57 CPU gates pass 126/126, including seed-43 and dataset-mutation negatives; all paired commands pin seed 42, runtime pins vLLM global seed 0, and P45/M15 primary train/eval row identities are registered by exact SHA. Native/IS/zero two-workload wave renders all pass resolved-env preflight, flag audit passes 322/322, and the pinned exact-image gate ends in `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. No target TPU run has yet certified the final step-300 evaluation path or cross-launch stochastic reproducibility.
- Next action: review the uncommitted diff. After separate commit/push approval, render fresh native and IS two-job waves from the immutable pushed SHA, request launch approval, and run the four 300-update jobs. Do not render separate milestone evaluators and do not launch deferred `zero` yet.
- Blockers: code review and commit/push approval, then separate cluster launch approval. The new 300-update/evaluation path is `TARGET NOT RUN`; earlier 200/450 attempts remain immutable evidence and are not resumed into this campaign.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-23 UTC

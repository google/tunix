# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: Wave 10 P45 (`canon-p57-fl-zero-f45w10-96544812`) is analysis-grade `INCONCLUSIVE_INFRA_SOURCE_MISSING`. The committed incident summary reports progress through Step 63/300 and 44.5% solve; all 14 retained non-source worker logs independently show that worker 2 stopped sending at about 03:32:41 and the Pathways 10-second pipe deadline caused fail-closed teardown at about 03:32:51. The evidence package does not contain worker-2 stdout/stderr, Pod termination reason/exit code, events, or a head log, so network, process crash/OOM, eviction/preemption, and node failure remain unresolved alternatives. M15 Wave 10 was reported unaffected and continuing.
- Next action: validate the new external JobSet log collector on the host, then start one collector per separately authorized fresh P45/M15 JobSet before apply. The collector must preserve worker indices 0..15, head/sidecar logs, Pod termination state, events, and node conditions to a live GCS mirror and sealed SHA package.
- Blockers: exact worker-2 termination evidence from f45w10 is irrecoverably missing. The new collector is host-only construction until exercised by a real attempt. Increasing the 10-second Pathways deadline is not an admitted repair without evidence that a worker merely paused and later recovered.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md); [Wave 10 Incident](evidence/f45w10_worker_pipe_timeout/incident_summary.md)
- Updated: 2026-08-28T05:23:19Z

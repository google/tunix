# State

- Status: G4 passed; source CL `ec9884e9` prepared for publication
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1c — Perf v2 step-boundary isolation](phases/p57-1c-perf-v2-step-boundary.md)
- Last verified fact: approved one-host v5p G4 `p57c_g4_cb38cf67_r7` completed 3/3 real AdamW transactions, 12/12 strict alignment PASS with zero differing bytes, finite nonzero gradients, and a readable beta-zero semantic Perfetto. It crossed the former Step-1 rollout underflow boundary without a tracer red. Steady Steps 1/2 were 36.93s and 35.98s. P57 CPU is 172/172; the final pinned-image P45 gate emits `P57_PERF_V2_STEP_BOUNDARY_PASS` and `P45_EXACT_IMAGE_CPU_PASS`; V1 Phase4 is 90/90; flag audit is 395/395. No numerical flag or training math changed.
- Next action: verify the published two-CL stack, then obtain separate target-launch approval before fresh P45 and M15 full identities exercise G5.
- Blockers: production render and full P45/M15 launch remain outside the current authorization.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1c-perf-v2-step-boundary.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md); [Wave 15 Incident](evidence/f45w15_timeline_tracer_incident/INCIDENT_REPORT.md)
- Updated: 2026-08-28T12:45:00Z

# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: Wave 15 P45 (`canon-p57-fl-zero-f45w15-799a0bd1`, 64 TPU v5p) completed Step 0 with bitwise exact pre-alignment (`S_decode_vs_S_prefill: 0 B`, `S_prefill_vs_T_old: 0 B`, `verdict: PASS`) and optimizer commit 1 (`stable_norm=0.5510`). During Step 1 Rollout, concurrent trajectory workers encountered an empty timeline span stack in `tunix/perf/experimental/tracer.py:346` / `timeline.py:236`, raising `ValueError: host-139531592390336: no more spans to end.` All 19 logs (3 head logs + 16 worker logs) were fully mirrored and verified in GCS `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57/f45w15-799a0bd1/` and sealed locally under `evidence/f45w15_timeline_tracer_incident/`. M15 Wave 15 continues running on Step 0.
- Next action: fix the async/concurrent span stack handling in `tunix/perf/experimental/tracer.py` before relaunching P45.
- Blockers: experimental tracer span underflow defect under concurrent rollout.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md); [Wave 15 Incident](evidence/f45w15_timeline_tracer_incident/INCIDENT_REPORT.md)
- Updated: 2026-08-28T10:35:00Z

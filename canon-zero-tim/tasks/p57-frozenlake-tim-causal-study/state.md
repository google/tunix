# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training curves and final isolated evaluations.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and final evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: the first four-job target attempt provisioned 256 chips but all four jobs failed before step 0 because the Python train validator still admitted only the historical `(mismatch,m15,selection)` discovery tuple. The attempts are `INCONCLUSIVE`. A local repair now admits a closed five-tuple stock matrix covering discovery plus P45/M15-main under `mismatch` and `is`; arbitrary combinations remain rejected. The P57 host suite passed 105/105. On the one-host v5p machine, the pinned production image emitted `P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval`, passed stock dependencies, Qwen3-8B TP8/fixed-lm-head forward+VJP, observer gates, and terminal `P45_EXACT_IMAGE_CPU_PASS`. The container intentionally had no `/dev/vfio`, so this is construction/runtime-contract evidence rather than target TPU evidence.
- Next action: review the uncommitted repair. Only after separate commit/push approval, render `native` and `is` from the new immutable repair SHA using fresh four-character run IDs and campaign root, record four manifest hashes, then request separate launch approval. Do not launch the deferred `zero` wave.
- Blockers: publication approval, then target launch approval and four independent 64-chip slices if all jobs are to run concurrently. The repaired validator has not yet been exercised by a target JobSet.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-22 UTC

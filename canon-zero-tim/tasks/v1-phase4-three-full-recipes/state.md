# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: release base is `238ca28cf6eb642429de66c0da58b68ea659309f`, including immutable attempt-2 error evidence. The repair is isolated as P59 q_proj, M15 APC-off, and evidence/ledger CLs; no target runtime has been published after attempt 2.
- Current phase: V1.P4.4 attempt-2 repair before direct full target reruns; host/static and r5 dependency-complete exact-image admission are green, while post-fix targets remain unrun.
- Last verified fact: attempt-2 GSM8K/P45 passed strict step-0 pre-alignment then stopped pre-optimizer because `n_shards=1` was misread as mesh TP1. M15 recorded a real APC-on `S_decode_vs_S_prefill` red (760 elements / 1389 bytes, max abs `0.998443603515625`) with `S_prefill_vs_T_old` exact, so APC is reverted for M15/main and remains on only for P45. The real q_proj one-layout-shard P59 carrier passes exact-image TP4/TP8 with serial/parallel exact gradients. FrozenLake APC-off now requires exactly one `enabled=0` runtime receipt and rejects missing, duplicate, and opposite-arm receipts. Host gates pass V1 19/19, P57 144/144, P59 31/31, APC 31/31, flags 366/366, syntax, manifest, and diff hygiene. Full r5 raw-log SHA is `90affa9db1ca8ba4df6d7334aa7897aa9bd77492d93fd1378753396ff531556e`.
- Next action: complete the authorized three-CL publication and exact remote readback, then render fresh full manifests and restart GSM8K first; gate P45/M15 on its first real optimizer commit exactly as already decided.
- Blockers: repaired target P59 TP4/TP8, P45 APC, fixed-head, strict alignment, optimizer, performance, and all full horizons remain unverified. M15 APC is target-VETOED and must remain off.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T05:43:21Z

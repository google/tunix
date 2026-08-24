# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: cleanly rebased onto pulled operator tip `0a68e1f7`, which adds the P60 GSM8K XProf carriers and trace summarizer after immutable Attempt-6 tip `f2dd9d90`. Three rollback-safe CLs are committed locally: P59 staged-spec runtime `26b8a36d`, uniform APC-off/auditable-cache carrier `ef481f02`, and the current evidence/ledger CL. Only the approved push and exact remote readback remain. Dependency-complete pinned-image and real-v5p DP2xTP2 mechanism gates are green for the pre-P60 Attempt-6 runtime.
- Current phase: V1.P4.4 admission green; publication and target execution pending.
- Last verified fact: exact-image raw SHA `8d8d776451615de58a749c0be0200d28107b86cc44504200afde4f5acffc712a` ends with `staged_spec_restore=2` and full V1 PASS. Real-v5p run `p59_rpa_a6restore_dp2tp2_20260824_2256utc` passes the replicated-leaf positive, wrong-placement negative, RPA VJP, and ordinary TP4 control in 32 seconds with zero optimizer commits; raw/driver hashes verify. After rebase, host gates pass V1 29/29, P57 144/144, P59 34/34, APC 31/31, P60 XProf 4/4, and flags 368/368.
- Next action: quiet-fetch to prove the operator tip still equals `0a68e1f7`, push the three approved CLs, and verify exact remote SHA readback. Preserve the claim boundary that inherited P60 runtime was not part of the earlier exact-image/TPU captures. After publication/readback and separate launch approval, render and apply fresh GSM8K/P45/M15 full JobSets in one wave.
- Blockers: repaired DP16×TP4/DP8×TP8 optimizer commits, target cache hit/JIT effect, TP8 fixed head, performance, and all full horizons remain unverified. M15 APC is target-VETOED; P45 is also production-disabled by user decision. P45/M15 Attempt-6 logs are non-terminal and receive no verdict.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T23:11:10Z

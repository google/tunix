# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: the attempt-1 repair is reconstructed as two functional CLs plus this registry/evidence/handoff CL on evidence base `5f3e8ff95075642b5e660af8d1219e1c98e71c72`. Commit and push were explicitly authorized on 2026-08-24; exact remote readback remains mandatory.
- Current phase: attempt-1 first-failure repair committed locally; host/static gates are green, while publication readback, post-fix pinned-image admission, and target reruns remain pending.
- Last verified fact: GSM8K `g64f` stopped pre-optimizer when a DP-only full-vocabulary cotangent `[256,151936]` reached a P59 TP-local fixed-head VJP expecting `[256,37984]`; FrozenLake P45 `f45g` stopped earlier in C-forward because Qwen3-8B/TP8 learner `M=2048` was not registered. Neither log contains a real alignment FAIL; GSM8K has 1 PASS and FrozenLake has no alignment verdict. The repair explicitly restores `P(data,model)` before the head VJP and registers M2048 only for 8B/TP8 with an M2048 receipt. Host gates pass V1 13/13, P57 139/139, P59 31/31, APC 31/31, fixed-head/receipt 27/27, and the other 94 executable P38 tests; one unrelated P38 renderer module remains host-INCONCLUSIVE because `metrax` is absent.
- Next action: push the authorized linear stack and verify exact remote readback. Then obtain separate approval and rerun the pinned-image V1 gate so the modified installed shim and production-style DP-only cotangent test execute. Do not render or launch until it passes; after admission, GSM8K remains first, followed only after its full postflight by P45 and M15.
- Blockers: dependency-complete exact-image execution and each 64-chip launch remain separate approval boundaries; post-fix real P59 TP4/TP8, APC, fixed-head, strict alignment, optimizer, and performance are unverified.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T03:15:00Z

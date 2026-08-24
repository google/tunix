# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: the three-CL attempt-1 repair was published and exactly read back at `dfec27378bfdd9b73b7bf8f7930bacaa685b3a16`. The follow-up contract, exact-image carrier, and evidence CLs are intentionally dirty pending their authorized commit/push.
- Current phase: V1.P4.4 source publication before full target reruns; host/static and post-fix exact-image admission are green, while every target remains unrun.
- Last verified fact: both FrozenLake TIM and V1-HP receipt paths select global M2048. P59-local receipt admission requires exact M4096/M256/DP16 or M2048/M256/DP8, one local chunk, and `all_gather_rank_order_f32_barrier`; full postflight requires exact recipe profile and head global/local vocabulary shape. Host gates pass V1 17/17, P57 144/144, P59 31/31, APC 31/31, fixed-head/receipt 32/32, flags 366/366, syntax, and diff hygiene. Fresh pinned r3 passes the installed TP4/TP8 shim terminal and the complete V1 terminal; raw log SHA is `7ef23c9b7f4997a1855a16e99e348e4c981a1f80f9614cc95be1703771338264`. Failed r1/r2 test carriers remain immutable evidence and are not numerical reds.
- Next action: commit the isolated three-CL follow-up stack, re-fetch, push, and exactly read back the remote SHA. Do not render from the dirty tree. After publication, GSM8K is first; P45/M15 remain gated on at least one real GSM8K optimizer commit and the prescribed postflight/order.
- Blockers: dependency-complete exact-image execution and each 64-chip launch remain separate approval boundaries; post-fix real P59 TP4/TP8, APC, fixed-head, strict alignment, optimizer, and performance are unverified.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T03:42:00Z

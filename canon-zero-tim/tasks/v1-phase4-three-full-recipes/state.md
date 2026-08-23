# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: tracked; the P58.8 repair is serialized as four functional commits plus one audit-only release-gate commit on `ccbcf572`, with publication authorized and exact remote readback pending at this checkpoint.
- Current phase: source repair complete after first target bootstrap logs; publication and fresh target reruns pending.
- Last verified fact: incoming GSM8K stopped on the P59 TP4 nested-mesh boundary and incoming FrozenLake stopped on P57 W&B admission before P59. The latest `ccbcf572` P58.8 tree passes post-barrier TP4/TP8 installed-shim gates plus complete P58/V1 pinned-image terminals (`p59_real_shim=4 p57_wandb=1`); host V1/P57/P59/APC are 12/12, 136/136, 30/30, and 31/31, with flags 366/366. These are source admission receipts, not target passes.
- Next action: complete the approved push and exact remote readback, then render fresh run IDs from that SHA. GSM8K remains first, followed only after its full postflight by P45 and M15.
- Blockers: each 64-chip launch remains a separate approval boundary; real P59 TP4/TP8, APC, fixed-head, strict alignment, optimizer, and performance remain unverified until fresh target runs.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-23T09:25:30Z

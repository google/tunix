# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: pulled operator tip `7e9b31cb` includes all three immutable Attempt-4 logs. Runtime/manifest/carrier/classifier CL `5bd90bff` is committed locally; the durable exact-image evidence and ledger are the following local CL. No post-fix target runtime has been launched.
- Current phase: V1.P4.4 Attempt-4 P59 local gate/up layout repair before publication and one concurrent three-full launch wave.
- Last verified fact: Attempt-4 `g64p`, `f45p`, and `m15p` all have strict step-0 pre-alignment PASS, zero FAIL, and zero optimizer commits. Published q_proj, RPA, and M15-width repairs took effect. All three then stopped at the same local `gate_proj` width comparison: physical 1536 versus global 6144/12288 under legitimate `config.n_shards=1`. The local repair uses live TP only for gate/up's TP-local last axis, retains q/k/v semantics, validates `site.n_local`, and adds exact receipts. P59 host is 34/34; V1 host is 23/23. Focused TP4/TP8 installed-shim and complete V1 pinned-image gates exit zero with `p59_fused_linear=2`, 2x36/36 manifests, and zero commits. Durable exact-image raw/receipt SHAs are `9d50ec495c189a77dfdab92b8496580a58a55d101ed03cd2b977728a69ef5001` / `62995bb94a849602eeb2390d8e83b75bb1bf6b082d7044d47912d8b9e694b205`.
- Next action: commit the evidence ledger, confirm the fetched operator tip remains unchanged, push both CLs, and read back the exact remote SHA. After separate launch approval, render and apply fresh GSM8K/P45/M15 full JobSets in one wave. Each independently requires its own first optimizer commit and strict receipts; no recipe gates the launch of another.
- Blockers: repaired real DP16×TP4 and DP8×TP8 optimizer commits, P45 APC at target, TP8 fixed-head at target, performance, and all full horizons remain unverified. M15 APC is target-VETOED and must remain off.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T09:35:31Z

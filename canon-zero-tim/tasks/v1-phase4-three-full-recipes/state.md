# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: release base is `65606a985aa869f09a3bd3a39a3c9268a432aa71`, including all three immutable Attempt-3 error logs. The local repair/ledger stack is `248c5f9d`, `0ab5ae76`, `aa84c147`, and `f0af2d9b`; the exact-image evidence ledger is the final local CL. No target runtime has been launched from this stack.
- Current phase: V1.P4.4 publication before one concurrent three-full launch wave. Host/static and dependency-complete exact-image admission are green; post-fix targets remain unrun.
- Last verified fact: all three Attempt-3 step-0 pre-alignments are strict PASS with 0 FAIL and 0 optimizer commits. `g64m` and `f45m` prove the same already-local RPA K/V expansion bug at TP4 and TP8; patch 25 selects the local boundary only under exact P59 manual DP×TP context. `m15m` stopped earlier because its signed 4096/8192 buffers hit the stale P45 4096/2048 gate; `aa84c147` admits only the registered M15 selection/main tuples and retains partial/foreign negatives. The real four-chip v5p mechanism gate passed P59 DP2×TP2 RPA forward+VJP2, wrong-cache negative, and ordinary DP1×TP4 global GQA with zero commits. Final host gates pass V1 21/21, P57 144/144, P59 34/34, APC 31/31, flags 366/366. P58 and V1 exact-image scripts both exited zero on tested tree `24675392adee620ab36b87f9a0c4f7e8111f4839`; their terminals include `p59_rpa=2` and `m15_token=1`.
- Next action: commit the evidence-only ledger, fetch the operator branch, push normally if the base remains unchanged, and read back the exact remote SHA. After separate launch approval, render and apply fresh GSM8K/P45/M15 full JobSets in one wave. Each independently requires its own first optimizer commit and strict receipts; no recipe gates the launch of another.
- Blockers: repaired real DP16×TP4 and DP8×TP8 optimizer commits, P45 APC at target, TP8 fixed-head at target, performance, and all full horizons remain unverified. M15 APC is target-VETOED and must remain off.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T07:39:21Z

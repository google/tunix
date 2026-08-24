# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: release base is `65606a985aa869f09a3bd3a39a3c9268a432aa71`, including all three immutable Attempt-3 error logs. Local CLs are `248c5f9d` (RPA repair), `0ab5ae76` (initial ledger), and `aa84c147` (M15 token contract); this updated ledger follows. The stack is not pushed because its new exact-image gate is separately approval-bound and unrun. No target runtime has been launched from it.
- Current phase: V1.P4.4 Attempt-3 repairs before one concurrent three-full launch wave. Host/static admission is green; the separately approved dependency-complete pinned-image gate and targets remain unrun.
- Last verified fact: all three Attempt-3 step-0 pre-alignments are strict PASS with 0 FAIL and 0 optimizer commits. `g64m` and `f45m` prove the same already-local RPA K/V expansion bug at TP4 and TP8; patch 25 selects the local boundary only under exact P59 manual DP×TP context. `m15m` stopped earlier because its signed 4096/8192 buffers hit the stale P45 4096/2048 gate; `aa84c147` admits only the registered M15 selection/main tuples and retains partial/foreign negatives. The real four-chip v5p gate passed P59 DP2×TP2 RPA forward+VJP2, wrong-cache negative, and ordinary DP1×TP4 global GQA in 32 seconds with finite gradients and zero commits. Final host gates pass V1 21/21, P57 144/144, P59 34/34, APC 31/31, flags 366/366; the dependency-complete exact-image execution remains pending.
- Next action: obtain separate approval for the pinned-image gate and require both `p59_rpa=2` and `m15_token=1`; only its green result permits push. After exact remote readback and separate launch approval, render and apply fresh GSM8K/P45/M15 full JobSets in one wave. Each independently requires its own first optimizer commit and strict receipts; no recipe gates the launch of another.
- Blockers: dependency-complete installed-attention DP2×TP4/TP8 and M15 token-gate execution, repaired real DP16×TP4 optimizer commit, P45 APC, TP8 fixed-head, performance, and all full horizons remain unverified. M15 APC is target-VETOED and must remain off.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T07:15:52Z

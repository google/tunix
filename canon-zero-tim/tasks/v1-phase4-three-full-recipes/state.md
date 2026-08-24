# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: release base is `614156c1ab067192ab65b2969543e23904f192be`, including immutable Attempt-3 GSM8K error evidence. Runtime/gates are committed locally as `8a9c8019`; this ledger is the following local CL. The stack is not pushed because its new exact-image gate is separately approval-bound and unrun. No target runtime has been launched from it.
- Current phase: V1.P4.4 Attempt-3 repair before the next direct GSM8K full target. Host/static admission is green; the separately approved dependency-complete pinned-image gate and target remain unrun.
- Last verified fact: `g64m` passed strict step-0 pre-alignment for 194,633 action elements with both byte deltas zero, completed all 16 forward groups, and passed P59-local fixed-head plus q/k/v projection boundaries. Before optimizer, the attention entry repeated already TP-local K/V from 2 to 4 heads while the correct local cache remained 2; RPA rejected the mismatch. Patch 25 selects the local boundary only under exact P59 manual DP×TP context, validates local Q/K/V/cache, preserves ordinary global GQA, and adds a fail-closed full-run receipt. The real four-chip v5p gate then passed P59 DP2×TP2 real RPA forward+VJP2, wrong-cache negative, and ordinary DP1×TP4 global GQA in 32 seconds with finite gradients and zero optimizer commits. Final host gates pass V1 21/21, P57 144/144, P59 34/34, APC 31/31, flags 366/366; pinned-stock overlay construction is 36/36 with generated attention SHA `58d102e8c385368e7d1b47ce81ff3e866a8a1c43ba0b370a5da4aea729fb52f7`.
- Next action: obtain separate approval for the pinned-image gate, require `p59_rpa=2`, then submit the additive repair for commit/push review. Only after published exact readback may a fresh GSM8K full run start; P45/M15 remain gated on its first optimizer commit.
- Blockers: installed-attention DP2×TP4/TP8 execution, repaired real DP16×TP4 optimizer commit, P45 APC, TP8 fixed-head, performance, and all full horizons remain unverified. M15 APC is target-VETOED and must remain off.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-24T07:00:22Z

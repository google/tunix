# State

- Status: active
- Objective: render and locally certify three strict optimized 64-chip full-training manifests.
- Definition of done: one GSM8K DP16xTP4 manifest plus P45/M15 DP8xTP8 manifests resolve the complete workload-scoped v1 bundle and pass CPU, real-env, exact-image, and negative gates.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: tracked in the final recipe/ledger CL; release worktree clean and exactly five commits ahead of base at the last audit.
- Current phase: push approval
- Last verified fact: pinned image `418dc632...e53a` passes the complete exact-image gate; the current supported bundle then passes a one-host v5p DP4xTP1 proxy with 3/3 updates, 51/51 strict PASS, 0 FAIL, classifier PASS, and six verified evidence hashes. The committed-tree host rerun is V1 12/12, P57 128/128, APC 31/31, P59 30/30, flags 359/359, and the runtime diff against tested freeze tree `331ac609...` is empty.
- Next action: obtain explicit approval to push the exact five-commit stack; after push, render from its immutable 40-character SHA, then request separate approval for GSM8K full before P45 and M15.
- Blockers: push and each 64-chip launch remain separate approval boundaries; APC/fixed-head target geometries and both 64-chip topologies remain unverified until their direct full runs.
- Key artifacts: `scripts/render_three_full_recipes.py`; `scripts/classify_full_recipe.py`; `RUNBOOK.md`
- Updated: 2026-08-23T09:25:30Z

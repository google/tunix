# State

- Status: active
- Objective: Causally determine whether, under equal initial weights, data, topology, sampling, optimizer, and objective, finite trainer-inference mismatch worsens dense Qwen3-8B long-context FrozenLake convergence, final capability, or cross-seed stability relative to bitwise-zero mismatch, and separately quantify the systems cost of the zero-mismatch contract.
- Definition of done: a frozen, nontrivial FrozenLake recipe is selected using only stock-fast/mismatch-arm discovery outcomes and before observing any zero-arm learning outcome; the zero arm is bitwise exact; the stock arm has a finite reproducible treatment dose; paired campaigns differ only by the registered complete numerical zero-TIM bundle while all nonnumerical inputs remain equal; and the analysis reports the preregistered capability, stability, mechanism, and systems outcomes without exceeding the claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: the M15 stock curve and DP8 evaluation-row repair are published; `p57_eval0_att1`, `p57_eval0_att2`, and `p57_eval0_att3` are committed analysis-grade failures; this worktree contains the uncommitted workload-entrypoint generation-contract repair
- Current phase: P57.1 — M15 stock full-curve selection (calibration complete; stock eval-0 and 0→50 launch pending)
- Last verified fact: `p57_eval0_att3` used source `8acfe784...`, rendered the intended eight-generation DP8 evaluation, then stopped before model load because the real FrozenLake workload entrypoint still expected the obsolete evaluation count of two. The local repair gives renderer and entrypoint one `GENERATIONS_PER_PROMPT=8` source of truth and adds a regression that inspects the real entrypoint wiring. Host `90/90` and the pinned-image gate pass; no target rerun has occurred.
- Next action: seek separate commit/push approval. With separate launch approval, rerun stock eval-0 in `new` mode. Only a complete 100-map/800-reward eval-0 authorizes M15 stock train 0→50. Do not render or inspect zero-TIM.
- Blockers: entrypoint repair publication and target relaunch approval are pending. Attempts 1–3 are `INCONCLUSIVE`, not resumable, and contain no scientific result.
- Key artifacts: `plan.md`; `HANDOFF.md`; `RUNBOOK.md`; `phases/p57-0-readiness.md`; `phases/p57-1-stock-discovery.md`; `phases/p57-2-freeze-and-dose.md`; `phases/p57-3-paired-pilot.md`; `phases/p57-4-main-campaign.md`; `phases/p57-5-analysis.md`
- Updated: 2026-08-21 UTC

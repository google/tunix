# State

- Status: active
- Objective: Causally determine whether, under equal initial weights, data, topology, sampling, optimizer, and objective, finite trainer-inference mismatch worsens dense Qwen3-8B long-context FrozenLake convergence, final capability, or cross-seed stability relative to bitwise-zero mismatch, and separately quantify the systems cost of the zero-mismatch contract.
- Definition of done: a frozen, nontrivial FrozenLake recipe is selected using only stock-fast/mismatch-arm discovery outcomes and before observing any zero-arm learning outcome; the zero arm is bitwise exact; the stock arm has a finite reproducible treatment dose; paired campaigns differ only by the registered complete numerical zero-TIM bundle while all nonnumerical inputs remain equal; and the analysis reports the preregistered capability, stability, mechanism, and systems outcomes without exceeding the claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: the M15 stock curve and Step 37/38 guard repair are published; `p57_eval0_att1` and `p57_eval0_att2` are committed analysis-grade failures; this worktree contains the uncommitted DP8 evaluation-row repair
- Current phase: P57.1 — M15 stock full-curve selection (calibration complete; stock eval-0 and 0→50 launch pending)
- Last verified fact: `p57_eval0_att2` reached real rollout and then failed before its receipt because trainer-side EVAL rescore passed global M=2 into a DP8-sharded Splash Attention input. The repair sets deterministic eval generations to 8 (global M=8, shard-local M=1), updates the 800-reward classifier contract, and adds a render-time non-divisibility negative. Host `89/89` and the pinned-image gate, including the 8-generation evaluator lifecycle, both pass.
- Next action: seek separate commit/push approval. With separate launch approval, rerun stock eval-0 in `new` mode. Only a complete 100-map/800-reward eval-0 authorizes M15 stock train 0→50. Do not render or inspect zero-TIM.
- Blockers: repair publication and target relaunch approval are pending. Attempts 1 and 2 are `INCONCLUSIVE`, not resumable, and contain no scientific result.
- Key artifacts: `plan.md`; `HANDOFF.md`; `RUNBOOK.md`; `phases/p57-0-readiness.md`; `phases/p57-1-stock-discovery.md`; `phases/p57-2-freeze-and-dose.md`; `phases/p57-3-paired-pilot.md`; `phases/p57-4-main-campaign.md`; `phases/p57-5-analysis.md`
- Updated: 2026-08-21 UTC

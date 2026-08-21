# State

- Status: active
- Objective: Causally determine whether, under equal initial weights, data, topology, sampling, optimizer, and objective, finite trainer-inference mismatch worsens dense Qwen3-8B long-context FrozenLake convergence, final capability, or cross-seed stability relative to bitwise-zero mismatch, and separately quantify the systems cost of the zero-mismatch contract.
- Definition of done: a frozen, nontrivial FrozenLake recipe is selected using only stock-fast/mismatch-arm discovery outcomes and before observing any zero-arm learning outcome; the zero arm is bitwise exact; the stock arm has a finite reproducible treatment dose; paired campaigns differ only by the registered complete numerical zero-TIM bundle while all nonnumerical inputs remain equal; and the analysis reports the preregistered capability, stability, mechanism, and systems outcomes without exceeding the claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: implementation and evidence docs locally admitted; user authorized commit and push on 2026-08-21; publication is in progress
- Current phase: P57.0 — stock-fast calibration readiness (local admission complete; immutable source and hardware launch pending)
- Last verified fact: the single stochastic M10/M15/M20 JobSet now uses `CANON_P57_INFERENCE_REGIME=stock-fast`. Renderer, resolved `00_env.sh`, training entrypoint, JSON v2 receipt, and offline classifier agree on 12 absent and 25 zero switches. Host 73/73 passed; pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passed base 109/109, P45 40/40, PEFT 2/2, Agentic 4/4, stock-fast contract 3/3, overlay and fixed-head probes. `git diff --check` is clean. No TPU target has run.
- Next action: publish the approved P57 concern after rebasing and revalidating against the latest remote tip. Then render and mechanically verify the one stock-fast stochastic calibration JobSet from the resulting 40-character SHA, and request separate explicit launch approval.
- Blockers: no TPU launch has been authorized; target DP8xTP8 startup and live HBM/KV capacity remain unverified until the first approved launch reaches rollout progress.
- Key artifacts: `plan.md`; `HANDOFF.md`; `RUNBOOK.md`; `phases/p57-0-readiness.md`; `phases/p57-1-stock-discovery.md`; `phases/p57-2-freeze-and-dose.md`; `phases/p57-3-paired-pilot.md`; `phases/p57-4-main-campaign.md`; `phases/p57-5-analysis.md`
- Updated: 2026-08-21 UTC

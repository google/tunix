# State

- Status: active
- Objective: Deliver one hardened DeepSWE clean-data evaluator and three reproducible Qwen3 workload families, each renderable for 64 or 256 TPU chips without changing workload semantics.
- Definition of done: Evaluation, renderer, profile-parity, timeout, dataset-fingerprint, trajectory-artifact, and adjacent DeepSWE CPU gates pass; six non-launchable manifests render from the three workload families; no target claim is made without a real cluster run.
- Task directory: `canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/`
- Directory state: P46.1-P46.5 are published through `a4d165e854cc4c2320d8120e89aed185eaf61465`; the invalid-attempt retry and exact campaign finalizer are published by `a642ab267425a5b08b0cebb6e12c607f50f71831`
- Current phase: [P46.5 — true reward-only evaluation](phases/p46-5-reward-only-evaluation.md)
- Last verified fact: Returned 256-chip run `p46e25609` at source `8c0e90f38b68832a8ba7093fe78d655fcfd06ec4` initialized Qwen3-4B DP32 x TP8 on mlperf-v5p-256-3 and evaluated all 64 identities for l0/p0. It produced 59 valid trajectories (all SUCCEEDED, reward=0.0) plus 5 invalid attempts (4 MAX_CONTEXT_LIMIT_REACHED, 1 MODEL_TIMEOUT). The new fail-closed retry evaluator accurately identified the 5 pending valid identities and emitted `P46_EVAL_PHYSICAL_INCOMPLETE pending_valid_samples=5 invalid_attempts=5`, proving that the invalid-attempt detection and trajectory logging work correctly without false-positive completion claims. Full trajectory JSONL (6.0 MB) and head log are archived in `evidence/p46e25609/`.
- Next action: Continue remaining physical shards for DeepSWE clean-data evaluation or proceed to Q32 32B training.
- Blockers: None for evaluation pipeline integrity; Subshard 0 produced zero positive rewards.
- Key artifacts: `HANDOFF.md`, `plan.md`, `log.md`, `../../cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`, `phases/p46-5-reward-only-evaluation.md`, `evidence/p46e25609/trajectories.jsonl`
- Updated: 2026-08-13T21:30:00Z

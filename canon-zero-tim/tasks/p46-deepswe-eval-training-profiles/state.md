# State

- Status: active
- Objective: Deliver one hardened DeepSWE clean-data evaluator and three reproducible Qwen3 workload families, each renderable for 64 or 256 TPU chips without changing workload semantics.
- Definition of done: Evaluation, renderer, profile-parity, timeout, dataset-fingerprint, trajectory-artifact, and adjacent DeepSWE CPU gates pass; six non-launchable manifests render from the three workload families; no target claim is made without a real cluster run.
- Task directory: `canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/`
- Directory state: P46.1-P46.5 are published; P46.5 implementation commit is `a4d165e854cc4c2320d8120e89aed185eaf61465`
- Current phase: [P46.5 — true reward-only evaluation](phases/p46-5-reward-only-evaluation.md)
- Last verified fact: On a direct-attached v5p-8 host, P46.5 passed real Qwen3-4B DP1 x TP4 development L1/L2: `None/None` requests, extraction bypass, identical observer tokens under an exact engine-RNG reset, one clean R2E Docker task, one real `search` action, final reward 0, valid `SUCCEEDED` trajectory, and zero residual containers. Report SHA-256 is `db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c`. After rebasing onto operator HEAD `23bb2a3c`, P46 31, P34, P39, P44 and targeted sampler publication gates pass at `a4d165e8`; neither target arm has been launched.
- Next action: Read back the exact published operator SHA, then run the 64-chip paired N16 L3 canary plus valid trajectories/hour before promoting reward-only as the clean-eval default.
- Blockers: L3 paired 64-chip statistics, Kubernetes cleanup/throughput, TP8 Pathways behavior, Qwen3-4B three updates, and Qwen3-32B training remain target-unverified.
- Key artifacts: `HANDOFF.md`, `plan.md`, `log.md`, `../../cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`, `phases/p46-5-reward-only-evaluation.md`, dependency ledgers `../p39-deepswe-production/` and `../p44-deepswe-qwen4b-parity/`
- Updated: 2026-08-13T06:46:31Z

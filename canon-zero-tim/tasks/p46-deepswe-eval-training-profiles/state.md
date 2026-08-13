# State

- Status: active
- Objective: Deliver one hardened DeepSWE clean-data evaluator and three reproducible Qwen3 workload families, each renderable for 64 or 256 TPU chips without changing workload semantics.
- Definition of done: Evaluation, renderer, profile-parity, timeout, dataset-fingerprint, trajectory-artifact, and adjacent DeepSWE CPU gates pass; six non-launchable manifests render from the three workload families; no target claim is made without a real cluster run.
- Task directory: `canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/`
- Directory state: visible and unignored; unpublished in the dirty development worktree
- Current phase: [P46.4 — remote execution and evidence return](phases/p46-4-remote-execution.md)
- Last verified fact: The worktree and `origin/yuxzhang/canon-zero-tim` now share base `99c3f7af761c859caa6c81ab509446cc3cc47dc0`. The archived P34r03 log completed 64/64 rollout records but clipped all 64 as `ENV_TIMEOUT`, then failed actor log-prob preparation because its trainer mesh was `dp,tp` while `data_sharding_axis` was stale `fsdp`. The launcher now derives the data axis from the actual trainer mesh and emits `[DEEPSWE.DATA_SHARDING] PASS`. P34 static/trajectory/update, P44 (41 cases), P46 (17 cases), and `git diff --check` pass. This host has no `libtpu.so`, so no target runtime claim was produced.
- Next action: After explicit commit/push approval, publish the reviewed lifecycle plus mesh-axis fix to `origin/yuxzhang/canon-zero-tim`, read back its 40-character SHA, then hand the updated `P46_DEEPSWE_PROFILES_RUNBOOK.md` and `HANDOFF.md` to the remote agent for the gated Q4 evaluation -> Q4 three-update -> Q32 sequence.
- Blockers: Real R2E Kubernetes cleanup, TP8 Pathways behavior, Qwen3-4B three updates, and Qwen3-32B training remain target-unverified.
- Key artifacts: `HANDOFF.md`, `plan.md`, `log.md`, `../../cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`, `phases/p46-4-remote-execution.md`, dependency ledgers `../p39-deepswe-production/` and `../p44-deepswe-qwen4b-parity/`
- Updated: 2026-08-13T03:47:53Z

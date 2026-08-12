# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.6 — target promotion ladder](phases/p44-6-target-promotion.md)
- Last verified fact: On operator baseline `7ea2176f807e3e13fde17499e15fef2bd497363b`, the r02 repair passes 32 P44 CPU cases plus two affected learner unit tests in the pinned dependency image; adjacent P43/P39/P34 and exact-image gates remain green. The optional local smoke is `BLOCKED_REAL_ENVIRONMENT` because this session exposes neither TPU/libtpu nor existing Qwen3-4B/R2E prerequisites.
- Next action: After explicit commit/push authorization, publish to `origin/yuxzhang/canon-zero-tim`, read back its exact SHA, then have the launch agent run rollout-only on whichever exact 64- or 256-device allocation is available.
- Blockers: target execution requires a published repair SHA and remote allocation; optional one-host smoke prerequisites are absent in this session.
- Key artifacts: `debug_logs/p44_p44r02_deepswe_256_parity.raw.log`, `plan.md`, `log.md`
- Updated: 2026-08-12T01:07:00Z

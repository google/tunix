# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.6 — target promotion ladder](phases/p44-6-target-promotion.md)
- Last verified fact: Repair implementation commit `5f0cf7e04b34932d8c9deb2463f3b205e3ad8b51`, based on operator SHA `7ea2176f807e3e13fde17499e15fef2bd497363b`, passes 32 P44 CPU cases plus two affected learner unit tests in the pinned dependency image; adjacent P43/P39/P34 and exact-image gates remain green.
- Next action: Read back the exact publication head from `origin/yuxzhang/canon-zero-tim`, then have the launch agent run rollout-only on whichever exact 64- or 256-device allocation is available.
- Blockers: target execution requires a remote allocation; optional one-host smoke prerequisites are absent in this session.
- Key artifacts: `debug_logs/p44_p44r02_deepswe_256_parity.raw.log`, `plan.md`, `log.md`
- Updated: 2026-08-12T01:11:42Z

# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: publication must be confirmed by remote branch read-back
- Current phase: P44.6 — target promotion ladder
- Last verified fact: P44 passes 27/27 locally and in the immutable local image, both dataset launchers avoid the removed datasets argument, and Qwen4B/Qwen8B/Qwen32B overlay regressions pass.
- Next action: Resolve and detach at the exact remote branch head, then run the available allocation's rollout-only stage, classify it, and continue that allocation's independent ladder.
- Blockers: none for local implementation; target evidence requires operator-owned 64-chip and 256-chip cluster runs.
- Key artifacts: `plan.md`, `log.md`
- Updated: 2026-08-12T00:20:18Z

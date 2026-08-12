# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.6 — target promotion ladder](phases/p44-6-target-promotion.md)
- Last verified fact: Archived 256-device attempt `p44r04` reached the Qwen3-4B MLP and failed because TP8-local SwiGLU width `1216` did not satisfy the unchanged BF256 Pallas kernel. On latest operator baseline `e4ead609498771987c011a9cbc16fec7e4b17f69`, the uncommitted P44.9 repair pins `1216->1280` for Qwen3-4B and `3200->3328` for Qwen3-32B, leaves Qwen3-8B width `3072` unpadded, and passes exact forward/VJP plus P44/P43/P39/P34 CPU and exact-image gates.
- Next action: Obtain explicit commit/push approval, publish the P44.9 repair to `yuxzhang/canon-zero-tim`, record the exact read-back publication SHA in this handoff, then launch a fresh rollout-only `p44r05` from that SHA.
- Blockers: the P44.9 repair has no publication commit yet; target execution also requires a remote 64- or 256-device allocation. The optional one-host smoke prerequisites remain absent in this session.
- Key artifacts: `debug_logs/p44_p44r03_deepswe_256_parity.raw.log`, `debug_logs/p44_p44r04_deepswe_256_parity.raw.log`, `phases/p44-9-r04-swiglu-feature-padding.md`, `plan.md`, `log.md`
- Updated: 2026-08-12T02:45:02Z

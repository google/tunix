# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 128 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.13 — Q4 128-chip topology migration](phases/p44-13-q4-128-topology.md)
- Last verified fact: The local unpublished topology migration renders the same Qwen3-4B B4/G4, clean-data, device-optimizer, three-update recipe on 64 and 128 devices with a shared 3600-second rollout-batch limit and durable trajectory capture. The 128-chip `4x4x8` split is host-complete with DP8 x TP8 per role. P44 CPU passes 41 cases. Historical 256-device attempts remain evidence but are no longer launchable Q4 contracts.
- Next action: Follow the current P46 handoff after publication: run Q4 clean evaluation on an available 64/128 topology, then render Q4-Instruct `q4-debug` three-update on an admitted topology and inspect all three trajectory batches plus sandbox cleanup before Qwen3-32B.
- Blockers: TP8, Pathways role separation, real Kubernetes sandbox deletion and three optimizer updates remain target-unverified. The older one-host zero-signal result still does not promote a local update claim.
- Key artifacts: `phases/p44-13-q4-128-topology.md`, `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`, `tests/p44_deepswe_qwen4b_parity/run_exact_image.sh`, `plan.md`, `log.md`
- Updated: 2026-08-13T22:01:34Z

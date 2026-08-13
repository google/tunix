# State

- Status: active
- Objective: Provide fail-closed Qwen3-4B DeepSWE debug launches on 64 and 256 TPU chips that share one functional recipe, stage ladder, artifact schema, and classifier contract.
- Definition of done: Local parity, renderer, exact-image, and adjacent DeepSWE gates pass; then both topologies produce classified rollout-only, one-update, and three-update target evidence from the same pinned recipe.
- Task directory: `canon-zero-tim/tasks/p44-deepswe-qwen4b-parity/`
- Directory state: tracked
- Current phase: [P44.12 — bounded three-update debug defaults](phases/p44-12-bounded-three-update.md)
- Last verified fact: P44.12 is included in published implementation commit `e1b4009394c49ea015919bda0cfdb97c12c221b5`. Both 64- and 256-device renderers emit the same Qwen3-4B B4/G4, clean-data, device-optimizer, three-update recipe with a shared 3600-second rollout-batch limit and durable trajectory capture. P44 CPU passes 41 cases; P43/P39/P34 regressions remain green. No target run occurred.
- Next action: Follow the current P46 handoff: run Q4 clean evaluation on the available 64/256 topology, then render Q4-Instruct `q4-debug` three-update on the available topology and inspect all three trajectory batches plus sandbox cleanup before Qwen3-32B.
- Blockers: TP8, Pathways role separation, real Kubernetes sandbox deletion and three optimizer updates remain target-unverified. The older one-host zero-signal result still does not promote a local update claim.
- Key artifacts: `phases/p44-12-bounded-three-update.md`, `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`, `tests/p44_deepswe_qwen4b_parity/run_exact_image.sh`, `plan.md`, `log.md`
- Updated: 2026-08-13T03:54:03Z

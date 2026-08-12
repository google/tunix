# P44 Qwen3-4B DeepSWE parity-debug handoff

The operator-facing source of truth is
`../../cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`.

## Objective

Run the same Qwen3-4B DeepSWE functional recipe on either 64 devices
(`4x4x4`, DP4 x TP8 per role) or 256 devices (`4x8x8`, DP16 x TP8 per role).
Each allocation has its own `rollout-only` -> `one-update` -> `three-update`
promotion ladder, but both write the same real-trajectory schemas and grouped
solve metrics and are judged by one topology-aware classifier.

This is a fast systems-debug lane. It does not replace or admit the Qwen3-32B
production workload and does not claim bitwise, performance, quality, or
zero-TIM equivalence between allocations.

## Publication contract

- Required remote branch: `yuxzhang/canon-zero-tim`
- Exact publication SHA: resolve the current remote head with
  `git ls-remote origin refs/heads/yuxzhang/canon-zero-tim`, detach at that
  exact SHA, and record it in the rendered JobSet and returned evidence
- Local development branch: `codex/p43-deepswe-64-debug`
- Remote execution owner: the launch agent/operator, not the implementation
  agent

The launch agent must fetch the required remote branch, detach at its exact
read-back SHA, verify a clean checkout, and pass that same SHA to the renderer.
Do not launch a local development worktree or an unverified symbolic branch.

## Current evidence

- P44 shared-recipe, Qwen3-4B TP8 overlay, both topology renderers/preflights,
  artifact schemas, both dataset entrypoints, and classifier controls: PASS
  locally (27 tests).
- Qwen4B overlay exact-image CPU gate: PASS in local immutable image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  This is not a remote registry digest; rerun with the launch image digest.
- Adjacent P43/P39/P34 regressions pass: P43 21/21, P39 15/15, and P34 10
  suites. Qwen8B and Qwen32B also reinstall 29/29 and pass their exact-image
  gates after the shared model-pinned BK change.
- Remote 64-device and 256-device stages: NOT RUN.

## First operator action after publication

Follow the runbook's fetch and immutable-input preflight. Start only the
64-device `rollout-only` stage unless the launch owner explicitly selects the
256-device ladder first. Return the complete failure package on any red or
inconclusive result; do not edit the recipe or immediately jump allocations.

If 257 devices are available, P44 still renders an exact 256-device
`4x8x8` target. The additional device is outside the mesh.

# P43 DeepSWE 64-chip debug handoff

The operator-facing source of truth is
`../../cluster/P43_DEEPSWE_64CHIP_DEBUG_RUNBOOK.md`.

## Objective

Bring up real DeepSWE on one 64-chip `4x4x4` slice using Qwen3-8B while
retaining TP8. Run the promotion ladder `rollout-only` -> `one-update` ->
`three-update`. Persist and inspect every real trajectory batch and grouped
solve/advantage metrics before interpreting training health.

## Publication

- Required branch: `yuxzhang/canon-zero-tim`
- Exact publication SHA: use the read-back SHA delivered with this handoff;
  independently confirm it with
  `git ls-remote origin refs/heads/yuxzhang/canon-zero-tim`
- Local development branch: `codex/p43-deepswe-64-debug`
- Remote execution owner: the launch agent/operator, not the implementation
  agent

Do not launch from the development branch or from a symbolic branch without
also pinning the exact 40-character publication SHA in the rendered JobSet.

## Current evidence

- P43 CPU artifact/contract/renderer/env/classifier gates: PASS locally (21
  tests).
- Adjacent P39 64-chip CPU gate: PASS locally.
- Adjacent P34 static gate: PASS locally (10 suites).
- Qwen8b overlay exact-image CPU gate: PASS locally in immutable image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Remote 64-chip stages: NOT RUN.

The tested implementation commit and gate counts are recorded in `state.md`
and `log.md`; the final documentation commit is the branch read-back SHA
delivered by the publishing agent. Remote logs and outcomes belong in a
follow-up P43.5 ledger entry; a local pass must not be relabeled as remote
evidence.

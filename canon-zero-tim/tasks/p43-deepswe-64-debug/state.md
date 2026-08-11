# State

- Status: active
- Objective: Publish a Qwen3-8B DeepSWE debug launch on 64 TPU chips with TP8, durable real trajectories, grouped solve metrics, and a remote-agent runbook.
- Definition of done: Local and pinned-image gates pass; an immutable commit is published to `yuxzhang/canon-zero-tim`; the remote 64-chip `rollout-only`, `one-update`, and `three-update` stages persist their required artifacts and the final stage records three finite optimizer commits.
- Task directory: `canon-zero-tim/tasks/p43-deepswe-64-debug`
- Directory state: tracked
- Current phase: P43.3 — commit-bound integrated verification
- Last verified fact: P43's 21-test CPU gate, P39's 15-test adjacent gate,
  P34's 10-suite static gate, all three renderer CLI smokes, and the qwen8b
  exact-image gate passed; the immutable local image ID was
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Next action: Fetch the publication branch again, integrate any remote
  movement, commit the candidate, and rerun the publication-critical gates
  from that exact SHA before entering P43.4.
- Blockers: none
- Key artifacts: `plan.md`, `log.md`, `phases/p43-1-debug-contract.md`
- Updated: 2026-08-11T23:33:19Z

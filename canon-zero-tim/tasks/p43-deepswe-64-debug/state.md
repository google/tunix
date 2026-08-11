# State

- Status: active
- Objective: Publish a Qwen3-8B DeepSWE debug launch on 64 TPU chips with TP8, durable real trajectories, grouped solve metrics, and a remote-agent runbook.
- Definition of done: Local and pinned-image gates pass; an immutable commit is published to `yuxzhang/canon-zero-tim`; the remote 64-chip `rollout-only`, `one-update`, and `three-update` stages persist their required artifacts and the final stage records three finite optimizer commits.
- Task directory: `canon-zero-tim/tasks/p43-deepswe-64-debug`
- Directory state: tracked
- Current phase: P43.5 — remote 64-chip evidence pending
- Last verified fact: Implementation commit
  `c73443e3a63a022976b2fa07d6c1b0475903933f`, rebased onto fetched remote tip
  `340b0e364f374fde8798d8f62331e6bc33e0e58a`, passed the P43 21-test gate,
  P39 15-test gate, P34 10-suite gate, qwen8b exact-image gate, and adjacent
  P38 postflight/renderer controls. The surrounding publication transaction
  pushes the documentation commit containing this state to the authorized
  branch and reports its read-back SHA to the operator.
- Next action: The remote agent fetches `yuxzhang/canon-zero-tim`, verifies the
  read-back SHA supplied with this handoff, then runs `rollout-only` exactly as
  documented. No remote P43 stage has run yet.
- Blockers: none
- Key artifacts: `plan.md`, `log.md`, `phases/p43-1-debug-contract.md`
- Updated: 2026-08-11T23:35:46Z

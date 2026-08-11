# Plan

## Outcome

Create a separate default-off DeepSWE debug path for one 64-chip `4x4x4`
Pathways slice. Split it into 32-device rollout and trainer roles, each
DP4xTP8; use Qwen3-8B and a small fixed real workload to shorten iteration;
persist readable post-environment trajectories and per-prompt grouped reward
metrics on the run PVC. Do not weaken or relabel the P34/P39 Qwen3-32B
production contracts. The reviewed result is ultimately published to
`yuxzhang/canon-zero-tim` at one immutable 40-character SHA for another agent
to launch on the remote cluster.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P43.0 | Clean isolated branch and canonical task ledger | Exact fetched base SHA and clean starting worktree are recorded | passed |
| P43.1 | Separate Qwen3-8B DP4xTP8 debug profile, renderer, and bounded stage contract | CPU renderer/preflight tests accept only the signed 64-chip debug geometry and reject production-contract drift | passed |
| P43.2 | Durable real-trajectory dump and grouped solve/advantage metrics | Unit fixtures pass for all-solved, all-failed, mixed, incomplete, and non-binary-reward cases; batch artifacts round-trip | passed |
| P43.3 | Integrated local and pinned-image evidence | P43 gates and adjacent P34/P39 regressions pass from the exact candidate SHA | active |
| P43.4 | Remote-agent operator handbook and immutable publication | Runbook render examples validate; reviewed commit is pushed to `yuxzhang/canon-zero-tim` and read back at the exact SHA | pending |
| P43.5 | Remote 64-chip evidence | `rollout-only`, `one-update`, then `three-update` produce the required raw logs, trajectory files, metric rows, and three finite commits | pending |

## Decisions

- Confirmed: the fetched canonical base is `39e18f7ddee8f5c7ab7dfcd269e83d7785a684c2`.
- Confirmed: development occurs on `codex/p43-deepswe-64-debug`; final publication goes to `yuxzhang/canon-zero-tim` after review.
- Decision: use Qwen3-8B while retaining TP8 on both roles; do not introduce a smaller-model support variable before the 8B path runs.
- Decision: start with 4 prompts x 4 generations, at most 5 turns and a bounded response, then use `rollout-only`, `one-update`, and `three-update` stages.
- Decision: R2E-Gym exposes no independent boolean verdict in this path, so
  P43 defines solved as a complete trajectory with final reward exactly `1.0`
  (`r2egym_final_reward_eq_1`). Raw reward remains in every artifact and
  positive non-binary rewards are counted separately.
- Decision: distinguish all-solved, all-failed, mixed, and incomplete prompt groups. Mixed groups are the direct measure of nonzero within-group learning signal.
- Decision: remote execution is operator-owned. Code publication is authorized; cluster apply is not performed by this task owner.

# V1 system-optimization workload rollout state

- Status: `OFFLINE COMPLETE / TARGET NOT RUN`
- Bound worktree: `/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`
- Bound source: `d4128940464054866d466a6cce5adf326f513caf`
- Active phase: `P4 — canonical handoff integration complete`
- Objective: deliver the already-reviewed reverse-path optimization bundle to
  the exact FrozenLake P45/M15 strict full recipes and the exact DeepSWE
  Qwen3-4B strict Zero-HP full recipe, without changing Native, diagnostic,
  non-HP, or neighboring workload behavior.
- Claim ceiling: host construction only until separately approved target runs
  complete. FrozenLake DP8xTP8 and DeepSWE DP8xTP8 performance are
  `TARGET NOT RUN` for this rollout.
- Mutation policy: implementation, tests, and task records are in scope.
  Commit, push, image publication, TPU/Kubernetes launch, and remote mutation
  require a separate explicit approval.
- Canonical phase: `phases/v1-p4-handoff-integration.md`

## Current decision

P74 is a source-level checked-VMA dispatch repair, not a new workload flag.
It is selected by the already registered `CANON_P59_CHECKED_VMA=1` path. The
rollout therefore adds the reviewed host-receipt and forward-tape selectors to
the exact production renderers while leaving
`CANON_DP_COLLECTIVE_REDUCE` absent because DP8/target certification has not
run.

The source, renderer, persistence, runtime-contract, negative-control, and
pinned-image CPU gates are green. `RUNBOOK.md` contains render-only commands;
no TPU/Kubernetes run was launched and no DP8xTP8 performance claim is made.

## Resolved workload identities

| Workload | Exact admitted identity | Neighbor isolation |
|---|---|---|
| FrozenLake P45 | Zero/full, DP8xTP8, 300 updates, P45 readiness, HP profile | stock, eval, checkpointed, non-HP, and diagnostics unchanged |
| FrozenLake M15 | Zero/full, DP8xTP8, 300 updates, `m15:main`, HP profile | selection split, APC experiment, and diagnostics unchanged |
| DeepSWE Qwen3-4B | Zero/full, DP8xTP8, 1,000 updates, strict alignment, HP profile | Native, Native+IS, ordinary Zero, three-update, checked-VMA diagnostics, and seam diagnostics unchanged |

## Required optimization tuple

The exact production full renderers must carry:

```text
CANON_P59_CHECKED_VMA=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
```

FrozenLake and DeepSWE additionally retain
`CANON_P67_P66_VMA_P59_ONLY=1`. The collective reducer remains absent.

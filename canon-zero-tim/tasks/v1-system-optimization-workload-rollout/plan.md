# V1 system-optimization workload rollout plan

| Phase | Status | Exit gate |
|---|---|---|
| P0 — bind source and audit workload identities | `DONE` | Exact worktree, source, profiles, renderers, flag readers, and neighboring arms identified |
| P1 — workload contracts and implementation | `DONE` | One shared reviewed optimization tuple is consumed only by FrozenLake P45/M15 full and DeepSWE Zero-HP full renderers; forbidden reducer remains absent |
| P2 — host and exact-image validation | `DONE` | Positive renders, authoritative `00_env.sh` reloads, Python contracts, negative controls, adjacent suites, flag audit, and pinned-image gates pass |
| P3 — delivery and target handoff | `DONE` | Phase/result log and runbook commands are complete; local CL description is ready; target debt is explicitly recorded |
| P4 — canonical handoff integration | `DONE` | FrozenLake and DeepSWE owning handoffs/runbooks select the registered wrappers, pin the exact tuple, preserve scientific blockers, and pass a stale-doc regression |

## P1 implementation contract

1. Put the reviewed performance-only selector tuple in one Python source of
   truth under `cluster/`; keep checked-VMA, first-update, and P67 numerical
   protection explicit in their registered workload profiles.
2. Make the existing FrozenLake two-full renderer consume the tuple for both
   P45 and M15, and make its validator reject drift or accidental collective
   reducer admission.
3. Make the P58 renderer consume the tuple only for
   `--arm zero --stage full --high-performance`. Diagnostic HP-shaped carriers
   deliberately do not inherit the production performance tuple.
4. Strengthen renderer, profile, real environment reload, and neighboring-arm
   tests. A successful test must prove both the positive values and the
   absence of `CANON_DP_COLLECTIVE_REDUCE`.

## Validation order

1. Python compile and `git diff --check`.
2. Focused FrozenLake and P58 renderer/profile/environment tests.
3. Real `cluster/steps/00_env.sh` persistence and reload tests.
4. Flag registry audit and adjacent CPU suites.
5. Exact pinned-image FrozenLake and P58 gates with the worktree mounted
   read-only.
6. No TPU or Kubernetes action in this phase without a separate explicit
   launch approval.

## Numerical and operational rejection rules

- Any change to Native, Native+IS, non-HP Zero, diagnostic, eval, checkpoint,
  or neighboring profile resolution is a rollout reject.
- Any appearance of `CANON_DP_COLLECTIVE_REDUCE` is a performance-certification
  reject.
- Any checked-VMA/P67/first-update removal is a numerical-safety reject.
- Host and exact-image passes are construction evidence only; they do not
  promote DP8xTP8 target performance or convergence claims.

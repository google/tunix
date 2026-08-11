# Plan

## Outcome

Make the existing `very-high` Pod priority a rendered-manifest invariant for all
GSM8K, FrozenLake, and DeepSWE runs. The change must not create or mutate a
cluster-scoped PriorityClass, alter restart policy, relax alignment gates, or
change training numerics.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P40.1 | P33/P34 renderer assertions and workload-specific positive/negative tests | Focused renderer unit tests pass and reject missing/mismatched priority | passed |
| P40.2 | P33/P34 operator preflight documentation | Both runbooks require `very-high value=1000 policy=PreemptLowerPriority` before render/apply | passed |
| P40.3 | Integrated verification and handoff | P33/P34 static suites and `git diff --check` pass | passed |

## Decisions

- Confirmed: Both reviewed base JobSet templates already assign `very-high` to the Pathways head and worker Pod specs.
- Confirmed: P33 renders GSM8K and FrozenLake from the 64-chip base; P34 renders DeepSWE from the 256-chip base.
- Decision: Keep priority as a fixed renderer invariant rather than a workload environment variable; Kubernetes scheduling policy belongs to the Pod contract, not model numerics.
- Decision: Validate but do not create the cluster-scoped PriorityClass. Cluster mutation remains an explicit operator action.

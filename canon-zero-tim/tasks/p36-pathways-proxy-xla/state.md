# State

- Status: active
- Objective: Determine whether delivering `--xla_allow_excess_precision=false` to the Pathways proxy removes the 64-chip cross-program way-count drift.
- Definition of done: A source-pinned attempt-zero gate-only JobSet passes local fail-closed checks and one 64-chip run emits the complete registered way-count table with archived proxy evidence.
- Task directory: `canon-zero-tim/tasks/p36-pathways-proxy-xla/`
- Directory state: tracked
- Current phase: P36.2 target 64-chip way-count measurement (`phases/p36-2-target-waycount.md`)
- Last verified fact: P36.1 local gate passed 6/6, P35 renderer regression passed 7/7, and the P33 renderer passed 6/6 in the pinned image.
- Next action: Publish the reviewed source, render a source-pinned P36 JobSet, and launch it as Attempt 0 on the authorized 64-chip cluster.
- Blockers: This host has no `kubectl` binary or configured GKE context; target execution must be performed by the authorized cluster operator.
- Key artifacts: `cluster/P36_PROXY_XLA_HANDOFF.md`
- Updated: 2026-08-10 UTC after the P36.1 local gate

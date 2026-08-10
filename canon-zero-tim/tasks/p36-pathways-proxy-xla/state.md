# State

- Status: active
- Objective: Determine whether delivering `--xla_allow_excess_precision=false` to the Pathways proxy removes the 64-chip cross-program way-count drift.
- Definition of done: A source-pinned attempt-zero gate-only JobSet passes local fail-closed checks and one 64-chip run emits the complete registered way-count table with archived proxy evidence.
- Task directory: `canon-zero-tim/tasks/p36-pathways-proxy-xla/`
- Directory state: tracked
- Current phase: P36.2 target 64-chip way-count measurement (`phases/p36-2-target-waycount.md`)
- Last verified fact: Target Attempt `flagon1` rejected the raw XLA flag before running a probe;
  the corrected proxy-`XLA_FLAGS` renderer passes 7/7 local fail-closed tests.
- Next action: Review and publish the corrected source, render a new source-pinned `envon1`
  JobSet, and run the attempt-zero 64-chip gate.
- Blockers: The corrected source is not committed or published. This host has no `kubectl` binary
  or configured GKE context; target execution must be performed by the authorized cluster operator.
- Key artifacts: `cluster/P36_PROXY_XLA_HANDOFF.md`,
  `debug_logs/p36_flagon1/pathways_proxy.raw.log`,
  `phases/p36-1a-onehost-sensitivity.md`,
  `artifacts/p36_onehost_excess_{off,on}.raw.log`
- Updated: 2026-08-10 UTC after correcting the rejected raw-argument contract

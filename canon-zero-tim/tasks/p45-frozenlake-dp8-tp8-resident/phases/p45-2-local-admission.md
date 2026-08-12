# P45.2 — local static and render admission

- Status: completed

## Evidence

- Pinned image: `tunix_frozenlake_image:vllm-tpu0.25.0`
- Focused command:
  `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_cpu.sh`
- Result: 77 workload/renderer/classifier tests passed; 29 alignment tests
  passed; merged profile preflight emitted
  `[P45.PROFILE] ADMITTED_PREFLIGHT_PASS`.
- Adjacent command:
  `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh`
- Result on the final source: complete adjacent P33/P38 CPU gate passed; the
  existing DP16xTP4 contract remains admitted and its values are unchanged.
- Static gates: `git diff --check` and shell syntax checks passed.

## Rendered contract

The isolated renderer emits exactly:

- `jobset-p45-frozenlake-full-dp8-tp8-resident.yaml`
- `jobset-p45-frozenlake-full-dp8-tp8-resident-eval.yaml`

Both attest DP8xTP8, local/global M256/M2048, 32 local trajectories, global
trajectory microbatch 8, resident optimizer placement, offload disabled,
online W&B, and warning-only FrozenLake alignment. Only the evaluation variant
adds four held-out batches every ten policy steps.

## Scope

This is local wiring evidence. A four-chip local host cannot execute an 8x8
mesh, so this phase does not prove target HBM capacity, collective behavior, or
throughput.

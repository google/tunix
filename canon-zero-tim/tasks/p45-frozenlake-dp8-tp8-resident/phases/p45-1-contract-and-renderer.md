# P45.1 — DP8xTP8 resident contract and renderer

- Status: completed

## Finding

- Confirmed: the current 64-chip FrozenLake path is DP16xTP4 with global M4096 and pinned-host optimizer offload.
- Confirmed: DP8xTP8 still uses all 64 devices. With local canonical M256 its global token bucket is 2048; keeping 32 prompts and 8 generations produces 32 local trajectories per DP rank.
- Hypothesis: TP8 resident placement has materially safer HBM capacity than the measured TP4 resident canary, but only a target run can quantify the reserve and throughput.

## Execution

1. Register a separate `frozenlake-dp8-tp8` workload and profile.
2. Generalize P33 workload validation, mesh construction, segmented update geometry, and adapter topology lookup to the active workload.
3. Add isolated full and full-eval renderer entries with resident optimizer placement.
4. Preserve 32 global prompts, 8 generations, learning rate `1e-6`, sequence limits, 450 steps, and evaluation cadence. Set the global trajectory microbatch to 8 so one fixed group contains exactly one trajectory from every DP rank.
5. Require DP8xTP8, 32 ordered gradient groups, local M256/global M2048, per-DP scheduler capacity 32/256, and zero optimizer offload.

## Exit gate

- Command: focused `pytest` suites for DP workload, P33 rendering, learner geometry, and FrozenLake evaluation plus `git diff --check`.
- Pass: both new manifests satisfy the P45 contract; existing P33 manifests retain their old topology/placement; invalid mixed placement and topology drift fail closed.
- Fail: do not launch. Record the exact contract mismatch and keep the existing DP16xTP4 path unchanged.

## Result

Implemented a separate `frozenlake-dp8-tp8` workload and resident profile,
generalized the FrozenLake recipe/adapter/learner contracts, and added an
isolated P45 renderer. The old DP16xTP4 P33 renderer still emits its original
six workload variants. The P45 renderer emits only full and full-eval.

The old value 16 was separated into three real quantities: DP size 8, global
trajectory microbatch 8, and 32 local rank-major gradient groups. A dedicated
test now rejects recombining those quantities.

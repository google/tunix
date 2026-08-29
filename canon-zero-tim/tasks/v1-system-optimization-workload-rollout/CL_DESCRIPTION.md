Roll the V1 system optimization into full training

Apply the reviewed checked-VMA reverse-path optimization bundle to the exact
FrozenLake P45/M15 full HP recipes and the exact DeepSWE Qwen3-4B Zero/full/HP
recipe. Keep the tuple in one renderer helper, enforce it again at the runtime
environment boundary, and reject leakage into neighboring or diagnostic arms.

Route the canonical FrozenLake and DeepSWE handoffs and runbooks through those
registered render-only entry points so operators cannot accidentally reuse a
legacy resident carrier or an older Zero-HP manifest.

The rollout keeps checked-VMA, P67, and first-update protection enabled. It
also leaves `CANON_DP_COLLECTIVE_REDUCE` absent because no DP8 target oracle
admits that selector.

Verified by running the focused FrozenLake, Phase4, P58, shared-rollout, and
flag-registry tests documented in `validation.log`.

Verified by running the FrozenLake and DeepSWE pinned exact-image CPU gates
documented in `validation.log`.

Not verified on FrozenLake or DeepSWE TPU targets because this CL preparation
did not have a separately approved TPU/Kubernetes launch.

本方案的缺点

This expands an experimental performance bundle to two new workload
renderers before target performance is measured. The fail-closed contracts
make configuration drift visible, but DP8xTP8 speedup and convergence still
require fresh target runs, and the unverified collective reducer remains
disabled.

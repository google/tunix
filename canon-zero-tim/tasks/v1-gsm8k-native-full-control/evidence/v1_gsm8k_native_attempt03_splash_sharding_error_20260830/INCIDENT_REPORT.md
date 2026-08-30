# GSM8K Native Control Attempt 03 Splash Attention Sharding Error Report

**Incident ID**: `v1_gsm8k_native_attempt03_splash_sharding_error_20260830`  
**Workload**: `canon-v1ctl-gsm-nat-gnat03-0b62b6bb` (64 TPU v5p, DP16xTP4)  
**Execution Date**: 2026-08-30  
**Source Commit**: `0b62b6bbd3d9fa44268c7640047d4b60047cb4d5`  
**Failure Point**: Step 0 learner `agentic_grpo_learner._process_results` -> `compute_per_token_logps` -> `qwen3/model.py:678` `sharded_splash_attn`

---

## 1. Error Summary

Commit `0b62b6bb` successfully fixed the activation sharding constraint on Explicit meshes by using `jax.sharding.reshard`.
However, during the first post-rollout logps forward pass on DP16xTP4 mesh (`axis_types=(AxisType.Explicit, AxisType.Explicit)`), `tunix/models/qwen3/model.py` invoked `sharded_splash_attn` via `shard_map`.

The splash attention kernel pytree contains array leaves (such as `int8[4,8,8]` mask structure) created unsharded (`P(None, None, None)`).
The `shard_map` `in_specs` declared `kernel_spec` which partitions the kernel over `('model', None, None)`.
On an Explicit-axis mesh, JAX enforces that arguments passed to `shard_map` must already match `in_specs`:

```text
ValueError: in_specs passed to shard_map: P('model', None, None) does not match the specs of the input: P(None, None, None) for arg: int8[4,8,8]. `in_specs` is an optional argument so you can omit specifying it and shard_map will infer the in_specs from the arguments. If you want to reshard your inputs, you can use `jax.reshard` on the arguments and then pass those args to shard_map.
```

---

## 2. Root Cause Analysis

1. `splash_attn_kernel` is instantiated inside `Attention.block` without an initial sharding layout.
2. `kernel_spec = splash_attn_kernel.manual_sharding_spec(shd.NamedSharding(mesh, P(shd_n, shd_t)))` returns the intended target `PartitionSpec` tree.
3. On an Explicit mesh, JAX requires the `splash_attn_kernel` object to be explicitly resharded via `jax.sharding.reshard(arr, NamedSharding(mesh, spec))` before passing it as an input to `shard_map`.


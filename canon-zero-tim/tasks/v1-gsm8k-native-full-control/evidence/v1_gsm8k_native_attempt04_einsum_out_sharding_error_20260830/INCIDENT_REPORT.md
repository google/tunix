# GSM8K Native Control Attempt 04 Einsum Output Sharding Error Report

**Incident ID**: `v1_gsm8k_native_attempt04_einsum_out_sharding_error_20260830`  
**Workload**: `canon-v1ctl-gsm-nat-gnat04-0d224e4a` (64 TPU v5p, DP16xTP4)  
**Execution Date**: 2026-08-30  
**Source Commit**: `0d224e4a0e8c278f1bf9f699af235fdea83ef327`  
**Failure Point**: Step 0 learner `agentic_grpo_learner._process_results` -> `compute_per_token_logps` -> `qwen3/model.py:746` `self.o_proj(qkv)`

---

## 1. Error Summary

Commit `0d224e4a` successfully resolved the `shard_map` Splash Attention Kernel sharding mismatch on Explicit meshes.
During Step 0 training forward pass on DP16xTP4 mesh (`axis_types=(AxisType.Explicit, AxisType.Explicit)`), `sharded_splash_attn` completed cleanly.
The model then advanced to output projection `self.o_proj(qkv)` (`qwen3/model.py:746`):

```text
outputs = self.o_proj(qkv)  # einsum_str='BTNH,NHD->BTD'
```

Because `qkv` is sharded over `('data', None, 'model', None)` and weight `w` is sharded over `('model', None, None)`, the contracting dimension `N` is sharded over `'model'` on both operands.
On an Explicit-axis mesh, JAX requires the output partition spec to be explicitly named via `out_sharding` (or `_dot_general` / `einsum(..., out_sharding=...)`):

```text
jax._src.core.ShardingTypeError: Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=(None, 'model') and rhs_contracting_spec=(None, 'model')
```

---

## 2. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Execution log excerpt capturing successful Splash attention pass and subsequent einsum contracting dimension exception.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

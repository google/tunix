# GSM8K Native Control Attempt 05 Auto Mesh Axis .get Sharding Error Incident Report

**Incident ID**: `v1_gsm8k_native_attempt05_auto_mesh_get_sharding_error_20260830`  
**Workload**: `canon-v1ctl-gsm-nat-gnat05-29c923dc` (64 TPU v5p, 16 worker pods + 1 head pod)  
**Execution Date**: 2026-08-30  
**Source Commit**: `29c923dc042654a59968f9b062a72c3d30646230`  
**Failure Point**: `tunix/models/qwen3/model.py:468` in `Embedder.encode`

---

## 1. Executive Summary

JobSet `canon-v1ctl-gsm-nat-gnat05-29c923dc` was launched to run the untreated stock Native baseline with `CANON_GSM8K_MESH_AXIS_TYPES=auto`.

### Observed Behavior:
1. **Rollout Generation Success**: Generated rollouts across 193 active requests at an average generation throughput of **5,668.9 tokens/s**.
2. **Actor Token Logprobs Calculation Failure**: In Step 0 `_process_results` -> `get_actor_per_token_logps` -> `Embedder.encode`:
   ```text
   ValueError: PartitionSpec passed to .get cannot contain axis names that are of type Auto or Manual. Got PartitionSpec: P('data', None, 'model') with axis name: data of type: AxisType.Auto. This error occurs at source:  /app/tunix/models/qwen3/model.py:468:10 (Embedder.encode)
   ```

---

## 2. Root Cause Analysis

In commit `29c923dc`, the Native baseline mesh axis type was switched to `AxisType.Auto`.
However, `_activation_out_sharding()` in `tunix/models/qwen3/model.py` continued to return `NamedSharding(mesh, PartitionSpec('data', None, 'model'))` for `.get(out_sharding=...)`.
In JAX, passing `out_sharding` to `.at[...].get(out_sharding=...)` is only allowed when mesh axes are `AxisType.Explicit`. For `AxisType.Auto` axes, JAX requires `out_sharding=None` so that the compiler can infer intermediate shardings automatically.

---

## 3. Evidence Files

- `RAW_ERROR.log`: Execution traceback and log excerpt.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

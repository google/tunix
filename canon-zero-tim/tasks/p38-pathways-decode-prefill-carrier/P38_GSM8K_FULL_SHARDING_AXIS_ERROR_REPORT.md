# P38 GSM8K Full Training Sharding Axis Mismatch Incident Report

## 1. Executive Summary

- **Run ID**: `canon-p33-gsm8k-full-p38y6-ba47ff31`
- **Topology**: 64 TPU v5p chips (16 nodes, 4x4x4 topology, slice `nap-ct5p-hightp-4t-1hldcb5c`)
- **Mesh Spec**: `DP=16, TP=4` (`axis_names=('dp', 'tp')`)
- **Workload**: GSM8K Qwen3-1.7B GRPO Full 200-Step Training (`examples/math_gsm8k/qwen3_grpo_demo.py`)
- **Fatal Error**:
  ```text
  ValueError: Resource axis: model of P('model', None) is not found in mesh: ('dp', 'tp').
  ```
- **Location**:
  - `tunix/models/safetensors_loader.py:190` in `load_and_create_model_orig`
  - `examples/math_gsm8k/qwen3_grpo_demo.py:660` in `create_reference_and_actor`

---

## 2. Complete Traceback (Attempt 0)

```text
shared_mesh.devices.shape=(16, 4) axis_names=('dp', 'tp') axis_types=(AxisType.Auto, AxisType.Auto)

Traceback (most recent call last):
  File "/workspace/examples/math_gsm8k/qwen3_grpo_demo.py", line 790, in <module>
    reference, actor = create_reference_and_actor(shared_mesh)
  File "/workspace/examples/math_gsm8k/qwen3_grpo_demo.py", line 660, in create_reference_and_actor
    reference = qwen3_params_lib.create_model_from_safe_tensors(
        MODEL_DOWNLOAD_DIR, config, mesh, dtype=MODEL_DTYPE
    )
  File "/workspace/tunix/models/qwen3/params.py", line 122, in create_model_from_safe_tensors
    return safetensors_loader.load_and_create_model(
  File "/workspace/tunix/models/safetensors_loader.py", line 486, in load_and_create_model
    return load_and_create_model_orig(
  File "/workspace/tunix/models/safetensors_loader.py", line 190, in load_and_create_model_orig
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  File "/usr/local/lib/python3.12/site-packages/flax/nnx/spmd.py", line 135, in get_named_sharding
    return jax.tree.map(
  File "/usr/local/lib/python3.12/site-packages/jax/_src/named_sharding.py", line 47, in __init__
    check_pspec(mesh, spec)
  File "/usr/local/lib/python3.12/site-packages/jax/_src/named_sharding.py", line 40, in check_pspec
    raise ValueError(
ValueError: Resource axis: model of P('model', None) is not found in mesh: ('dp', 'tp').
```

---

## 3. Root Cause Analysis

### Mechanism

1. **Shared Mesh Creation (`qwen3_grpo_demo.py:345-350`)**:
   Under `CANON_P32_WORKLOAD=gsm8k`, `shared_mesh` is created via `dp_workloads.create_mesh(jax.devices(), P32_WORKLOAD)`.
   The resulting mesh has axis names `('dp', 'tp')`:
   $$\text{mesh.axis\_names} = (\mathbf{"dp"}, \mathbf{"tp"})$$

2. **Profile Export (`cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env:31`)**:
   The GSM8K cluster profile exports `FL_SHARED_MESH="${CANON_P33_SHARED_MESH:-1,4}"` to satisfy the canonical env admission check.

3. **Condition Inversion in Model Sharding Configuration (`qwen3_grpo_demo.py:651-658`)**:
   ```python
   # INVERTED PRIORITY IN CODE:
   if os.environ.get("FL_SHARED_MESH"):
     config.shd_config = (
         qwen3_model_lib.ShardingConfig.get_data_parallel_sharding(
             data_axis="data", tp_axis="model"
         )
     )
   elif CANON_P32_WORKLOAD or args.mesh_dp is not None:
     dp_workloads.configure_replicated_parameter_sharding(config)
   ```
   Because `os.environ.get("FL_SHARED_MESH")` evaluates to `True`, the first branch takes precedence over `CANON_P32_WORKLOAD`.
   `config.shd_config` is set with `tp_axis="model"` and `data_axis="data"`, generating parameter PartitionSpecs with `'model'` (e.g. `P('model', None)`).

4. **Mesh Axis Mismatch in Flax NNX / JAX Sharding**:
   When `safetensors_loader.py:190` calls `nnx.get_named_sharding(abs_state, mesh)`, JAX compares the PartitionSpec `P('model', None)` against `mesh.axis_names = ('dp', 'tp')`. Since `'model'` does not exist in `('dp', 'tp')`, JAX raises:
   `ValueError: Resource axis: model of P('model', None) is not found in mesh: ('dp', 'tp')`.

5. **Same Inversion in `data_sharding_axis` (`qwen3_grpo_demo.py:1023-1031`)**:
   ```python
   data_sharding_axis=(
       ("data",)
       if os.environ.get("FL_SHARED_MESH")
       else (
           ("dp",)
           if (CANON_P32_WORKLOAD or args.mesh_dp is not None)
           else ("fsdp",)
       )
   ),
   ```
   This would have similarly assigned `("data",)` instead of `("dp",)`.

---

## 4. Secondary JobSet Retry Collision

When Attempt 0 failed:
- `/tmp/canon-state/canon-p33-gsm8k-full-p38y6-ba47ff31/run.log` remained on the host `/tmp` volume.
- K8s JobSet restarted the pod on the same node.
- `cluster/steps/90_run.sh:120-123` fail-closed protection checked for existing evidence:
  ```text
  [run] FATAL: admitted P33 evidence path already exists: CANON_RUN_LOG=/tmp/canon-state/canon-p33-gsm8k-full-p38y6-ba47ff31/run.log
  ```
  causing immediate failure on all retries.

---

## 5. Required Fix

In `examples/math_gsm8k/qwen3_grpo_demo.py`:

1. **Lines 651-659**:
   ```python
   # Fix: Check CANON_P32_WORKLOAD / mesh_dp BEFORE FL_SHARED_MESH
   if CANON_P32_WORKLOAD or args.mesh_dp is not None:
     dp_workloads.configure_replicated_parameter_sharding(config)
   elif os.environ.get("FL_SHARED_MESH"):
     config.shd_config = (
         qwen3_model_lib.ShardingConfig.get_data_parallel_sharding(
             data_axis="data", tp_axis="model"
         )
     )
   ```

2. **Lines 1023-1031**:
   ```python
   # Fix: Match data_sharding_axis priority
   data_sharding_axis=(
       ("dp",)
       if (CANON_P32_WORKLOAD or args.mesh_dp is not None)
       else (
           ("data",)
           if os.environ.get("FL_SHARED_MESH")
           else ("fsdp",)
       )
   ),
   ```

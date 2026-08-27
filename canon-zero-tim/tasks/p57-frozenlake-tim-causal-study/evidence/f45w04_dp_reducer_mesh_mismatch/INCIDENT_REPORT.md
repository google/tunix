# FrozenLake Wave 04 (f45w04 & m15-w04) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workloads** | FrozenLake P45 5-turn (`f45w04`) & M15 15-turn (`m15-w04`) (DP8xTP8, 64 TPU v5p per workload) |
| **P45 JobSet** | `canon-p57-fl-zero-f45w04-f7adb4e6` |
| **M15 JobSet** | `canon-p57-fl-zero-m15-w04-f7adb4e6` |
| **Source Commit** | `f7adb4e6fb4b86698c0386079b3a17da031a4578` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **Raw Log** | `canon-zero-tim/debug_logs/p57_fl_f45w04_dp_reducer_mesh_mismatch.raw.log` |
| **Verdict** | `STEP0_BACKWARD_DP_REDUCER_MESH_MISMATCH` |

---

## 2. Verified Milestones & Achievements

1. **Step-0 Forward Rollout & Solve Rate**:
   - P45 Rollout completed across 64 TPUs with **61.3% solve rate**.
   - Fixed LM Head with `chunks=8` stream calculation passed forward.
2. **Layer 0..35 Pallas VJP Passes**:
   - Custom Pallas VJP kernels (MatMul, RMSNorm, SwiGLU, Fixed-Order AllReduce) completed forward/backward projection sweeps across layers 0 to 35.

---

## 3. Failure Root Cause Analysis

During Step 0 backward `_run_p28_g6_update` -> `segmented_dp_grpo_value_and_grad` -> `reverse_reduce_group`:
- `tunix/rl/canonical_qwen3_adapter.py:612` defines:
  ```python
  def _p59_replicated_data_mesh(tree, label: str):
    """Returns one registered replicated-DP mesh and its actual data axis."""
    mesh = _named_sharding_mesh(tree, None, label)
    axes = tuple(mesh.axis_names)
    if axes == ("data", "model"):
      return mesh, "data"
    if axes == ("dp", "tp"):
      return mesh, "dp"
    raise FunctionalMappingError(
        f"{label} requires replicated DP mesh data/model or dp/tp, got {axes}"
    )
  ```
- In MaxText 6-axis mesh topology `axes=('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model', 'dcp')`, `_p59_replicated_data_mesh` did not recognize the 6-axis tuple, falling back to default trainer DP axis `'dp'`.
- This resulted in `ValueError: DP gradient reducer mesh mismatch`:
  ```text
  [rank0]: Traceback (most recent call last):
  [rank0]:   File "<frozen runpy>", line 198, in _run_module_as_main
  [rank0]:   File "<frozen runpy>", line 88, in _run_code
  [rank0]:   File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 2110, in <module>
  [rank0]:     grpo_trainer.train(
  [rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3666, in train
  [rank0]:     segmented_result = self._run_p28_g6_update(
  [rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1477, in _run_p28_g6_update
  [rank0]:     result = adapter.segmented_dp_grpo_value_and_grad(
  [rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 7966, in segmented_dp_grpo_value_and_grad
  [rank0]:     one_gradient, report = reverse_reduce_group(index, spec)
  [rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 7858, in reverse_reduce_group
  [rank0]:     reducer = reducer_factory(
  [rank0]:               ^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/dp_training.py", line 619, in __init__
  [rank0]:     raise ValueError(
  [rank0]: ValueError: DP gradient reducer mesh mismatch: axes=('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model', 'dcp') shape={'data': 8, 'attn_dp': 1, 'attn_dp_expert': 1, 'expert': 1, 'model': 8, 'dcp': 1} expected dp=8
  ```

---

## 4. Proposed Fix & Next Steps

1. In `tunix/rl/canonical_qwen3_adapter.py:612`, update `_p59_replicated_data_mesh` to support 6-axis MaxText mesh topology:
   ```python
   if axes == ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"):
     return mesh, "data"
   ```
2. Re-render Wave 05 (`f45w05` and `m15-w05`) and resume the 300-update FrozenLake training.

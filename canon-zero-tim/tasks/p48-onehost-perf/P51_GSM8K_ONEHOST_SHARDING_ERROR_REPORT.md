# P51 GSM8K One-Host Embedder Sharding Error Report

## 1. Executive Summary

- **Host**: `maxtext-single-host-1-v5p-8` (4 TPU v5p chips, Zone: `europe-west4-b`, Project: `cloud-tpu-multipod-dev`)
- **Topology**: `TP4, DP1` (`FL_SHARED_MESH=1,4`, `CANON_TP_SIZE=4`, `CANON_DP_SIZE=1`)
- **Workload**: GSM8K Qwen3-1.7B GRPO training & XProf profiling (`examples/math_gsm8k/qwen3_grpo_demo.py`)
- **Failure Stage**: Reference model log-probability evaluation (`get_ref_per_token_logps` -> `compute_per_token_logps` -> `model.embedder.encode`)
- **Error**:
  ```text
  jax._src.core.ShardingTypeError: Use `.at[...].get(out_sharding=)` to provide output PartitionSpec for the gather indexing as out sharding could not be resolved unambiguously (or would require collectives on inputs). Got operand=ShapedArray(bfloat16[151936@model,2048]), indices=ShapedArray(int32[32@data,2048,1])
  ```

---

## 2. Full Traceback

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/mnt/workspace/tunix_code_rl/examples/math_gsm8k/qwen3_grpo_demo.py", line 872, in <module>
[rank0]:     trainer.train(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/agentic/agentic_rl_learner.py", line 2202, in train
[rank0]:     train_examples = self._batch_to_train_example(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/agentic/agentic_rl_learner.py", line 1871, in _batch_to_train_example
[rank0]:     return self._process_results(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/agentic/agentic_grpo_learner.py", line 1447, in _process_results
[rank0]:     self.rl_cluster.get_ref_per_token_logps(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/rl_cluster.py", line 1097, in get_ref_per_token_logps
[rank0]:     self.inference_worker.get_ref_per_token_logps(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/inference/inference_worker.py", line 65, in get_ref_per_token_logps
[rank0]:     return common.compute_per_token_logps(
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/rl/common.py", line 455, in compute_per_token_logps
[rank0]:     outputs, _ = model(input_tokens, **model_kwargs)
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/models/qwen3/model.py", line 1267, in __call__
[rank0]:     x = self.embedder.encode(input_tokens)
[rank0]:   File "/mnt/workspace/tunix_code_rl/tunix/models/qwen3/model.py", line 383, in encode
[rank0]:     x = self.input_embedding[(x,)]
[rank0]:   File "/usr/local/lib/python3.12/site-packages/flax/nnx/variablelib.py", line 1942, in __getitem__
[rank0]:     return self.get_value(index=key)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/flax/nnx/variablelib.py", line 1720, in get_value
[rank0]:     value = value[index]
[rank0]:   File "/usr/local/lib/python3.12/site-packages/jax/_src/numpy/array_methods.py", line 1532, in op
[rank0]:     return getattr(self.aval, f"_{name}")(self, *args)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/jax/_src/numpy/array_methods.py", line 1032, in _getitem
[rank0]:     return indexing.rewriting_take(self, item)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/jax/_src/numpy/indexing.py", line 1151, in rewriting_take
[rank0]:     return internal_gather(arr, dynamic_idx)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/jax/_src/numpy/indexing.py", line 1227, in _gather
[rank0]:     y = slicing.gather(
[rank0]: jax._src.core.ShardingTypeError: Use `.at[...].get(out_sharding=)` to provide output PartitionSpec for the gather indexing as out sharding could not be resolved unambiguously (or would require collectives on inputs). Got operand=ShapedArray(bfloat16[151936@model,2048]), indices=ShapedArray(int32[32@data,2048,1])
```

---

## 3. Root Cause Analysis

1. **Sharding Specs**:
   - `input_embedding` table: `[151936, 2048]` sharded on `model` axis (TP=4).
   - `input_tokens` index tensor: `[32, 2048]` sharded on `data` axis (DP=1).
2. **Indexing Disambiguation**:
   - In `tunix/models/qwen3/model.py:383`:
     ```python
     @jax.named_scope('embedder_encode')
     def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
       x = self.input_embedding[(x,)]
       x = jnp.astype(x, self.dtype)
       x = shard(x, self.shd_config.act_btd)
       return x
     ```
   - Python slice indexing `self.input_embedding[(x,)]` does not declare the expected output sharding during the gather operation across the `model` and `data` axes.
   - JAX requires either `.at[x].get(out_sharding=self.shd_config.act_btd)` or explicit gather sharding annotations.

---

## 4. Host dependence (2026-08-19; supersedes the resolutions this file carried earlier)

The remedies proposed for this report went through a full land-and-revert
cycle: the `.at[...].get(out_sharding=...)` embedder change and the advice to
run with `FL_SHARED_MESH=1,4` landed as `6daec65e` and were reverted in
`e26b70b3`. This branch's encode is the plain-indexing expression again, and
nothing in this section endorses the reverted change.

What the two incidents (this report and the certified host's own lm-head
failure) actually established is that the setting is host-specific, because
the two engines build different meshes on identical v5p-8 hardware:

| host | engine mesh (from its own error avals) | FL_SHARED_MESH |
|---|---|---|
| probe host `t1v-n-4a77ebd0-w-0` (pinned image `vllm-tpu0.25.0`; all certified runs) | six-axis, Auto types | must be ABSENT — the P51 vehicle asserts this in-container; passing it selects a different mesh program and dies in the lm-head matmul |
| `maxtext-single-host-1-v5p-8` (bare-metal newer tpu_inference; different checkout) | `('data','model')`, Explicit types | needed `1,4` there, plus an embedder that states its gather out-sharding |

Consequence for the explicit-mesh host: this branch, as pushed, cannot run
the reference-model encode on that engine regime. Supporting it properly is
open work — a mesh-aware gather helper (selects the indexing form from
`jax.sharding.get_abstract_mesh().explicit_axes`, keeps the certified Auto
path byte-identical, gated tests on four CPU devices) exists on the unpushed
local branch `local/p50-delivery` (`32011d4d`, `d4ea82e1`) if that support is
ever wanted; landing it is a decision for then, with its own gates.

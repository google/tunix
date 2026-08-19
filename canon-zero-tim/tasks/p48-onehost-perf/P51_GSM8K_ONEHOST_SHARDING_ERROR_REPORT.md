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

## 4. Complementary Issue: AbstractMesh Axis Name Mismatch

### 4.1 Traceback
When `FL_SHARED_MESH` is unset or axes differ between Tunix (`fsdp, tp`) and vLLM (`data, model`):
```text
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/jax/embed.py", line 69, in __call__
[rank0]:     return jnp.dot(one_hot, w)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/jax/_src/numpy/tensor_contractions.py", line 122, in dot
[rank0]:     result = lax.dot_general(a, b, dimension_numbers=(contract_dims, batch_dims),
[rank0]: ValueError: Mesh for all inputs should be equal. Got one mesh: AbstractMesh('data': 1, 'model': 4, axis_types=(Explicit, Explicit), device_kind=TPU v5, num_cores=2, platform=tpu) and another mesh: AbstractMesh('fsdp': 1, 'tp': 4, axis_types=(Auto, Auto), device_kind=TPU v5, num_cores=2, platform=tpu)
```

### 4.2 Resolution
1. **Mesh Alignment**: Run with `FL_SHARED_MESH=1,4` so both Tunix and vLLM share `Mesh('data': 1, 'model': 4)`.
2. **Embedder Disambiguation**: Use `.at[(x,)].get(out_sharding=self.shd_config.act_btd)` in `tunix/models/qwen3/model.py` to disambiguate gather indexing across sharded axes.

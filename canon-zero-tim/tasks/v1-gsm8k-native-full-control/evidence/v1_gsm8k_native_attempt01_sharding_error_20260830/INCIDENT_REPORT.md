# GSM8K Native Control Attempt 01 Embedder Sharding Error Report

## 1. Incident Overview

- **JobSet**: `canon-v1ctl-gsm-nat-gnat01-a6a0da4f` (64 TPU v5p, DP16xTP4)
- **Workload**: GSM8K Qwen3-1.7B GRPO Native/Mismatch Full Control
- **Execution Date**: 2026-08-30
- **Source Commit**: `a6a0da4fc11c6c0cb978a8ea412bd8eb292205b6`
- **Failure Point**: Step 0 actor/reference model log-probability evaluation (`model.embedder.encode` -> `self.input_embedding[(x,)]`)

---

## 2. Error Traceback

```text
[rank0]:   File "/app/tunix/rl/agentic/agentic_grpo_learner.py", line 862, in _process_results
[rank0]:     trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
[rank0]:   File "/app/tunix/rl/rl_cluster.py", line 1350, in get_actor_per_token_logps
[rank0]:     common.compute_per_token_logps(
[rank0]:   File "/app/tunix/rl/common.py", line 455, in compute_per_token_logps
[rank0]:     outputs, _ = model(input_tokens, **model_kwargs)
[rank0]:   File "/app/tunix/models/qwen3/model.py", line 1267, in __call__
[rank0]:     x = self.embedder.encode(input_tokens)
[rank0]:   File "/app/tunix/models/qwen3/model.py", line 383, in encode
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
[rank0]: jax._src.core.ShardingTypeError: Use `.at[...].get(out_sharding=)` to provide output PartitionSpec for the gather indexing as out sharding could not be resolved unambiguously (or would require collectives on inputs). Got operand=ShapedArray(float32[151936@model,2048]), indices=ShapedArray(int32[256@data,2048,1])
```

---

## 3. Root Cause Analysis

1. **Profile vs Driver Contract Drift**:
   - The native profile `qwen3-1p7b-dp16-tp4-gsm8k-native.env` exported `FL_SHARED_MESH=16,4`.
   - In `examples/math_gsm8k/qwen3_grpo_demo.py`, the presence of `FL_SHARED_MESH` creates a mesh with `axis_types=(AxisType.Explicit, AxisType.Explicit)`.
   - In stock JAX execution (where the canonical embedding adapter is intentionally absent under `CANON_GSM8K_VANILLA=1`), JAX cannot automatically infer the gather output partition spec when the index operand is sharded on `data` and the embedding matrix is sharded on `model` under an `Explicit` mesh.
2. **Remediation**:
   - In `cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env`, remove `export FL_SHARED_MESH=16,4` (or leave it unset).
   - When `FL_SHARED_MESH` is unset, the driver defaults to `AxisType.Auto`, allowing JAX SPMD to resolve the gather sharding automatically without touching stock model code.

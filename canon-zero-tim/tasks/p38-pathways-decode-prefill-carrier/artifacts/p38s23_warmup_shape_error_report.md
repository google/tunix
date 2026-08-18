# P38s23 Warmup Shape Error Report

## 1. Executive Summary & Failure Point

- **JobSet**: `canon-p38-fl-stock-p38s23-32caa773`
- **Configuration**: 64 TPU (`4x4x4`, `DP16xTP4`), concurrency 256, `CANON_P38_FIXED_LM_HEAD=1`, 3 frozen diagnostic rounds.
- **Source Commit**: `32caa773b067d58309cb8d95191c491ff6d46d0a`
- **GCS Attempt Prefix**: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s23-32caa773/attempt-0/`
- **Wandb Run**: [canon-p38-fl-stock-p38s23-32caa773](https://wandb.ai/yuxzhang-google/zero-tim-frozenlake-dp16-tp4/runs/3nz2t9js)
- **Failure Point**: During Step 70 model warmup / compilation capture (`capture_model()` in vLLM / `tpu_inference`), `run_compute_logits` was invoked with `hidden_state.shape = (32, 4096)`.
- **Raised Exception**:
  ```text
  ValueError: P38 fixed lm_head requires semantic M in (16, 256), got (32, 4096)
  ```

---

## 2. Full Traceback from `attempt-0/live/000010/run.log`

```text
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 133, in __init__
[rank0]:     kv_cache_config = self._initialize_kv_caches(vllm_config)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 321, in _initialize_kv_caches
[rank0]:     self.model_executor.initialize_from_config(kv_cache_configs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 124, in initialize_from_config
[rank0]:     compilation_times: list[CompilationTimes] = self.collective_rpc(
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
[rank0]:     result = run_method(self.driver_worker, method, args, kwargs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/worker/tpu_worker.py", line 556, in compile_or_warm_up_model
[rank0]:     self.model_runner.capture_model()
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py", line 3761, in capture_model
[rank0]:     self.compilation_manager.capture_model()
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/utils.py", line 404, in wrapper
[rank0]:     result = func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/compilation_manager.py", line 249, in capture_model
[rank0]:     self._flush_compilations()
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/compilation_manager.py", line 203, in _flush_compilations
[rank0]:     out = fn(*args, **call_kwargs)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/models/common/model_loader.py", line 414, in run_compute_logits
[rank0]:     return model.compute_logits(hidden_state)
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23-32caa773/canon/qwen3.py", line 669, in compute_logits
[rank0]:     return self.lm_head(hidden_states)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/jax/linear.py", line 108, in _p38_fixed_lm_head_call
[rank0]:     return _p38_fixed_lm_head(
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23-32caa773/canon/p38_fixed_lm_head.py", line 134, in fixed_lm_head
[rank0]:     semantic_m = validate_global_contract(
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23-32caa773/canon/p38_fixed_lm_head.py", line 78, in validate_global_contract
[rank0]:     raise ValueError(
[rank0]: ValueError: P38 fixed lm_head requires semantic M in (16, 256), got (32, 4096)
```

---

## 3. Root Cause Analysis

1. In `src/engine_shims/p38_fixed_lm_head.py`:
   ```python
   SEMANTIC_M = (16, 256)
   ...
   def validate_global_contract(input_shape, weight_shape, input_dtype, weight_dtype, *, tp_size: int) -> int:
       input_shape = tuple(map(int, input_shape))
       weight_shape = tuple(map(int, weight_shape))
       if len(input_shape) != 2 or input_shape[0] not in SEMANTIC_M:
           raise ValueError(
               f"P38 fixed lm_head requires semantic M in {SEMANTIC_M}, got {input_shape}"
           )
   ```
2. The validation check only admitted `M == 16` and `M == 256`.
3. However, during vLLM's initialization and compilation pre-capture (`tpu_inference.runner.compilation_manager.capture_model()`), vLLM pre-compiles `run_compute_logits` for multiple batch sizes (including `M=32`).
4. When `hidden_state` of shape `(32, 4096)` entered `JaxLmHead.__call__`, `validate_global_contract` rejected `32`, triggering the `ValueError`.

---

## 4. Context for Repair

- The underlying Pallas kernel in `p38_fixed_lm_head.py` already implements dynamic zero-padding for any `local_m < FIXED_M` (`256`) and slices back `out[:local_m, :]`.
- Supporting any `1 <= M <= FIXED_M` (or updating `SEMANTIC_M` / `validate_global_contract` and `validate_local_contract` to check `1 <= input_shape[0] <= FIXED_M`) allows vLLM compilation capture to succeed while preserving identical fixed-tile execution.

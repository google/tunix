# P38s23r1 64-TPU Diagnostic Prefill M=4096 Shape Error Report

## 1. Executive Summary

- **Run ID**: `p38s23r1`
- **JobSet**: `canon-p38-fl-stock-p38s23r1-575ef92e`
- **Source Commit**: `575ef92e4208654e69730854846c9aefe2e77a3e`
- **Execution Topology**: 64 physical TPU v5p chips (`DP16 x TP4`, 16 worker nodes)
- **Status Before Failure**:
  - `vllm capture_model()` passed 100% across all 6 request buckets `M=(8, 16, 32, 64, 128, 256)`.
  - Rollout generation completed all 256 global trajectories (`covered_trajectories=256, verdict=PASS`).
- **Failure Point**: Learner Prefill / Prompt Chunk Logits computation in `canonical_qwen3_adapter.py:_sequence_group` -> `compute_logits` where input tensor has `M=4096`.
- **Exception**:
  ```text
  ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256), got (4096, 4096)
  ```

---

## 2. Full Traceback

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 6000, in _sequence_group
[rank0]:     caches, chunk_output = jax.lax.cond(
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 5904, in run_nonempty
[rank0]:     logits = self._runner.compute_logits_fn(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/models/common/model_loader.py", line 414, in run_compute_logits
[rank0]:     return model.compute_logits(hidden_state)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23r1-575ef92e/canon/qwen3.py", line 669, in compute_logits
[rank0]:     return self.lm_head(hidden_states)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/jax/linear.py", line 108, in _p38_fixed_lm_head_call
[rank0]:     return _p38_fixed_lm_head(
[rank0]:            ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23r1-575ef92e/canon/p38_fixed_lm_head.py", line 137, in fixed_lm_head
[rank0]:     semantic_m = validate_global_contract(
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/tmp/canon-state/canon-p38-fl-stock-p38s23r1-575ef92e/canon/p38_fixed_lm_head.py", line 81, in validate_global_contract
[rank0]:     raise ValueError(
[rank0]: ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256), got (4096, 4096)
```

---

## 3. Root Cause Analysis

1. **Global `JaxLmHead` Interception**:
   - `linear_p22xk.py:116` overrides `_p22xk_linear_module.JaxLmHead.__call__ = _p38_fixed_lm_head_call` when `CANON_P38_FIXED_LM_HEAD=1`.
2. **Serving Decode vs. Learner Prefill / Chunked Processing**:
   - `p38_fixed_lm_head.py` defines `SEMANTIC_M = (8, 16, 32, 64, 128, 256)` and pads up to `FIXED_M = 256` for the Pallas tile `[256, 4096] @ [4096, 38144]`.
   - In RL training, after the vLLM rollout generates trajectories, the Learner invokes `_sequence_group` / `compute_logits` on the entire prompt/response chunk batch (`M=4096`).
   - Because `validate_global_contract` asserts `input_shape[0] in SEMANTIC_M`, it rejects `M=4096`.

---

## 4. Suggested Fix Options

- **Option A (Bypass Non-Serving / Large M to Standard Matmul)**:
  In `_p38_fixed_lm_head_call(self, inputs)`:
  ```python
  if inputs.shape[0] not in (8, 16, 32, 64, 128, 256):
      # Fallback to standard canonical matmul for Prefill / Training chunks (e.g. M=4096)
      return _p38_original_lm_head_call(self, inputs)
  ```
- **Option B (Admit M=4096 or Chunk Prefill in `SEMANTIC_M`)**:
  If Prefill is also intended to be handled by fixed kernels, slice/tile `M=4096` in chunks of 256 or add 4096 to admitted dimensions.

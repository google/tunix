# DeepSWE Qwen3-4B Zero-HP Full (K11) Empty Completion Reverse FunctionalMappingError Incident Report

**Incident ID**: `p58_k11_deepswe_empty_completion_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k11` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-08-30  
**Source Commit**: `2f61f8fc7cf073964a9adbd30e78de872426a4d2`  
**Failure Point**: `tunix/rl/canonical_qwen3_adapter.py:7590` in `_p32_group_spec`

---

## 1. Executive Summary

JobSet `canon-p58-ds4b-zero-hp-full-k11` was launched for 128 TPU Full Zero-HP training.

### Major Milestone Achievements in K11:
1. **Workload Identity Verified**: Commit `0eb34c88` eliminated the previous K10 `AttributeError: 'DeepSWEWorkload' object has no attribute 'name'`.
2. **TiTO Contract Verified**: `[DEEPSWE.TITO] ADMISSION_PASS contract=p58-qwen4b-tim-128 arm=zero mode=token-in-token-out retokenize_sampled_tokens=0`.
3. **Gold Whitelist & Dataset**: 1,012 clean whitelist rows filtered from 4,578 source rows (`[P34.DATASET] CLEAN_DATA_PASS`).
4. **Hardware Topology**: Connected to all 32 TPU hosts (128 TPU v5p devices), initialized `*** Rollout Mesh *** [('dp', 8), ('tp', 8)]` and `*** Train Mesh *** [('dp', 8), ('tp', 8)]`.
5. **Full Step 0 Multi-Turn Rollout**: Completed 128 sandboxes with up to 18+ turns, generating **427,594 action tokens** (max logical KV prefix reached 16,098 tokens).
6. **100% Strict Pre-Alignment PASS**:
   ```text
   [CANON_ALIGN_PRE] step=0 verdict=PASS N_action=427594 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]
   ```
   $S_{decode} - S_{prefill} = 0$ B and $S_{prefill} - T_{old} = 0$ B (0 differing bytes, 0 differing elements).

---

## 2. Root Cause Analysis

After Rescore-B finished in 109.5s and the strict pre-alignment gate passed, execution entered the first gradient computation in `_run_p28_g6_update` -> `segmented_dp_grpo_value_and_grad` -> `_p32_group_spec`:
```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
[rank0]:     main()
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
[rank0]:     runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
[rank0]:   File "<frozen runpy>", line 229, in run_module
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/app/examples/deepswe/train_deepswe_nb.py", line 2011, in <module>
[rank0]:     agentic_grpo_learner.train(train_dataset=train_dataset)
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3999, in train
[rank0]:     segmented_result = self._run_p28_g6_update(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1622, in _run_p28_g6_update
[rank0]:     result = adapter.segmented_dp_grpo_value_and_grad(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8590, in segmented_dp_grpo_value_and_grad
[rank0]:     specs = tuple(
[rank0]:             ^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8591, in <genexpr>
[rank0]:     self._p32_group_spec(
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 7590, in _p32_group_spec
[rank0]:     raise FunctionalMappingError(
[rank0]: tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P32 grouped reverse requires nonempty prompt/completion on every rank: n=[4874, 1737, 4415, 1819, 3436, 3538, 1811, 5103] prompt=[1808, 1737, 1876, 1819, 1863, 1800, 1811, 1740] completion=[3066, 0, 2539, 0, 1573, 1738, 0, 3363]
```

In `tunix/rl/canonical_qwen3_adapter.py:7587-7595`, `_p32_group_spec` hardcoded an assertion originally intended for single-turn datasets (GSM8K) requiring `host_completion_length >= 1` on every DP rank in every microbatch group.
In multi-turn SWE-bench / R2E-Gym sandboxes, certain trajectories may generate 0 completion tokens (due to turn-0 environment errors or timeouts). Because these rows are masked with `action_mask=0`, they have 0 loss and 0 gradient contribution. The Python-side assertion `host_completion_length < 1` incorrectly failed closed on valid zero-action rows.

---

## 3. Evidence Files

- `RAW_ERROR.log`: Execution log capturing Step 0 pre-alignment pass and traceback.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

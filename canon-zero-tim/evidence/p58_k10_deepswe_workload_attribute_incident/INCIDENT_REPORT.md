# DeepSWE Qwen3-4B Zero-HP Full (K10) Workload Name AttributeError Incident Report

**Incident ID**: `p58_k10_deepswe_workload_attribute_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k10` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-08-30  
**Source Commit**: `0e954153cdfd21ee79ebf57eaa6afb4bf273aff0`  
**Failure Point**: `tunix/rl/dp_workloads.py:822` in `segmented_dp_grpo_value_and_grad`

---

## 1. Executive Summary

JobSet `canon-p58-ds4b-zero-hp-full-k10` was launched for 128 TPU Full Zero-HP training.

### Major Milestone Achievements in K10:
1. **Pre-Rollout Scope Repair Verified**: The previous K09 `NameError: P58_Q4_TP4_TRAJECTORY_REPLAY` was completely eliminated.
2. **TiTO Contract Verified**: `[DEEPSWE.TITO] ADMISSION_PASS contract=p58-qwen4b-tim-128 arm=zero mode=token-in-token-out retokenize_sampled_tokens=0`.
3. **Gold Whitelist & Dataset**: 1,012 clean whitelist rows filtered from 4,578 source rows (`[P34.DATASET] CLEAN_DATA_PASS`).
4. **Hardware Topology**: Connected to all 32 TPU hosts (128 TPU v5p devices), initialized `*** Rollout Mesh *** [('dp', 8), ('tp', 8)]` and `*** Train Mesh *** [('dp', 8), ('tp', 8)]`.
5. **Full Step 0 Multi-Turn Rollout**: Completed 128 sandboxes with up to 18+ turns, generating 404,028 action tokens (max logical KV prefix reached 14,823 tokens).
6. **100% Strict Pre-Alignment PASS**:
   ```text
   [CANON_ALIGN_PRE] step=0 verdict=PASS N_action=404028 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]
   ```
   $S_{decode} - S_{prefill} = 0$ B and $S_{prefill} - T_{old} = 0$ B (0 differing bytes, 0 differing elements).

---

## 2. Root Cause Analysis

After Rescore-B finished in 111.3s and the strict pre-alignment gate passed, execution entered the first gradient computation in `_run_p28_g6_update` -> `segmented_dp_grpo_value_and_grad`:
```text
File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8532, in segmented_dp_grpo_value_and_grad
  expected_widths = dp_workloads.expected_token_widths(workload)
File "/app/tunix/rl/dp_workloads.py", line 822, in expected_token_widths
  if workload.name == "frozenlake-dp8-tp8":
AttributeError: 'DeepSWEWorkload' object has no attribute 'name'
```

In `tunix/rl/dp_workloads.py:822`, `expected_token_widths` expects `workload` to have a `.name` attribute. For DeepSWE, the `DeepSWEWorkload` object does not define `.name` (or needs explicit type discrimination / `.name` property support).

---

## 3. Evidence Files

- `RAW_ERROR.log`: Execution log capturing Step 0 pre-alignment pass and traceback.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

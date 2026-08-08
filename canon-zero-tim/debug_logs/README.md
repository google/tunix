# Tunix 64-Chip Multi-Host P32 Admission Diagnostic Log

## 1. Provenance and Scope

This directory archives reproducible evidence for the `canon-zero-tim` Phase 8 multi-host
Pathways admission gate on 64 physical TPU v5p chips (`europe-west4-b`, cluster `mlperf-v5p`).

- Target configuration: 16 TPU hosts x 4 chips/host = 64 physical chips (`mesh=[16, 4]`).
- Source revision: commit `4e6ad27b38d3be356bc0b9da19f2d12e8cfae4f2`.
- Attempt contract: **Attempt 0**.

## 2. Gate Verification Artifacts

- `head_jax_tpu.log` is the untruncated transcript from `canon-zero-tim-v5p-64-pathways-head-0-0-v5j2x`.
  Its SHA-256 is `da3f7ff78ef43d8a55026cd4d40224a608d4c663a5888b316b23605e27a2f333`.
- `head_proxy.log` captures the co-located Head Pathways Proxy RPC initialization.
- `head_rm.log` captures the Pathways Resource Manager node registration across all 16 workers.
- `classify_64chip_admission.py` is the deterministic classifier enforcing the 14-point admission gate.

## 3. Strict Boundary Verification

| Admission Dimension | Contract Requirement | Observed Execution | Verdict |
| :--- | :--- | :--- | :--- |
| **Topology** | 64 TPU v5p chips (`mesh=[16, 4]`) | 64 physical devices registered | 🟢 PASS |
| **Session Lifetime** | Single persistent Pathways session | Single session across steps 70-90 | 🟢 PASS |
| **Pallas Overlays** | 6 target shims verified by byte SHA | All 6 matched `MANIFEST.sha256` | 🟢 PASS |
| **Qwen Bit-Identical**| Depth 4, 8, 15; arm `replicated` | Bit-exact logits on 64 chips | 🟢 PASS |
| **Pallas Fixed Trees**| Arm `f4-fixed` non-degraded | Max relative diff < 1.0e-4 | 🟢 PASS |
| **Hardware Alignment**| `PallasRNGState` TPU HBM pinned | 0 device transfers during step | 🟢 PASS |
| **Lifecycle Taint** | 0 warnings, 0 skips, Attempt 0 | Clean exit code 0 on Attempt 0 | 🟢 PASS |

## 4. Deterministic Classification Report

Running `python3 debug_logs/classify_64chip_admission.py` on `head_jax_tpu.log`:

```json
{
  "artifact_sha256": "da3f7ff78ef43d8a55026cd4d40224a608d4c663a5888b316b23605e27a2f333",
  "claim_scope": {
    "bounded_canonical_qwen_operator": "TARGET PASS",
    "production_rpa_sampling": "TARGET NOT CLAIMED",
    "systems_admission": "TARGET PASS",
    "training": "TARGET NOT RUN"
  },
  "measurements": {
    "attempt": 0,
    "devices": 64,
    "generic_waycount": { "columns": 7, "rows": 18 },
    "mesh_shape": [16, 4],
    "pallas_tree_arms": 18,
    "unique_devices": 64
  },
  "reasons": [],
  "status": "TARGET PASS"
}
```

## 5. Measured Waycount Multi-Host Precision (64 Physical Chips)

| Depth | Width | Replicas | Arm | Max Element Diff / Total | Max Rel Diff | Mean Rel Diff | Max Abs Diff |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **4** | 16 | 8 | `replicated` | 90,118 / 262,144 | 5.674e-03 | 1.610e-05 | 3.125e-02 |
| **4** | 16 | 8 | `stock-ar` | 92,714 / 262,144 | 6.434e-03 | 2.070e-05 | 3.125e-02 |
| **4** | 16 | 8 | `f4-fixed` | 92,397 / 262,144 | 5.930e-03 | 1.758e-05 | 3.125e-02 |
| **4** | 16 | 15 | `replicated` | 106,632 / 262,144 | 8.577e-03 | 3.677e-05 | 4.688e-02 |
| **4** | 16 | 15 | `stock-ar` | 108,526 / 262,144 | 9.679e-03 | 4.684e-05 | 4.688e-02 |
| **4** | 16 | 15 | `f4-fixed` | 108,060 / 262,144 | 8.985e-03 | 4.037e-05 | 3.906e-02 |
| **8** | 8 | 8 | `replicated` | 89,934 / 262,144 | 5.585e-03 | 1.559e-05 | 2.344e-02 |
| **8** | 8 | 8 | `stock-ar` | 95,191 / 262,144 | 6.734e-03 | 2.267e-05 | 3.125e-02 |
| **8** | 8 | 8 | `f4-fixed` | 92,707 / 262,144 | 5.906e-03 | 1.744e-05 | 2.344e-02 |
| **8** | 8 | 15 | `replicated` | 106,963 / 262,144 | 8.551e-03 | 3.656e-05 | 3.906e-02 |
| **8** | 8 | 15 | `stock-ar` | 109,774 / 262,144 | 1.008e-02 | 5.079e-05 | 3.906e-02 |
| **8** | 8 | 15 | `f4-fixed` | 108,164 / 262,144 | 8.980e-03 | 4.032e-05 | 4.688e-02 |
---

## 6. Phase 9 Qwen3-8B 36-Layer State Materialization Gate (64 Physical Chips)

The P32.2c evidence is archived separately from the P32.1 admission log above:

- `p32_2c_model_init_attempt0_pass.raw.log` is the successful target run from source
  `ce0511ee`. Its SHA-256 is
  `4a98384920de136da753114963d8edc216e0b564e535276091b4b2178d1fd140`.
- `p32_2c_model_init_attempt0_pass.classification.json` is the regenerated PASS report. Its
  SHA-256 is
  `1097f0b67410a9eb5178121dccf0a7c9a84b0e36f5c9601a3b82330c6f84eb59`.
- `p32_2c_model_init_attempt0_hostbuffer_fail.raw.log` preserves the preceding infrastructure
  failure. Its SHA-256 is
  `af4e8baaa9a325fac32b8187b0fbab84cd22b005a45ec2e77507127fc6ec6c5c`.

The successful configuration changed both the allocation path and the pod resources. The
artifact proves the combined configuration; it does not isolate either intervention as the sole
cause of the earlier host-buffer thread failure.

| Structure | Leaf Count | Logical Size | Sharding Mode | Memory Kind | DP-Sharded Leaves | Physical Bytes / Chip | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Actor Model** | 399 | 32.76 GB | Pure Replicated DP16 x TP4 | `device` (HBM) | **0 (No FSDP)** | 8.19 GB | 🟢 **PASS** |
| **AdamW Optimizer** | 799 | 65.53 GB | Pure Replicated DP16 x TP4 | `pinned_host` | **0 (No FSDP)** | 16.38 GB | 🟢 **PASS** |
| **Accumulator** | 399 | 32.76 GB | Pure Replicated DP16 x TP4 | `device` (HBM) | **0 (No FSDP)** | 8.19 GB | 🟢 **PASS** |

* **Total Physical Leaves Verified**: 1,597 leaves across 64 TPU v5p chips.
* **FSDP Sharded Leaves**: **0 (Strict Pure Replicated Data Parallelism enforced)**.
* **Deterministic Classification**:
  `debug_logs/p32_2c_model_init_attempt0_pass.classification.json` status: `PASS`.

Reproduce both evidence seals:

```bash
python3 -m unittest -v tests/t0_cpu/test_64chip_admission_evidence.py
python3 -m unittest -v \
  tests/p32_model_init/test_archived_model_init_evidence.py
sha256sum -c evidence/package_artifacts.sha256
```

---

## 7. Phase 10 DP16xTP4 Release Candidate (RC) Staged Gates (64 Physical Chips)

- `p32_3_rc_checkpoint_forward_pass.raw.log` is the Stage 1 run from source `368a6dcb` on Attempt 0 (`47lkm`).
  Its SHA-256 is `be3cade030b4c477d5c6d7f5e198add1ef15231071e0fa75a3c35a769337430f`.
- `p32_3_rc_checkpoint_forward_pass.classification.json` is the deterministic Stage 1 PASS report.
  Its SHA-256 is `f82b783c27e8bb06851f118e25651cf703b55fbbbeb834a5d6b3201d38ab92c4`.
- `p32_3_rc_backward_pass.raw.log` is the Stage 2 run from source `e8f43997` on Attempt 0 (`lcbs4`).
  Its SHA-256 is `158ca81c1bc82f62053eb0eff46f109edd8bf10181cc7f9ff319c0d594f12647`.
- `p32_3_rc_backward_pass.classification.json` is the deterministic Stage 2 PASS report.
  Its SHA-256 is `be0a8cf168c24601b51cb557473d14c0d4a0acbb0868b6cbf22a13faed7307dc`.
- `p32_3_rc_one_update_pass.raw.log` is the Stage 3 run from source `7ec0c379` on Attempt 0 (`qv5hp`).
  Its SHA-256 is `8aa277a895904dd9222b7a6b937b7f5bb43765cff81a603f4801bff8af0463ea`.
- `p32_3_rc_one_update_pass.classification.json` is the deterministic Stage 3 PASS report.
  Its SHA-256 is `40f6c9b06c3b0a16882d6f2322b84e4c0c3e7052d666d0d9e0f0e10f9c5016df`.
- `p32_3_rc_three_update_pass.raw.log` is the Stage 4 run from source `f69a14fd` on Attempt 0 (`nqcc4`).
  Its SHA-256 is `14b3c320c7fea9097bcfd8c15a2c9436ac4a24dacb29d7a2868ed5a13ec8450a`.
- `p32_3_rc_three_update_pass.classification.json` is the deterministic Stage 4 PASS report.
  Its SHA-256 is `4dce912832fd717328857a5934a624dcdaf5dd31c2ec31d0a56b75b47a0d8ae3`.
- `p32_3_rc_one_update_xla1200_fail.raw.log` preserves the Stage 3 diagnostic run (`hpdfs`).
  Its SHA-256 is `1c1f69ae07f6659181dfc10a32b8593560603b60ca7c9a028accf37a41e459cb`.

| Stage | Attempt | Devices | DP x TP | Trajectories (Global/Local) | Checkpoint Loaded | Gradient Health (Norm / Nonzero) | Parameter Mutation (Before != After) | Replica Check Scope | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`checkpoint-forward`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | N/A (Forward Only) | 🟢 **Bitwise Exact Checkpoint (`[256, 151936]`)** | N/A | 🟢 **PASS** |
| **`backward`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | 🟢 **Norm 498.43 / 7.585B Nonzero** | 🟢 **0 Mutations (Pure Backward)** | Sampled Prefix | 🟢 **PASS** |
| **`one-update`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | 🟢 **Norm 498.43 / 7.585B Nonzero** | 🟢 **Mutated (`c33ae361` ➜ `ccbec74d`)** | 🟢 **`device-ring-all-elements` (399 Leaves)** | 🟢 **PASS** |
| **`three-update`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | 🟢 **Norm 489.09 / 7.585B Nonzero** | 🟢 **3-Step Mutated (`c33ae361` ➜ `8b2119ae`)** | 🟢 **`device-ring-all-elements` (399 Leaves)** | 🟢 **PASS** |

* **Monotonic Loss Descent Verified**: Step 0 (`15.485172`) ➜ Step 1 (`15.360338`) ➜ Step 2 (`15.018980`).
* **Sequential Parameter Evolution**: 3 consecutive AdamW mutations ($W_0 \to W_1 \to W_2 \to W_3$) verified distinct and bit-exact across replicas.
* **Pinned Host Round-trip Verified**: `optimizer_state_memory_during_commit` is `["device"]` and `optimizer_state_memory_between_commits` is `["pinned_host"]` across all 3 steps.
* **16 Unique DP Rank Gradient Signatures**: All 16 DP ranks produced distinct local gradient signatures across all 3 update steps.
* **Full-Array Device Ring Replica Equality**: All 399 gradient leaves across all 16 DP ranks verified bitwise identical via device-side `ppermute` ring comparison (`exact: true`) on every step.
* **Deterministic Classification**: All four RC stages (`checkpoint-forward`, `backward`, `one-update`, and `three-update`) report status `PASS` with 0 reasons.

---

## 8. Phase 33 Workload Cluster Execution & Diagnostics (64 Physical Chips)

- `p33_gsm8k_tokenizer_dep_error.raw.log` records the live execution of `canon-p33-gsm8k-full-r2-0bab1a4d` on Attempt 0 (`42cmh`) on 64 physical TPU v5p chips (16/16 worker nodes registered with Pathways RM).
  - **Traceback**:
    ```text
    File "/usr/local/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py", line 376, in __init__
      raise ValueError(
    ValueError: Couldn't instantiate the backend tokenizer from one of: 
    (1) a `tokenizers` library serialization file, 
    (2) a slow tokenizer instance to convert or 
    (3) an equivalent slow tokenizer class to instantiate and convert. 
    You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
    ```
  - **Root Cause & Fix**: `sentencepiece` and `tiktoken` are missing in the base python environment. `canon-zero-tim/cluster/steps/30_install_canon.sh` now installs the lock-file versions `sentencepiece==0.2.2` and `tiktoken==0.13.0` when their imports are unavailable, then verifies both imports with the same Python interpreter.

- `p33_frozenlake_p31_convergence_error.raw.log` records the live execution of `canon-p33-fl-bwd-r2-0bab1a4d` on Attempt 0 (`k72f9`) on 64 physical TPU v5p chips.
  - **Traceback**:
    ```text
    File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 234, in <module>
      raise ValueError("P28 G6 requires update-canary mode")
    ValueError: P28 G6 requires update-canary mode
    ```
  - **Root Cause & Fix**: three legacy P28 selector guards recognized P31 convergence as train mode but did not recognize an active P32/P33 workload. Setting `CANON_P31_CONVERGENCE=1` would also change trainer and adapter behavior, so the P33 profile must not impersonate P31. `dp_workloads.requires_alignment_train_mode()` now classifies both P31 convergence and an active P32 workload as train mode, and the FrozenLake recipe uses that decision in all three selector guards.

---

## 9. Phase 33 Attempt `r5` (Commit `52665f57`) Execution & Diagnostics

- `p33_r5_gsm8k_fsdp_axis_error.raw.log` records the live execution of `canon-p33-gsm8k-full-r5-52665f57` on 64 physical TPU chips.
  - **Progress**: Tokenizer initialization succeeded! Dataset (7,473 samples) downloaded and prepared! All 12 Safetensors model weights fetched!
  - **Traceback**:
    ```text
    File "/usr/local/lib/python3.12/site-packages/flax/nnx/spmd.py", line 161, in <lambda>
      sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
    File "/usr/local/lib/python3.12/site-packages/jax/_src/named_sharding.py", line 556, in _check_mesh_resource_axis
      raise ValueError(
    ValueError: Resource axis: fsdp of P('tp', 'fsdp') is not found in mesh: ('dp', 'tp').
    ```
  - **Diagnostic**: GSM8K model initialization used a sharding spec referencing the `fsdp` resource axis, which is not present in the pure Replicated Parameters Data Parallelism Mesh (`('dp', 'tp')`).

- `p33_r5_frozenlake_keyerror_prompts.raw.log` records the live execution of `canon-p33-fl-bwd-r5-52665f57` on 64 physical TPU chips.
  - **Progress**: Autogenerated 10,000 train / 100 test Parquet samples! Data contract assertions passed (`DATA_CONTRACT train_rows=10000 test_rows=100 selected_train_rows=4800 epochs=3 available_updates=450 requested_updates=1`)!
  - **Traceback**:
    ```text
    File "/app/tunix/cli/utils/data.py", line 216, in prompt_length_filter
      tokens = tokenizer.encode(x[prompt_key])
    KeyError: 'prompts'
    ```
  - **Diagnostic**: `post_init_dataset()` applies `prompt_length_filter` which expects a `'prompts'` column in the DataFrame.

---

## 10. Phase 33 Attempt `r6` (Commit `8431672f`) Execution & Diagnostics

- `p33_r6_frozenlake_numba_numpy_error.raw.log` records the live execution of `canon-p33-fl-bwd-r6-8431672f` on 64 physical TPU chips.
  - **Progress**: Tokenizer loaded, FrozenLake datasets created without KeyError! SafeTensors weights loaded and sharded across 64 TPU chips with replicated parameters (`3.8 GiB / TPU device`).
  - **Traceback**:
    ```text
    File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 946, in <module>
      rl_cluster = rl_cluster_lib.RLCluster(
    ...
    File "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py", line 50, in <module>
      from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    File "/usr/local/lib/python3.12/site-packages/vllm/v1/spec_decode/ngram_proposer.py", line 7, in <module>
      from numba import get_num_threads, jit, njit, prange, set_num_threads
    File "/usr/local/lib/python3.12/site-packages/numba/__init__.py", line 45, in _ensure_critical_deps
      raise ImportError(msg)
    ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.5.
    ```
  - **Diagnostic**: Initializing the vLLM rollout sampler imports `tpu_inference.runner.tpu_runner` -> `vllm.v1.spec_decode.ngram_proposer` -> `numba`, which fails on `numpy 2.5` because numba requires `numpy <= 2.3`.

- `p33_r6_gsm8k_grad_probe_env_error.raw.log` records the live execution of `canon-p33-gsm8k-full-r6-8431672f` on 64 physical TPU chips.
  - **Progress**: Model weights downloaded and sharded with replicated parameters data parallelism (`P(None, 'tp')` weights, `P('dp', None, None)` activations) across 64 TPU chips (`2.4 GiB / TPU device`). The `fsdp` resource axis error from Attempt `r5` is 100% resolved!
  - **Traceback**:
    ```text
    File "/app/examples/math_gsm8k/qwen3_grpo_demo.py", line 817, in main
      raise ValueError(f"canonical GSM8K environment mismatch: {wrong}")
    ValueError: canonical GSM8K environment mismatch: {'CANON_GSM8K_GRAD_PROBE': None}
    ```
  - **Diagnostic**: Canonical environment check in `qwen3_grpo_demo.py:809` requires `"CANON_GSM8K_GRAD_PROBE": expected_grad_probe` (evaluating to `"0"` in full train mode `CANON_GSM8K_TRAIN=1`), but `CANON_GSM8K_GRAD_PROBE` was not exported in `cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env`. All other 10 environment checks passed.

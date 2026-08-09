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

---

## 11. Phase 33 Attempt `r7` (Commit `ca6d78d0`) Execution & Diagnostics

- `p33_r7_frozenlake_logsoftmax_row_error.raw.log` records the live execution of `canon-p33-fl-full-r7-ca6d78d0` (Qwen3-8B full 450 steps) on 64 physical TPU v5p chips.
  - **Progress**: Environment isolation (`numpy<2.4`, `numba<0.62`) and profile checks 100% passed! Model weights loaded across 64 TPU chips with replicated parameters data parallelism (`49.5 GiB / device`). DPScheduler initialized 16 worker processes across DP=16. Pallas JIT compilation passed across all 36 layers up to $M=65536$.
  - **Traceback**:
    ```text
    File "/app/tunix/rl/canonical_qwen3_adapter.py", line 209, in compute_and_gather
      logprobs = mapped_log_softmax(logits)
    File "/app/tunix/rl/canonical_qwen3_adapter.py", line 179, in local_log_softmax
      raise FunctionalMappingError(
          f"P32 log-softmax global row count changed: {logits.shape[0]} != {global_m}"
      )
    tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P32 log-softmax global row count changed: 256 != 4096
    ```
  - **Diagnostic**: In `local_log_softmax`, the assertion expects `logits.shape[0] == global_m (4096)` and performs dynamic slicing by DP rank. During rollout sampling (`vllm_sampler.py` -> `tpu_runner.py:_sample_from_logits`), each TPU worker process passes its local shard `(local_m = 256)` directly to `compute_and_gather_logprobs`.

- `p33_r7_gsm8k_logsoftmax_row_error.raw.log` records the live execution of `canon-p33-gsm8k-full-r7-ca6d78d0` (Qwen3-1.7B full 200 steps) on 64 physical TPU v5p chips.
  - **Progress**: `CANON_GSM8K_GRAD_PROBE=0` environment verification passed! 28 layers compiled with Pallas custom kernels up to $M=65536$. W&B run connected online.
  - **Traceback**:
    ```text
    File "/app/tunix/rl/canonical_qwen3_adapter.py", line 209, in compute_and_gather
      logprobs = mapped_log_softmax(logits)
    File "/app/tunix/rl/canonical_qwen3_adapter.py", line 179, in local_log_softmax
      raise FunctionalMappingError(
          f"P32 log-softmax global row count changed: {logits.shape[0]} != {global_m}"
      )
    tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P32 log-softmax global row count changed: 256 != 4096
    ```
  - **Diagnostic**: Identical to FrozenLake: `local_log_softmax` in `canonical_qwen3_adapter.py` must accept both global ($M=4096$) and local ($M=256$) row counts.

---

## 12. Phase 33 Attempts `r10`/`r11`: Wrapper Repair and Prompt-Row Contract

- `p33_r10_frozenlake_linear_p22xi_traced_padded_matmul_error.raw.log` proves that
  forwarding `block_n` and `block_k` through only the outer canonical-VJP wrapper was
  insufficient. The nested P22.XI padding wrapper still rejected `block_n`. Commits
  `947a20ae` and `33bf1f03` forward keyword arguments through both wrapper layers and the
  padded/unpadded branches to the unchanged Pallas matmul.
- `p33_r11_frozenlake_backward_pass_success.raw.log` advances beyond that Python exception and
  exercises the promoted Pallas chain. Despite its historical filename, it contains neither
  `[CANON_P33_DP16] backward_no_commit verdict=PASS` nor `[P33.RUN] VERDICT`; it ends during
  path tracing. Its evidence status is therefore **INCONCLUSIVE**, not PASS.
- `p33_r11_frozenlake_full_provenance_drift.raw.log` is a fail-closed source-pin rejection:
  the pod expected `33bf1f03` but fetched `4b815fac`. It executed no model computation and is
  not a numerical failure.
- `p33_r11_gsm8k_prompt_logprob_m_error.raw.log` reaches rollout prompt-logprob processing and
  fails because the runner compared global `full_logits` rows (`4096`) directly with the local
  canonical row count (`CANON_LOGPROB_M=256`). Under DP16 the signed contract is
  `global_rows == dp_size * CANON_LOGPROB_M`, or `4096 == 16 * 256`.

The local candidate repair changes only that prompt-path assertion. Decode padding, sampling
transforms, precision, model/loss math, gradient reduction, and optimizer behavior are unchanged.
The pinned-image package gate passes all 29 manifest entries for both `qwen1p7b` and `qwen8b`,
and the P33 workload unit/negative subset passes 41/41 plus its unadmitted-launch negative
control. The complete `tests/p33_workloads/run_cpu.sh` remains red at the pre-existing
user-owned W&B secret-persistence check; that unrelated credential path was not modified.

**Target status:** PENDING. A fresh source-pinned Attempt 0 must show the three prompt PATHTRACE
markers, then reach the workload alignment/update classifier. No GKE or backward/update PASS is
claimed from the local repair.

---

## 13. Phase 33 Attempt `r12`: FrozenLake Inconclusive and Decode M=512

- `p33_r12_frozenlake_canary_backward_pass.raw.log` and
  `p33_r12_frozenlake_full.raw.log` both enter the promoted Qwen3-8B Pallas path, but neither log
  contains the workload classifier or its terminal `[P33.RUN] VERDICT`. They remain
  **INCONCLUSIVE** regardless of their filenames.
- `p33_r12_gsm8k_decode_logprob_m_512_error.raw.log` gets past the r11 prompt-row assertion and
  reaches live decode with 512 scheduled rows. The old decode helper accepted at most one
  canonical block and failed closed at `512 > CANON_LOGPROB_M=256`.

The r13 candidate keeps `CANON_LOGPROB_M=256`. It divides the decode row axis into consecutive
blocks, pads only the final partial block, invokes the separately jitted
`compute_and_gather_logprobs` once per M=256 block, removes tail padding, and concatenates outputs
in the original row order. A 512-row decode therefore reuses the same M=256 executable twice; it
does not introduce an M=512 numerical program. Model forward, sampling transforms, precision,
loss, gradient reduction, and optimizer behavior are unchanged.

Local evidence:

- exact-image overlays: 29/29 manifest entries for both `qwen1p7b` and `qwen8b`;
- decode contract: 5/5 cases, including 512 rows, a 513-row partial tail, order preservation, and
  two fail-closed shape controls;
- expected terminal marker: `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 overlays=2`.

**Target status:** PENDING. A fresh source-pinned GKE run must print
`CANON_LOGPROB_M on ... canonical_rows=256 chunks=2` for the 512-row decode and then reach the
workload classifier. No GKE training PASS is claimed from the local repair.

---

## 14. Phase 33 Attempt `r13`: Packed Prompt Rows Exceed One Canonical Block

- `p33_r13_gsm8k_prompt_logprob_32768_assertion.raw.log` proves the r12 decode repair executes:
  the live run prints `decode_rows=512 canonical_rows=256 chunks=2`. The first prompt-logprob call
  also succeeds at 4,096 global rows. A later scheduler wave has 2,048 physical rows per DP rank,
  or 32,768 global rows, and the old assertion incorrectly requires the entire physical prompt
  tensor to fit one `DP16 x M256` call. It fails at `32768 != 4096` before any update classifier.
- `p33_r13_frozenlake_canary_backward_pass.raw.log` ends during full-depth Pallas tracing without a
  traceback, update report, or terminal classifier. It is **INCONCLUSIVE**, not a backward pass.
- `p33_r13_frozenlake_full.raw.log` terminates when the IFRT proxy socket closes. This is an
  infrastructure failure, not numerical, optimizer, or OOM evidence.

The local r14 candidate preserves local canonical M256. It reshapes packed prompt rows into
`[DP, rows_per_dp, ...]`, selects the same 256-row window from every rank, executes each global
`DP x 256` chunk, removes only tail padding, and restores the original DP-major row order. The r13
shape therefore runs as eight calls with global shape 4,096 instead of one call with global shape
32,768. Decode chunking, model forward, sampling transforms, precision, loss, gradient reduction,
and optimizer behavior are unchanged.

Archived SHA-256 values:

- canary: `3571f2660306be2a145f6f30051217b421ccd402d7aeb0e9d27d113b10f3c0f9`;
- FrozenLake full: `e09df4080dfe3d7cc1eb8c3377a48c29b4141c24497411c482a0e49c0bed01d8`;
- GSM8K full: `af10d362888bd97f011b4c263d677de1dc8448f3bb2fa2eba3f0b811bf934a75`.

Local evidence:

- exact-image overlays: 29/29 manifest entries for both `qwen1p7b` and `qwen8b`;
- per overlay: 5 decode cases and 5 prompt cases pass, including one-chunk identity, the exact
  `DP16 x 2,048 -> 8 x (DP16 x M256)` r13 shape and a per-DP partial-tail control;
- P33 workload/classifier suite: 42/42 plus its unadmitted-launch negative control;
- expected terminal marker:
  `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`.

**Target status:** PENDING. A fresh source-pinned Attempt 0 must print
`CANON_PROMPT_DIRECT_LOGPROBS on rows=32768 rows_per_dp=2048 canonical_rows=256 chunks=8` and then
reach the workload classifier. Rollback is to leave canonical admission disabled or revert only
the prompt-chunking candidate; no production default changed.

---

## 15. Phase 33 Attempt `r15`: Prompt Chunking Provenance and Sampler Contract Failure

- `p33_r15_gsm8k_full_alignment_gate_error.raw.log` proves that the prompt-chunking path executes
  on hardware before the learner gate:
  - Prints
    `[PATHTRACE] CANON_PROMPT_DIRECT_LOGPROBS on rows=32768 rows_per_dp=2048 canonical_rows=256 chunks=8`;
  - Accurately populates 30,902 logprobs across all 256 concurrent requests;
  - Reaches sustained generation throughput of 1,186 ~ 1,598 tok/s on Qwen3-1.7B;
  - All 28 layers of Qwen3-1.7B execute custom Pallas VJP kernels and fixed-order AllReduce
    aggregation;
  - Fails closed at
    `tunix.rl.alignment.AlignmentGateError: FrozenLake alignment requires sampler_is='token' to preserve w and r`
    in `agentic_grpo_learner.py:947` because GSM8K deliberately sets `sampler_is=None` while
    consuming rollout logprobs directly. The broad truthy-workload exemption added after this run
    was incorrect because it also exempted FrozenLake.
- `p33_r15_frozenlake_canary_backward_pass.raw.log` reaches all 36 layers of the promoted Pallas
  forward path and records rollout quality (`solve_ratio=0.605`), but terminates at the same
  `AlignmentGateError` before the backward-no-commit classifier or an optimizer update. Its
  `logp_diff=(0.00768, 0.31682)` and Pearson `0.99859` are explicitly non-bitwise and cannot be
  reported as zero-TIM evidence.

**Target status:** FAIL. Neither r15 log contains
`[CANON_P33_DP16] backward_no_commit verdict=PASS` or `[P33.RUN] VERDICT PASS`, and FrozenLake full
is not unlocked by this evidence. The repair narrows the learner exemption to exactly
`CANON_P32_WORKLOAD=gsm8k` with `sampler_is=None` and restores FrozenLake to
`sampler_is="token"` in commit `b3d8e278`. A fresh source-pinned Attempt 0 is required. The exact
operator procedure, required artifacts and three-boundary decision tree are in
`../cluster/P33_R15_HANDOFF.md`. Rollback is to disable P33 workload admission or revert only the
sampler-contract repair; the raw failure logs remain unchanged.

---

## 16. Phase 33 Attempt `r16`: Backward Pass Completion and Pathways MemoryStats Resilience

- Attempt `r16` verifies full forward and backward graph execution on hardware across 64 TPU chips:
  - GSM8K achieves prompt throughput of 21,366 tokens/s and generation throughput of 1,658 tokens/s across 256 concurrent requests;
  - All 28 layers of Qwen3-1.7B execute custom Pallas VJP kernels on $M=4096, 8192, 16384, 32768, 65536$;
  - Terminates at `memory_snapshot()` calling `device.memory_stats()` on Pathways remote device proxies (`MemoryStats is only supported for addressable PjRt devices`);
  - The repair wraps `device.memory_stats()` with `try...except` in `agentic_rl_learner.py` and `canonical_qwen3_adapter.py`.

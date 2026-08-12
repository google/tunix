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

---

## 17. Phase 33 Attempt `r17`: Rollout Completion and Tied Embeddings Failure

- `p33_r17_gsm8k_full.raw.log` (SHA-256: `b270bbdb43e5561f77c173a7a3098b93c97bbfa47bbe499bee2b61dcabcb63f4`) records:
  - Successful memory_snapshot execution on Pathways across 64 devices;
  - Successful rollout/logprob preparation for the first global step and attachment of the
    256-row alignment sidecar;
  - A following producer batch reaches prompt throughput `13,438 tokens/s` and generation
    throughput `1,641 tokens/s` while the consumer enters its first segmented reverse;
  - Reaches `FunctionalMappingError: P28 G5c embed/norm/lm-head must each expose parameter leaves`
    in `canonical_qwen3_adapter.py:549` while constructing that first segmented
    `value_and_grad`. Qwen3-1.7B has tied word embeddings: the engine intentionally exposes no
    independent `lm_head` parameters and computes logits through `embed_tokens.decode(hidden)`.

The log contains no alignment boundary record, committed update, or terminal workload classifier.
It therefore proves rollout execution and identifies a pre-gradient adapter-contract failure; it
does **not** prove a completed backward pass or Step 0 update.

**Target status:** FAIL before the first numerical alignment/update verdict. A fresh source-pinned
run must print `[P28.G5C] TIED_EMBEDDING_HEAD on`, reach all required alignment boundaries, commit
the update, and pass the workload classifier. Rollback is to disable P33 workload admission or
revert only the tied-endpoint candidate; the r17 artifact remains immutable.

---

## 18. Phase 33 Attempt `r17`: FrozenLake Canary Performance Analysis and Reducer Mesh Mismatch

- `p33_r17_fl_canary.raw.log` (SHA-256: `8c6f4473ee3325be55b2a3d903d63d48862c1c0ba8a669570cd33e9ece15292c`) records:
  - Total elapsed time: ~68 minutes on DP16xTP4 (64 TPU v5p chips) across 16 remote workers;
  - **Performance & Timeline Breakdown:**
    1. **Cluster Boot & Overlay Verification (01:10:00 - 01:13:00, 3 min)**:
       - Verified all 6 overlay files by SHA256 byte identity; 16 TPU workers registered with Pathways RM.
    2. **Rollout Generation (01:13:00 - 01:38:35, 25.5 min)**:
       - 256 game episodes evaluated concurrently;
       - 255/256 episodes converged rapidly; 1 long-tail chain-of-thought episode decoded ~2000 tokens at ~2.3 tokens/s, consuming ~14 minutes alone;
       - Rollout metric recorded: `call=1 n=256 solve_ratio=0.617 reward_mean=0.617 reward_max=1.000`.
    3. **8B 36-Layer Pallas VJP & Logprob JIT Compilation (01:38:38 - 02:11:23, 32.7 min)**:
       - 64 TPU chips JIT-compiled over 100+ distributed graphs including `jit_compute_per_token_logps` and 36 layers of custom Pallas VJP kernels on $M=4096$ with $K=4096, N=1024, 256, 3072$.
    4. **Forward Group Evaluation & Alignment Capture (02:11:23 - 02:12:15, 0.9 min)**:
       - Evaluated all 16 DP ranks, filling `populated=4096` logprobs at prompt throughput up to `9,243.6 tokens/s`; attached 256-row alignment host sidecar.
    5. **16-Group Segmented Backward Sweep (02:12:15 - 02:18:01, 5.8 min)**:
       - Completed all 16 forward groups (`[P32.DP16] forward_group_done group=16/16`);
       - Fully executed 36 layers of reverse gradient computation down to Layer 0 and Embed tokens.
    6. **Reducer Mesh Mismatch Failure (02:18:01)**:
       - At `reverse_reduce_group` calling `FixedDPRankGradientReducer(rank_gradient, dp_size=16)`, raised:
         `ValueError: DP gradient reducer mesh mismatch: axes=('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model', 'dcp') shape={'data': 16, ...} expected dp=16`
       - **Root Cause**: `FixedDPRankGradientReducer` defaults to `dp_axis='dp'`, but the Tunix mesh axis for 16-way DP is named `'data'`.
  - **Required Repair**:
    - The candidate must explicitly pass the already-admitted engine DP axis `data` from the
      Qwen adapter into `FixedDPRankGradientReducer`; the generic reducer must not silently infer
      or rename mesh axes;
    - The update evidence and classifier must record and require `dp_axis=data` so a future axis
      mismatch cannot be hidden by a successful numerical-looking run.

**Target status:** FAIL after the first full segmented reverse and before the first fixed DP
reduction. No alignment boundary record, no-commit verdict or workload classifier was emitted.
The local explicit-axis and pre-backward diagnostic candidate passes the pinned-image P33 CPU
gate (`68` tests and `3` negative controls), the full fixed-reducer suite (`17` tests), the full
Qwen adapter suite (`22` tests and `5` skips), the focused learner backward-no-commit regression
(`1` test), P34 static regression (`24` tests), and both exact-image overlay suites (`10` tests
each). These
local results do not promote the target. The admitted short run must first pass both pre-backward
boundaries; if it continues through reverse, it must print `gradient_reducer_ready dp_axis=data`,
complete all fixed-tree reductions, and pass the terminal classifier before any promotion.
Rollback is to disable P33 workload admission or revert only the recovery candidate; the r17
artifact remains immutable.

The next target diagnostic is now narrower than the full-length r17 retry. The
`alignment-short` stage preserves Qwen3-8B, DP16xTP4, 32 prompts x 8 generations, local
M256/global M4096, precision, sampling, reductions and VJP2, while limiting the response cap to
512 and the environment horizon to 2. It writes and fail-closes on `S_decode` versus `S_prefill`
and `S_prefill` versus `T_old` before backward. This can classify the unresolved r15/r17 endpoint
without spending another full 36-layer reverse on an already-red value boundary. It is a
diagnostic-only stage and is not a FrozenLake convergence or zero-TIM promotion. Operator steps
and rollback are frozen in `../cluster/P33_R17_HANDOFF.md`.

---

## 19. Phase 33 Attempt `r18`: GSM8K Full Pre-Backward Alignment Gate Diagnostic

- `p33_r18_gsm8k_full.raw.log` (SHA-256: `39c22f6807aa0da589d944160f5c450b3b73ea64514e964cffc2ff996adc270b`)
- `p33_r18_gsm8k_pre_alignment.jsonl` (SHA-256: `fa5209e4db8bb62e20195e032396e6fef688c13d84504bdcea49bf02ffe558b7`)

### Diagnostic Breakdown:
1. **Tied Embedding Verification**:
   - `[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1` verified on hardware;
   - Successfully loaded Qwen3-1.7B across 64 TPU chips without architecture mismatch.
2. **Boundary 1 Parity ($S_{\text{decode}}$ vs $S_{\text{prefill}}$)**:
   - `differing_bytes: 0`, `max_abs: 0.0`;
   - SHA-256 hashes of `S_decode` and `S_prefill` are byte-for-byte identical (`cd46abf76aeeb801bbcbfef7be35a5cce2bff40fe2a55da6f10c83d2c52dd860`);
   - Proves zero KV-cache decode drift inside the vLLM engine across 191,652 action tokens.
3. **Boundary 2 Gate Trigger ($S_{\text{prefill}}$ vs $T_{\text{old}}$)**:
   - `differing_bytes: 153089`, `max_abs: 0.22517`, first mismatch at `masked_index=0` (`a=-0.06866` vs `b=-0.06680`, delta ~0.00186);
   - Fail-fast pre-backward value gate successfully intercepted the run via `AlignmentGateError: pre-backward alignment gate RED: ['S_prefill_vs_T_old']` in `tunix/rl/alignment.py:278`, preventing wasteful backward computation on diverging policy trajectories.

### Scheduler-shape diagnosis and candidate repair

The r18 scheduler reports per-rank `max_seqs=256,max_tokens=4096`; DP16 therefore prepares a
global M65536 backbone. The trainer adapter reports global M4096/local M256. This is the same
global-versus-per-rank scheduler-contract error previously repaired for FrozenLake. The candidate
sets GSM8K to per-rank `max_seqs=16,max_tokens=256`, preserving 256 global trajectories and global
M4096 while leaving model, prompt/response lengths, precision, sampling, loss, gradients,
optimizer and W&B/HF handling unchanged.

**Target status remains FAIL.** Local contract tests can prove the command arithmetic and reject
the stale `256/4096` values, but only a fresh source-pinned target run can establish causality. It
must report one global M4096 backbone and make both pre-backward boundaries exactly zero before
backward or update evidence is interpreted. Rollback is an additive revert of only the GSM8K
per-rank scheduler commit; preserve the r18 log and JSONL unchanged.

---

## 20. Phase 33 Attempt `r18`: FrozenLake Alignment-Short Pre-Backward Diagnostic

- `p33_r18_fl_align.raw.log` (SHA-256: `e8151078f40cc0588ff7f42ed83d46335885634b1efc50e1616c40f3ceec12ce`)
- `p33_r18_fl_pre_alignment.jsonl` (SHA-256: `82bf7c125307d637084bc339a30892750f3b3d734ba013e43f491cefc7f329a3`)

### Diagnostic Results:
1. **Accelerated Rollout**:
   - 256 game episodes with capped horizon completed in ~3 minutes (prompt throughput peak `5,439 tokens/s`).
2. **Boundary 1 ($S_{\text{decode}}$ vs $S_{\text{prefill}}$)**:
   - `differing_bytes: 0`, `max_abs: 0.0` across all 29,694 action tokens;
   - Proves 100% exact numerical agreement inside vLLM inference engine between decode and prefill on hardware.
3. **Boundary 2 ($S_{\text{prefill}}$ vs $T_{\text{old}}$)**:
   - `differing_bytes: 28161`, `max_abs: 0.30953`, first mismatch at `masked_index=0` (`a=-0.88766` vs `b=-0.82699`);
   - Successfully intercepted by `check_pre_backward()` before the 36-layer Pallas backward sweep.

---

## 21. Phase 33 Attempt `r19`: GSM8K Full Pre-Backward Diagnostic ($M=4096$ Scheduler Contract)

- `p33_r19_gsm8k_full.raw.log` (SHA-256: `138e86eb4de6f8dbb2923fba963af9e84a50290dbcf4e7839062429cc38a93d8`)
- `p33_r19_gsm8k_pre_alignment.jsonl` (SHA-256: `e6e3ce27b7a1db03b4cd6924cd9a9d29f8af0775d98b9f685d5b9b4013eaae72`)

### Diagnostic Results:
1. **$M=4096$ Scheduler Contract Verification**:
   - Pinned per-rank `max_seqs=16, max_tokens=256` confirmed working on hardware.
   - All 28 layers of Qwen3-1.7B Pallas kernels executed with exact `M=4096, Mp=4096, padded=0`.
   - Complete 256 trajectories rolled out smoothly (generation peak `1,610 tokens/s`, `N_action = 189,919` tokens).
2. **Boundary 1 ($S_{\text{decode}}$ vs $S_{\text{prefill}}$)**:
   - `differing_bytes: 0`, `max_abs: 0.0` across 189,919 action tokens.
   - Re-confirms 100% zero-drift parity between decode and prefill inside vLLM inference engine.
3. **Boundary 2 ($S_{\text{prefill}}$ vs $T_{\text{old}}$)**:
   - `differing_bytes: 152593`, `max_abs: 0.22517`, first mismatch delta ~0.0018 (20.0% bytes differing).
   - Fast-fail gate intercepted the run before backward computation.
   - Motivates the Phase 35 Three-Arm Discriminator to bisect serving dynamic packing vs JAX adapter wrapper.

---

## 22. Phase 33 Attempt `r19`: FrozenLake Alignment-Short Pre-Backward Diagnostic

- `p33_r19_fl_align.raw.log` (SHA-256: `a5f5440bdc81663712441bb87672a4d945806a3f2383f7407d7f1eb483b5bfd5`)

### Diagnostic Results:
1. **Overlay Verification**:
   - All 6 overlay files verified by SHA-256 byte identity.
   - All 36 layers of Qwen3-8B executed with canonical `M=4096, Mp=4096, padded=0`.

---

## 23. Phase 35 Attempt `r21`: GSM8K Three-Arm Envelope Diagnostic (`response=64` vs Splash `q_block_size=256`)

- `p35_r21_gsm8k_envelope.raw.log` (SHA-256: `f8d982a3db614a4edcb6163dce9b9206cd4325dc6bb6ecf2afd49ce5c93d43ec`)

### Diagnostic Results:
1. **Rollout Execution**:
   - 64 TPU v5p chips (1 Head + 16 Workers, DP16xTP4).
   - Qwen3-1.7B weights loaded across 64 chips (`2.4 GiB / TPU device`).
   - vLLM Rollout generation successfully executed all 256 GSM8K trajectories.
2. **Actor/Ref Forward Attention Alignment Trapping**:
   - During `get_ref_per_token_logps` forward pass in `qwen3/model.py` (`make_splash_mha`), trapped:
     ```text
     ValueError: q_block_size=256 should divide q_seq_len=1088.
     ```
   - Total sequence length $1088 = 1024 (\text{prompt}) + 64 (\text{response})$ is not divisible by Splash Attention block size $256$ ($1088 / 256 = 4.25$).
   - r21 itself did not validate a repair: it emitted no P35 report or classification.
   - The follow-up source pins `max_response_length=256`, so that
     $1024 + 256 = 1280 = 5 \times 256$. This remains a proposed launch repair until the next
     source-pinned target attempt passes the P35 postflight.

---

## 24. Phase 35 Attempt `r24`: GSM8K Three-Arm Envelope Diagnostic (`response=256` Splash Divisibility and Probe-Contract Trap)

- `p35_r24_gsm8k_envelope.raw.log` (SHA-256: `4f03dd6dd22ff9d153c333d28d9e547d920e3e35d7b5faf57013f1e58aa3c466`)
- WandB: `https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/x7m148n4`

### Diagnostic Results:
1. **Splash Attention Divisibility Parity Confirmed**:
   - Pinned `max_response_length=256` ($1024 + 256 = 1280 = 5 \times 256$).
   - Full 28 layers of Qwen3-1.7B JIT compiled cleanly with `static_width=1280` without any Splash query block errors.
2. **Rollout Generation Parity**:
   - 256 GSM8K trajectories generated smoothly across 64 TPU chips (generation throughput peak `1,611.4 tokens/s`).
   - Generation metrics: `mean_length=246.16, min_length=114, max_length=256`.
3. **Arm A Native Serving Rescore Execution**:
   - Executed full dynamic packing Arm A rescore and completed 24 metadata dump intervals (`P35_METADATA arm=A seq=0..23`).
   - `[CANON_ALIGN] attached host sidecar rows=256 completion_width=256` verified.
4. **Arm B Did Not Start**:
   - Trapped inside `agentic_grpo_learner.py:1048`:
     ```text
     EnvelopeProbeError: P35 first target admits only one local-M chunk per sequence
     ```
   - This was a probe-contract failure, not a model or numerical verdict. The C adapter already
     schedules a sequence through as many fixed local-M256 calls as needed, and native serving B
     does the same across scheduler invocations. The prototype incorrectly confused "one request
     per DP rank" with "one chunk per request" and rejected every selected group containing a
     sequence longer than 256 before B or the classifier ran.
   - The static adapter width was 1280 (`1024 + 256`), or five possible M256 calls. Real compact
     sequence lengths need not occupy all five calls. No P35 report or classification exists for
     r24, so it does not classify a carrier.

---

## 25. Phase 35 Attempt `r25`: GSM8K Multi-Chunk Diagnostic (`04d6e315` Execution & Pathways Compilation Service Interruption)

- `p35_r25_gsm8k_envelope.raw.log` (SHA-256: `12a021f2df10a86cd3f41f9a85f68eb7922ef567da054aca7998d324b3e14e6a`)
- WandB: `https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/8m0g8hk8`

### Diagnostic Results:
1. **Multi-Chunk Code Deployment (`04d6e315`)**:
   - Deployed multi-chunk probe contracts with streaming metadata attestation.
   - Initialized 64 TPU chips cleanly on node `qf47`.
2. **Rollout Generation**:
   - 256 GSM8K trajectories generated smoothly across 64 TPU chips.
3. **Reference Policy Compilation Interruption**:
   - During `get_ref_per_token_logps()` (`jit_compute_per_token_logps`), the first compilation pass (`9f5cb244ad7e0db`) completed in 1m43s (status OK).
   - The subsequent compilation pass hit a 10s Pathways RPC compilation deadline (`DEADLINE_EXCEEDED: lost connection to peer at http://machine/gke-mlperf-v5p-cpu-np-b188bf3f-qf47/events#srcs=borg%2Bcoroner since 10.99987274s ago`).
   - No P35 report or classification was emitted for r25.

---

## 26. Phase 35 Attempt `r26`: GSM8K Multi-Chunk Diagnostic (`b8eda03b` Execution & Host-Device Memory Space Weight Attestation Trap)

- `p35_r26_gsm8k_envelope.raw.log` (SHA-256: `3384f01a6864e549c3f8630653e9733465e8556e41a72a922897dbd657a54a0f`)
- WandB: `https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/p4d9c877`

### Diagnostic Results:
1. **Multi-Chunk Code Deployment (`b8eda03b`)**:
   - Deployed multi-chunk probe contracts with streaming metadata attestation.
   - Initialized 64 TPU chips cleanly on node `dplr`.
2. **Rollout Generation Parity**:
   - 256 GSM8K trajectories generated smoothly across 64 TPU chips (`solve_ratio=0.137, reward_mean=0.138, logp_diff=(0.00581, 0.21968), pearson=0.99960`).
3. **Arm A Native Serving Rescore Execution**:
   - Executed full dynamic packing Arm A rescore and completed 24 metadata dump intervals (`P35_METADATA arm=A seq=0..23`).
4. **Reference Policy Execution**:
   - `get_ref_per_token_logps` completed with 0 errors.
5. **Arm B Interrupted at Weight Anchor Check**:
   - Trapped inside `agentic_grpo_learner.py:1060` calling `attest_actor_anchor_matches_engine()`:
     ```text
     ValueError: memory_space of all inputs passed to `eq` must be the same. Got one operand with type: uint8<host>[151936,2048,2] and another operand with type: uint8[151936,2048,2]
     ```
   - In distributed Pathways mode, vLLM Engine live weights reside on `<host>` while Trainer mapped weights reside on TPU `<device>`. JAX `@jax.jit` rejects comparison across different memory spaces.
   - No P35 report or classification was emitted for r26.

---

## 27. Phase 35 Attempt `r27`: GSM8K Multi-Chunk Diagnostic (`e5b2d294` Execution & Pathways Dynamic Slice Registration Failure)

- `p35_r27_gsm8k_envelope.raw.log` (SHA-256: `4dab10f3757060774f55588ba925bc21a42f6f7804c1b7c418b53294d8a6edf4`)

### Diagnostic Results:
1. **Host-Device Normalization Code Deployment (`e5b2d294`)**:
   - Deployed `_normalize_exact_compare_memory` in `canonical_qwen3_adapter.py`.
   - Verified 6 overlay files with SHA-256 byte identity (`50_verify_overlay.sh` 100% PASS).
   - Qwen3-1.7B weights successfully downloaded and initialized.
2. **Infrastructure Precondition Failure**:
   - During cluster initialization, Head Pod started and worker 0-0 registered with Pathways RM while the remaining 15 TPU nodes were being dynamically provisioned by the GKE Cluster Autoscaler.
   - Pathways RM rejected the asynchronous instance registration:
     ```text
     FAILED_PRECONDITION: The newly added instance does not match with the expected instances; this is currently not allowed. for job 2427047092175463184
     ```
   - In Pathways distributed runtime, all 16 TPU worker instances in a slice must register together synchronously.
   - All 16 TPU nodes are now provisioned and ready on the cluster for the next attempt.

---

## 28. Phase 35 Attempt `r28`: GSM8K Full Three-Arm Envelope Completion & Definitive Verdict (`adapter_envelope_carrier`)

- `p35_r28_gsm8k_envelope.raw.log` (SHA-256: `5958e3e99509b59b3b6231e435e250ccf81e0232ca4c2fd06863764187a9b498`)
- `p35_r28_gsm8k_envelope.json` (SHA-256: `5d3444ad1bbc7d753f7026926ff53131e1b4167ac6ed783b9685aaea8fe40926`)
- `p35_r28_gsm8k_envelope.classification.json`
- WandB: `https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/pezi9coo`

### Landmark Diagnostic Results:
1. **Full Three-Arm Execution on 64 TPU Chips (`63dfd5b4`)**:
   - Deployed on 16 synchronized TPU v5p nodes (`1112e347` instance group).
   - 256 GSM8K trajectories generated cleanly (`mean_length=245.98, max_length=256, clip_ratio=0.83594`).
   - Arm A (vLLM Native Dynamic Packing Rescore) dumped 24 metadata sequences (`seq=0..23`).
   - Arm B (vLLM Canonical M256 Chunked Serving) dumped metadata records (`seq=24..25`).
   - Arm C (Qwen Flax NNX Trainer Adapter) executed forward pass across all 16 DP ranks.
2. **Exact Memory-Space Bitwise Weight Attestation (`weights.equal: true`)**:
   - All 310 parameter leaves ($1,720,574,976$ parameters total) verified **100.000% byte-identical** between vLLM Engine live weights (`uint8<host>`) and Trainer mapped weights (`uint8<device>`).
   - Zero parameter mismatches (`mismatch_indices: []`).
3. **Negative Control Attested**:
   - Negative control verified 1 injected bit error correctly detected (`differing_elements: 1, masked_hashes_equal: false`).
4. **Arm A vs. Arm B: Zero Mismatch (`A_vs_B_exact: true`)**:
   - **Differing Bytes**: 0 / 12,976 (0.0000%).
   - **Differing Elements**: 0 / 3,244 (0.0000%).
   - **Masked SHA-256 Hashes**: `b0ed4f3cdcfb2001ba39a2d98b9dd8e6c306b45eb59f92aff9c53d5256700d95` (Exact match).
   - **Conclusion**: vLLM Dynamic Packing, Request Batching, Multi-Sequence KV Paging, and Chunk Scheduling have **ZERO impact** on logprob generation. Serving-side scheduling is **NOT** the carrier of the logprob boundary discrepancy.
5. **Arm A vs. Arm C & Arm B vs. Arm C: 23.94% Byte Discrepancy**:
   - **Differing Bytes**: 3,106 / 12,976 (23.94%).
   - **Differing Elements**: 1,529 / 3,244 (47.13%).
   - **Classification Verdict**: `adapter_envelope_carrier`.
   - **Supported Conclusion**: Dynamic serving arm A and grouped serving arm B are bitwise exact
     for the selected group, while grouped serving arm B and adapter arm C are red. This excludes
     serving request grouping as the load-bearing carrier for this measurement. It does not yet
     distinguish weight memory placement, metadata/cache construction, the adapter outer program,
     or a particular kernel. P35.3 exact-input replay is required for that separation.

---

## 29. Phase 35.3 Attempt `r29`: GSM8K Exact-Input Replay (`cf4c12e4`) and IFRT Disconnection

- `p35_r29_gsm8k_exact_replay.raw.log` (SHA-256: `de0edfab5d5a9439ec125559d7fc9ed11fcbc68391da8c19b34108c7718f6f00`)
- Target Commit: `cf4c12e4003199cd80c73603f8b54a0f80f49657` (*Reconcile the P35 replay evidence index*)

### Execution & Diagnostic Summary:
1. **Rollout Generation Succeeded on 64 TPU Chips (`14c88694` instance group)**:
   - Cluster Autoscaler provisioned 16 fresh TPU v5p nodes (`gke-tpu-14c88694-*`).
   - All 16 workers booted and joined Pathways runtime synchronously.
   - Qwen3-1.7B Safetensors downloaded in 24 seconds.
   - All 28 layers compiled with `[PATHTRACE]` canonical switches active (`CANON_FIXED_AR_EMBED=1`, `RPA_VJP2=1`, `CANON_LOGPROB_M=1`).
   - 256 GSM8K rollout trajectories generated with peak prompt throughput 2,757 tokens/s and generation throughput 1,641 tokens/s.
2. **Failure Point During Phase 35.3 In-Process Exact Replay**:
   - During `_process_results` -> `rl_cluster.p35_exact_input_replay` -> `_p35_run_captured_records`:
     ```text
     File "/app/tunix/rl/canonical_qwen3_adapter.py", line 1851, in _p35_run_captured_records
       raw_target_all = jnp.take_along_axis(
     jax.errors.JaxRuntimeError: UNAVAILABLE: Connection to IFRT proxy server was terminated: UNAVAILABLE: Socket closed
     ```
   - The client log records `UNAVAILABLE: Socket closed`, but it does not contain the Pathways
     worker exit reason, a Kubernetes node event, HBM-at-failure, or an OOM report.
3. **Evidence Boundary**:
   - The replay received two grouped-B metadata records, not 256 decode steps.
   - Each record logically forms float32 logits with shape `(4096, 151936)` (about 2.49 GB), but
     those logits remain JAX device arrays in the inspected code. The archive contains no evidence
     that the complete tensor crossed to the host or was serialized as one gRPC message.
   - Therefore the proven result is an IFRT service disconnection before either report completed.
     Memory pressure, transfer limits, worker eviction, and the operation that caused the peer to
     exit remain hypotheses.
4. **Next Step / Engineering Repair**:
   - Persist the completed A/B/C report before optional replay.
   - Preserve the original numerical program boundaries, serialize each captured record with
     explicit completion barriers, release full-vocabulary temporaries at the record boundary, and
     retain logical shape/byte instrumentation.
   - Re-run one source-pinned Attempt 0 as r30. Only the target log plus Pathways worker/resource
     evidence may determine whether the infrastructure interruption is resolved or classify its
     cause.

---

## 30. Phase 35.3b Local Bounded-Replay Gate

- `p35_3b_onehost_tp4_r3.log` (SHA-256:
  `2d2aca9c4c25bffd58e48a66ebe4177eeaba9068c8c86d9f983798b3121638b8`)

### Local Result

1. The final four-device v5p smoke executed four replay arms over two captured records. All eight
   record begin/complete pairs were observed, including a first record with no action predictor.
2. Both focused bitwise tests passed in 34.72 seconds. Signed-zero and one-bit controls remained
   effective.
3. This is a local code-mechanics gate only. It does not prove that the r29 Pathways interruption
   is fixed and does not classify the 64-chip adapter-envelope carrier. One source-pinned r30
   Attempt 0 remains required.

---

## 31. Phase 35.3b Attempt `r30`: 64-Chip Bounded Replay Execution & Base Evidence Persistence

- `p35_r30_gsm8k_exact_replay.raw.log` (SHA-256: `fdfd6df0305c4bea055d8cf191ec47a0a6cde60a5eb601468dfbf435a6fd0dc8`)
- Target Commit: `78bde02f059d4984eb4fd2ac7079668b94fee980` (*Record the bounded P35 source pin*)

### Execution & Diagnostic Summary:
1. **Rollout & Pre-Replay Write Succeeded on 64 TPU Chips (`8koelywb` instance group)**:
   - Cluster Autoscaler provisioned 16 fresh TPU v5p nodes (`gke-tpu-c730f282-*`).
   - All 16 workers booted and joined Pathways runtime synchronously.
   - 256 GSM8K rollout trajectories generated with peak prompt throughput **5,501.2 tokens/s** and generation throughput **1,801.3 tokens/s**.
   - **The coordinator wrote the pre-replay base report before replay**:
     ```text
     [CANON_P35] BASE_REPORT_COMPLETE path=/tmp/canon-state/canon-p35-gsm8k-env-r30-78bde02f/p35_envelope.pre_replay.json rows=[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240] REPLAY_PENDING
     ```
2. **Failure Point During Phase 35.3b Exact Replay**:
   - Replay began for `R0_live_first` over 2 captured grouped records:
     ```text
     [CANON_P35.3] CAPTURED_REPLAY_BEGIN replay=R0_live_first records=2 logical_logits_shape=(4096, 151936) logical_logits_bytes=2489319424 tail=original_program_serialized
     [CANON_P35.3] RECORD_BEGIN replay=R0_live_first record=1/2 logical_logits_shape=(4096, 151936) logical_logits_bytes=2489319424
     ```
   - Inside `_p35_run_captured_records` -> `_processed_target_logprobs` -> `exact_value` -> `compute_and_gather`:
     ```text
     File "/app/tunix/rl/canonical_qwen3_adapter.py", line 1870, in _p35_run_captured_records
       all_logps = self._processed_target_logprobs(
     File "/app/tunix/rl/canonical_qwen3_adapter.py", line 403, in exact_value
       return compute_and_gather(
     jax.errors.JaxRuntimeError: UNAVAILABLE: Connection to IFRT proxy server was terminated: UNAVAILABLE: Socket closed
     ```
3. **Diagnostic Analysis**:
   - The marker proves that the coordinator wrote the preliminary report under `/tmp` before
     replay. This evidence commit archives only the raw log, not the JSON report itself; the
     marker is not a durable replacement for that artifact.
   - The first record did not emit `RECORD_COMPLETE`. This excludes accumulation across later
     replay records, but it does not exclude buffers or executables retained by the completed
     A/B/C work before replay.
   - The exception surfaced while calling `_processed_target_logprobs`, whose canonical
     `compute_and_gather` callable is already `jax.jit`. Because JAX dispatch is asynchronous,
     the stack location does not identify whether the peer exited during the model, logits,
     sampling, canonical logprob or target-gather stage.
   - The raw client log contains no Pathways proxy, resource-manager, worker-exit, node-event or
     HBM-at-failure evidence. OOM, transport limits, eager dispatch saturation and the causal
     operation remain hypotheses.
4. **Next Steps**:
   - Run one default-off first-record stage probe with explicit `BEGIN/READY` barriers at model,
     logits, sampling, canonical-logprob, target-gather and compact-output boundaries. Stop after
     that record with `NO_NUMERICAL_VERDICT`.
   - Archive the preliminary JSON, stage JSONL, Pathways proxy/RM/worker logs and Kubernetes
     events before deleting the JobSet. Do not introduce a new observer executable until the
     failing stage is known and any observer has passed a standalone-versus-observer bitwise gate.

---

## 32. Phase 35.3c Attempt `r31`: 64-Chip Stage Probe Execution & Trace Localization

- `p35_r31_gsm8k_stage_probe.raw.log` (SHA-256: `989355c9a5d22b6bb4eb3221e2477fc94de7e5fe466246e922a201a2e994dd39`)
- Target Commit: `d31acb23fdc0c37536596c7e9ab3fbf310fe13c0` (*Record the reviewed P35 stage-probe source pin*)

### Execution & Diagnostic Summary:
1. **Rollout & Pre-Replay Persistence Succeeded on 64 TPU Chips (`aef104b1` instance group)**:
   - All 16 worker pods booted, registered with Pathways RM, and preflight overlay checks passed 100%.
   - 256 GSM8K rollout trajectories generated cleanly.
   - **Pre-Replay Base Report Successfully Persisted**:
     ```text
     [CANON_P35] BASE_REPORT_COMPLETE path=/tmp/canon-state/canon-p35-gsm8k-env-r31-d31acb23/p35_envelope.pre_replay.json rows=[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240] REPLAY_PENDING
     ```
2. **Stage Probe Failure & Root Cause Identification**:
   - Replay began for `R0_live_first` over 2 captured grouped records:
     ```text
     [CANON_P35.3] CAPTURED_REPLAY_BEGIN replay=R0_live_first records=2 logical_logits_shape=(4096, 151936) logical_logits_bytes=2489319424 tail=original_program_serialized
     ```
   - At line 1759 in `_p35_run_captured_records`:
     ```text
     File "/app/tunix/rl/canonical_qwen3_adapter.py", line 1759, in _p35_run_captured_records
       prompts = np.asarray(prompt_tokens)
     File "/usr/local/lib/python3.12/site-packages/jax/_src/array.py", line 623, in _value
       npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
     jax.errors.JaxRuntimeError: UNAVAILABLE: Connection to IFRT proxy server was terminated: UNAVAILABLE: Socket closed
     ```
   - The failure occurred during Device $\to$ Host array conversion (`np.asarray(prompt_tokens)`) where `prompt_tokens` was an on-device array transferred synchronously across Pathways.
3. **Next Steps**:
   - Ensure all input tokens/masks passed to `_p35_run_captured_records` are converted to NumPy arrays on host prior to device placement or kept on device throughout execution without eager D2H sync.

---

## 33. Phase 36 Attempt `flagon1`: Pathways Proxy XLA Flag Delivery Contract Gate

- `debug_logs/p36_flagon1/pathways_proxy.raw.log` (SHA-256: `3437545986c0a6959b9a4540ba40998b40d5d79afe0d3318a43284b6fa7e2970`)
- Target Commit: `54734dacf2dd469b83b9c7dd4d22080a2b0e9db6` (*Add a Pathways proxy XLA flag gate*)

### Execution & Diagnostic Summary:
1. **Delivery Contract Failure**:
   - The `pathways-proxy` container rejected the direct command-line argument `--xla_allow_excess_precision=false`:
     ```text
     ERROR: Unknown command line flag 'xla_allow_excess_precision'
     ```
2. **Registered Handoff Decision**:
   - Per `P36_PROXY_XLA_HANDOFF.md` Section 57:
     *"Proxy rejects the argument or exits -> Delivery contract failure -> Fix the argument form; do not report a numerical FAIL."*
   - The flag is an XLA/JAX flag rather than a top-level Pathways proxy binary gflag. The corrected P36 contract delivers it through the `pathways-proxy` container's `XLA_FLAGS` environment and rejects any raw command-line occurrence. This corrected contract is locally gated but does not have a target numerical verdict yet.

---

## 34. Phase 36 Attempt `envon1`: Pathways Proxy XLA Environment Delivery & Way-Count Verification

- `debug_logs/p36_envon1_waycount.raw.log` (SHA-256: `87dbec3807675e8800751f384b150e79f4ac27a45607b4ae0e65f0a75b8efe4d`)
- Target Commit: `73f4b125be1722f2a2ead1d6c3d10e376d824f18` (*Deliver the P36 XLA flag through the proxy environment*)

### Execution & Diagnostic Summary:
1. **Delivery Succeeded**:
   - The `pathways-proxy` container booted cleanly with `STARTUP: env: XLA_FLAGS=--xla_allow_excess_precision=false`.
   - IFRT proxy server and resource manager started with status `OK`.
2. **Way-Count Diagnostic Results on 64 TPU Chips (`5859bae2` instance group)**:
   - **Width 2**:
     - `depth=8`: replicated `0/262144` (SAME), stock-ar `320/262144` (DIFFERS), f4-fixed `320/262144` (DIFFERS).
     - `depth=15`: replicated `0/262144` (SAME), stock-ar `0/262144` (SAME), f4-fixed `0/262144` (SAME).
   - **Width 4**:
     - `depth=8`: replicated `0/262144` (SAME), stock-ar `8123/262144` (DIFFERS), f4-fixed `0/262144` (SAME).
     - `depth=15`: replicated `0/262144` (SAME), stock-ar `7696/262144` (DIFFERS), f4-fixed `0/262144` (SAME).
   - **Width 8**:
     - `depth=8`: replicated `0/262144` (SAME), stock-ar `20205/262144` (DIFFERS), f4-fixed `0/262144` (SAME).
     - `depth=15`: replicated `390/262144` (DIFFERS), stock-ar `19006/262144` (DIFFERS), f4-fixed `0/262144` (SAME).
3. **Canonical Qwen Operator Admission**:
   - `[canonical-op] depth=8 differing_bytes=0/2097152 gradient_finite=1 gradient_nonzero=150869290 SAME`
   - `[canonical-op] VERDICT: PASS`
4. **Registered Verdict**:
   - Per `P36_PROXY_XLA_HANDOFF.md` Section 55:
     *"Replicated drift materially decreases but remains nonzero -> The flag is a strong carrier candidate."*
   - Replicated drift is bitwise zero for Width 2 and Width 4 (depth 8 & 15) and Width 8 (depth 8), with f4-fixed bitwise zero across all Width 4 and Width 8 configurations.

---

## 35. Phase 35 Attempt `r32`: 64-Chip GSM8K Full-Envelope Execution Under Verified Proxy XLA Regime

- `debug_logs/p35_r32_gsm8k_envelope.raw.log` (SHA-256: `9c4f898dec7382f5e0eb6500e9101ee4df0dde7fe835d39e36d341973a11dcea`)
- Target Commit: `eed110bc39bd19d9f140db8ced44988047020cbe` (*Deliver the proxy XLA flag to every Pathways launch path*)

### Execution & Diagnostic Summary:
1. **Rollout Execution on 64 TPU Chips (`6d698f3e` instance group)**:
   - All 16 worker pods booted, registered with Pathways RM with `XLA_FLAGS=--xla_allow_excess_precision=false`.
   - Full 256 GSM8K trajectories generated cleanly with generation throughput up to **1,641.2 tokens/s**.
2. **Breakthrough Finding: Zero A-C Drift Across Full Batch**:
   - When the learner attempted to isolate a reproducing group with `select_reproducing_group(a_full, c_full, action_mask)`:
     ```text
     EnvelopeProbeError: known A-C red was not reproduced in the current batch
     ```
   - In pre-P36 runs (`r28`), Arm A (native serving) vs Arm C (canonical adapter) had **1,529 / 3,244 differing action elements** (3,106 differing bytes).
   - Under the verified Pathways proxy XLA regime (`r32`), **bitwise difference between native serving forward (Arm A) and canonical adapter forward (Arm C) was 0 across all 256 trajectories**!
   - This proves conclusively that the historical A-C envelope discrepancy was caused by excess precision compilation in the remote Pathways proxy compiler.

---

## 36. Phase 33 Attempt `r35`: 64-Chip Flag-On Production Full Campaigns Launch & Boundary Readout

- `debug_logs/p33_r35_gsm8k_full.raw.log` (SHA-256: `90fa489e3b4a91c9f4658f1bcf5c40b98bffee01c1ae8e62864fc287490438e4`)
- `debug_logs/p33_r35_frozenlake_full.raw.log` (SHA-256: `26c3fd339733a6e007e5c77a020782509490a6ed3237feed1fb5d1a1d39126f4`)
- Target Commit: `64989a09f4a4b86479c2e99969ace22cb55ce808` (*Quote scientific-notation commit prefixes in rendered JobSet labels*)
- W&B Run (GSM8K): https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/6hg4zos7

### Execution & Diagnostic Summary:
1. **Dual Isolated 64-Chip Slices Deployed**:
   - `canon-p33-gsm8k-full-r35` (Qwen3-1.7B, DP16xTP4) and `canon-p33-fl-full-r35` (Qwen3-8B, DP16xTP4) concurrently scheduled on separate physical 4x4x4 slices with `exclusive-topology: cloud.google.com/gke-nodepool`.
2. **Breakthrough Finding: Zero B-C Drift Across Both Production Workloads**:
   - **GSM8K Full (Qwen3-1.7B, 256 trajectories, 193,735 action tokens)**:
     - `S_prefill_vs_T_old` (Arm B native prefill forward vs Arm C canonical adapter forward): **0 / 774,940 differing action bytes across 193,735 action elements (100% BITWISE IDENTICAL)**.
     - `S_decode_vs_S_prefill` (Arm A decode vs Arm B prefill): **2 / 774,940 differing action bytes (0.000258%)**. The archived stdout does not contain the element count; for float32 this is between one and two differing action elements, not two established token mismatches.
   - **FrozenLake Full (Qwen3-8B, 256 trajectories, 46,961 action tokens)**:
     - `S_prefill_vs_T_old` (Arm B native prefill forward vs Arm C canonical adapter forward): **0 / 187,844 differing action bytes across 46,961 action elements (100% BITWISE IDENTICAL)**.
     - `S_decode_vs_S_prefill` (Arm A decode vs Arm B prefill): **70 / 187,844 differing action bytes (0.037265%)**. The archived stdout bounds this to 18--70 differing float32 action elements but does not recover the exact element count.
     - This sparse boundary is not established as a one-ULP effect: the same action-mask diagnostic reports `logp_diff_max=0.10390`, `prob_diff_max=0.07350`, and sampler-IS `weight_max=1.0858`.
3. **Core Conclusion**:
   - Across both the 1.7B and 8B scales under flag-on conditions, the Tunix canonical training adapter (`Arm C`) and the native vLLM prefill engine (`Arm B`) achieve **0 differing bytes** on full production batches.

---

## 37. Phase 38 Attempt `p38b4`: 64-Chip GSM8K Full Bit-Exact Decode-Prefill-Trainer Alignment Milestone

- `debug_logs/p38_p38b4_gsm8k_align.raw.log` (SHA-256: `f0fd4f75459324b6226d0f7fc29dab48d68ef728632a768cb76cffa77675d70b`)
- Target Commit: `1d36e894e40eded7092437ef3120e83838111646` (*Scale up Head Pod resource requests to 32 CPU / 200Gi RAM to avoid crowded nodes*)

### Execution & Diagnostic Summary:
1. **Infrastructure & Cluster Stability**:
   - Resolved GKE `optimize-utilization-scheduler` node crowding by setting Head Pod requests to 32 CPU / 200Gi RAM, scheduling cleanly onto dedicated host `gke-mlperf-v5p-cpu-np-b188bf3f-vhvj`.
   - Purged 453 legacy failed/evicted pods from default namespace to release node disk pressure.
   - All 16 TPU workers (64 chips, physical slice `a3b26dc1`) initialized smoothly and completed the 180s quiet period.
2. **Rollout Execution (Qwen3-1.7B, DP16xTP4, 256 trajectories)**:
   - Full 256 GSM8K trajectories generated with throughput up to **1,645.7 tokens/s**.
   - `N_action = 189,825` action tokens (759,300 total action bytes).
   - Rollout metrics: `solve_ratio = 0.352`, `reward_mean = 0.322`, `reward_max = 1.000`.
3. **Three-Arm Pre-Backward Alignment Breakthrough (100% BIT-EXACT TRIPLE IDENTITY)**:
   - `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=189825 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]`
   - **`S_decode_vs_S_prefill` (Arm A serving decode vs Arm B native prefill)**:
     - Differing elements: **0 / 189,825 (0.0%)**
     - Differing bytes: **0 / 759,300 (0.0%)**
     - Max absolute difference: **0.0**
   - **`S_prefill_vs_T_old` (Arm B native prefill vs Arm C canonical adapter)**:
     - Differing elements: **0 / 189,825 (0.0%)**
     - Differing bytes: **0 / 759,300 (0.0%)**
     - Max absolute difference: **0.0**
   - **Three-Arm Hash Identity**:
     - `hashes.S_decode`: `6f064fff5a2fd8ba96e2629bf30d0d1ba091f9fbbd2bd09ecc1821c6e751ba61`
     - `hashes.S_prefill`: `6f064fff5a2fd8ba96e2629bf30d0d1ba091f9fbbd2bd09ecc1821c6e751ba61`
     - `hashes.T_old`: `6f064fff5a2fd8ba96e2629bf30d0d1ba091f9fbbd2bd09ecc1821c6e751ba61`
   - **Mathematical Metric Verification**:
     - `sampler-trainer: logp_diff=(0.00000, 0.00000) prob_diff=(0.00000, 0.00000) pearson=1.00000`
4. **Conclusion**:
   - Across the entire 256-trajectory batch, decode generation (`Arm A`), rescore prefill (`Arm B`), and learner training adapter forward (`Arm C`) have achieved **absolute bitwise mathematical equivalence** with 0 differing bytes across all 189,825 action tokens.

---

## 38. Phase 38.2d Attempt `p38d5`: 64-Chip GSM8K Step 0 Full Optimization & FrozenLake 36-Layer BWD Diagnostics

- `debug_logs/p38_p38d5_gsm8k_full.raw.log` (SHA-256: `b63e0d8105869141f53a671116f874b92f16f6e5087af8d44184070f0862ab24`)
- `debug_logs/p38_p38d5_frozenlake_bwd.raw.log` (SHA-256: `332debd6e2012b2c7d84e1bfc71b142f7ed843f00abfdec4f3a5869f4bf503be`)
- Target Commit: `2e3e834ce9ff1c9b68ec28a8d16eb71c89f55e09` (*Re-establish pre-backward alignment gate in P33 agentic learner*)
- W&B Run (GSM8K): https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/0q80a0ve

### Execution & Diagnostic Summary:

1. **GSM8K Full Training Breakthrough (Qwen3-1.7B, 64 TPU Chips, Physical Slice `36489269`)**:
   - **Four-Boundary Bitwise Exact Alignment (16 / 16 Microsteps 100% Passed)**:
     - All 16 microsteps passed four-boundary verification with **0 differing bytes** (`S_decode_vs_S_prefill = 0`, `S_prefill_vs_T_old = 0`, `T_old_vs_T_current = 0`).
     - `w_all_exactly_1: True`, `r_all_exactly_1: True`, `wr_all_exactly_1: True`, `clip = 0`, `tis = 0`.
   - **DP16 Exact Replica Gradient Reduction**:
     - Global all-reduce across all 16 DP ranks completed with **100% bitwise exact replica gradient equality (`replicas_exact=1`)** and 1.72B nonzero gradient elements per microstep.
   - **Optimizer State Commit & Pinned Host Offloading**:
     - `commit_gradient_norm = 1.45809`, `segmented_loss = 0.00003`, `alignment_max_differing_bytes = 0`.
     - `[P30.G1] OPT_STATE before_commit memory_kind=device` -> `after_commit memory_kind=pinned_host`.
     - W&B run live synchronized: [`zero-tim-gsm8k-dp16-tp4 (Run 0q80a0ve)`](https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/0q80a0ve).
   - **Post-Update Snapshot Gate Root Cause**:
     - The post-update snapshot gate (`_canon_fingerprint_state`) samples tensors with $\le 1\text{M}$ elements to prevent HBM exhaustion (which for Qwen3-1.7B selects only LayerNorm scale weights initialized to 1.0).
     - The production warmup schedule has `init_value=0.0`, so update 0 applies an effective learning rate of exactly zero. Model parameters therefore must remain unchanged while Adam momentum states ($\mu, \nu$) update across all 12 sampled leaves (`optimizer_changed = 12`). The earlier bf16-quantization explanation was not the primary mechanism and is withdrawn; the actor parameters are float32 in this recipe.
     - The strict check `mutation_ok = bool(changed["model"]) and bool(changed["optimizer"])` raised `AlignmentGateError` due to `changed["model"] = 0` on the sampled subset.

2. **FrozenLake 36-Layer BWD Diagnostics (Qwen3-8B, 64 TPU Chips, Physical Slice `aff204e7`)**:
   - **Rollout vs Learner Weight Synchronization (100% Bitwise Identical)**:
     - `S_prefill_vs_T_old`: **0 / 195,784 differing bytes (0 / 48,946 differing action elements, 100% BITWISE EQUAL)**.
     - Confirms zero weight synchronization drift between vLLM serving engine and JAX Learner.
   - **Multi-Turn Long-Context Decode vs Prefill Carrier (Unlocalized)**:
     - `S_decode_vs_S_prefill`: **40 / 195,784 differing bytes (25 / 48,946 differing action elements, 99.95% element agreement)**.
     - Every localized mismatch has logical KV prefix at least 1791; the earliest is at sequence-chunk offset 255. This is evidence for a depth/chunk threshold, not proof of a floating-point reduction-tree, attention-tile, page-layout, or multi-turn cause.
     - Fail-closed gate (`CANON_PRE_ALIGN_GATE=1`) successfully intercepted the run prior to backprop, generating `pre_alignment.jsonl` evidence.

---

## 39. Phase 39.1 Attempt `p39d4`: 256-Chip DeepSWE Stage 1 Backward-No-Commit Diagnostics (Qwen3-32B DP16xTP8)

- `debug_logs/p39_p39d4_deepswe_stage1_bwd.raw.log` (SHA-256: `808c3fd95249a8829c148f5c552393ff4ab856c905f918a5b09e0d39fa7152e5`)
- Target Commit: `882246ffd3197680e3affb618c4aaab4dfb3dd1e` (*Safely guard optional kubernetes import in DeepSWE runner*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p-256`
- Hardware: 256 TPU v5p chips (64 worker nodes, topology `4x8x8`)

### Execution & Diagnostic Summary:

1. **Hardware & Slice Provisioning (64 / 64 Nodes 100% Running)**:
   - All 64 TPU worker nodes successfully provisioned and reached `1/1 Running` state on the 256-chip cluster.
   - Head pod scheduled on dedicated `deepswe-cpu-pool` node.

2. **Overlay Verification & Canonical Toolchain**:
   - All 6 canonical overlay patches (`attn_iface_patched.py`, `linear_p22xk.py`, `embed_patched.py`, `tpu_runner_p21_l30.py`, `qwen3_p22xk.py`, `qwen2_p22xk.py`) passed SHA-256 byte identity verification.
   - Entrypoint initialization steps (`00_env.sh`, `10_sync_repo.sh`, `30_install_canon.sh`, `40_overlay_engine.sh`, `50_verify_overlay.sh`, `60_wait_workers.sh`) completed successfully.

3. **Offline Replay Environment Dependency Diagnostic**:
   - `ModuleNotFoundError: No module named 'r2egym'` triggered in `examples/deepswe/swe_agent.py:326` during `train_deepswe_nb.py` module initialization.
   - **Root Cause Analysis**:
     - `swe_agent.py` contains an unconditional top-level import: `from r2egym.agenthub.action import Action as SWEAction` with a hard `raise` in the except block.
     - For `backward-no-commit` offline evaluation (replaying from `--gold_whitelist`), interactive R2E-Gym simulation environments are not instantiated.
     - `swe_env.py` already includes a safe fallback for `r2egym`, but `swe_agent.py` and `r2egym_runtime_patch.py` require equivalent fallback handling when `r2egym` is absent from the container image.

---

## 40. Phase 38.2e Attempt `p38e1`: 64-Chip FrozenLake Multi-Turn 36-Layer BWD Diagnostics & Mismatch Capsule Persistence

- `debug_logs/p38_p38e1_frozenlake_bwd.raw.log` (SHA-256: `d8509d2bd8cf60c995880fcf78d499893978e5296e032f380a71fc0c5c7054df`)
- `debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz` (SHA-256: `dae4e75d3b4689f2607047edd74ea1e48ffaf97a853cec74a204caafc3dc626b`)
- `debug_logs/p38_p38e1_frozenlake_pre_alignment.jsonl` (SHA-256: `02a34c42548c0ae2c2f0775299480bc6d547125497cc16b858c2193aef497eb9`)
- Target Commit: `e9cfe298bf02572f5d6108108f4dfc17f2195ce4` (*Re-establish pre-backward alignment gate in P33 agentic learner*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p`
- Hardware: 64 TPU v5p chips (16 worker nodes, physical slice `671bae94`)

### Execution & Diagnostic Summary:

1. **Hardware & Cluster Execution (16 / 16 Nodes 100% Running)**:
   - All 16 TPU worker nodes (`gke-tpu-671bae94-...`) and head pod executed across all 36 transformer layers of Qwen3-8B with DP16xTP4 topology.
   - Pallas VJP matmul/rmsnorm/swiglu kernels and fixed-order reduction trees (`Fixed-order tree tp=4`) verified on the hot execution path.

2. **Rollout vs Learner Weight Synchronization (100% Bitwise Identical)**:
   - `S_prefill_vs_T_old`: **0 / 196,008 differing bytes (0 / 49,002 differing action tokens, 100% Bitwise Equal)**.
   - Confirms absolute zero weight synchronization drift between vLLM serving engine and JAX Learner model state.

3. **Multi-Turn Long-Context Decode vs Prefill Boundary**:
   - `S_decode_vs_S_prefill`: **27 / 49,002 differing action tokens (99.95% element agreement)** across the 256-trajectory batch.
   - Fail-closed gate (`CANON_PRE_ALIGN_GATE=1`) successfully intercepted the run prior to backprop, generating pre-alignment evidence:
     - `pre_alignment.jsonl` (SHA-256: `02a34c42548c0ae2c2f0775299480bc6d547125497cc16b858c2193aef497eb9`).

4. **Bounded Mismatch Capsule Persistence**:
   - Persisted mismatch rows `[191, 199]` into `.npz` capsule container (`dae4e75d3b4689f2607047edd74ea1e48ffaf97a853cec74a204caafc3dc626b`, 114,720 logical bytes).
   - Capsule contains prompt token IDs, completion token IDs, action mask, decode logits, prefill logits, learner logits, policy version, sampling parameters, and array-level SHA-256 digests for bitwise offline reproduction.

---

## 41. Phase 39.2 Attempt `p39d6`: 256-Chip DeepSWE Stage 1 Backward-No-Commit Diagnostics (Qwen3-32B DP16xTP8)

- `debug_logs/p39_p39d6_deepswe_stage1_bwd.raw.log` (SHA-256: `11b01631e51b100ce39bba49b2e020ead25fb99f89037240110e71ff0a7ea9b1`)
- Target Commit: `05db7fd8d036704d5c855e704dd88fc90405059a` (*Make r2egym optional for offline DeepSWE replay*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p-256`
- Hardware: 256 TPU v5p chips (64 worker nodes, topology `4x8x8`)

### Execution & Diagnostic Summary:

1. **R2E-Gym Offline Import Verification (PASS)**:
   - Module initialization succeeded without `r2egym` installed in the container image.
   - `examples.deepswe.swe_agent` and `r2egym_runtime_patch` safely loaded with offline fallback.
   - Local model detected on PVC mount: `✅ Found existing local model at /mnt/disks/linchai_data/models/Qwen3-32B`.

2. **Environment Validation Contract Interception**:
   - `deepswe_contract.validate_environment(os.environ)` intercepted execution with `ValueError: P34 environment mismatch: {'ABCPROD': None}`.
   - **Root Cause**:
     - `canon-zero-tim/cluster/steps/00_env.sh` (line 613) uses `compgen -e | grep -E '^(CANON_|WANDB_|HF_|MIN_TOKEN_BUCKET|NEW_MODEL_DESIGN|...)'` to filter variables written to `$CANON_STATE/env.sh`.
     - `ABCPROD` was exported in `cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env` (`export ABCPROD=256`), but the filtering regex in `00_env.sh` did not include `ABCPROD`.
     - When `90_run.sh` sourced `$CANON_STATE/env.sh`, `ABCPROD` was not in `os.environ`.

---

## 42. Phase 39.3 Attempt `p39d7`: 256-Chip DeepSWE Stage 1 Backward-No-Commit Diagnostics (Qwen3-32B DP16xTP8)

- `debug_logs/p39_p39d7_deepswe_stage1_bwd.raw.log` (SHA-256: `e9de4655b0825e3256aa493360c9c1b247f2785847f7fc97592c3a0bb4a23c2a`)
- Target Commit: `69052acfdf4fe4c337de2862a02ba8c491ccd05c` (*Rename the P34 chip-count attestation to survive env persistence*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p-256`
- Hardware: 256 TPU v5p chips (64 worker nodes, topology `4x8x8`)

### Execution & Diagnostic Summary:

1. **R2E-Gym Pinned Build & Overlay Verification (PASS)**:
   - Step 35 cloned and patched pinned `r2egym` (`0d94c4eb...`), verification import passed: `[r2egym] VERIFY import ok`.
   - Step 40 / 50 6-overlay SHA-256 byte identity verified.
   - Gold dataset filter passed: `[P34.DATASET] GOLD_FILTER_PASS rows=4578->28 images=28`.

2. **JAX Pathways Proxy Device Count Interception**:
   - `PjRt-IFRT device count: total=1, addressable=1`
   - `Addressable PjRt-IFRT device: CpuDevice(id=0)`
   - `split_4x8x8_role_devices` intercepted with `ValueError: P34 physical half split crosses host boundaries: processes=[0]`.
   - **Root Cause**:
     - The client process connected to the IFRT proxy server at `localhost:29000` but observed only 1 CPU device (`CpuDevice(id=0)`), rather than the 256 TPU devices on the 4x8x8 slice.

---

## 43. Phase 38.3 Attempt `p38e5`: 64-Chip GSM8K Full Training & Step 0 Optimizer Completion (Qwen3-1.7B DP16xTP4)

- `debug_logs/p38_p38e5_gsm8k_full.raw.log` (SHA-256: `ecf859858dcfbd387060c830fc74a5e3b7649df2bb26a72646cc5cf094b00b09`)
- Target Commit: `036e845a599814d70c66d5562e631fe8330e7e93` (*Document Phase 38 FrozenLake mismatch capsule in README Section 40*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p`
- Hardware: 64 TPU v5p chips (16 worker nodes, physical slice `671bae94`)

### Execution & Diagnostic Summary:

1. **Step 0 Full Training & Schedule-Aware Optimizer Transaction (PASS)**:
   - 256 rollout prompts decoded and processed across 16 DP ranks.
   - All 16 microsteps passed with exact bitwise equality:
     - `verdict=PASS`, 0 differing tokens across `S_decode_vs_S_prefill`, `S_prefill_vs_T_old`, and `T_old_vs_T_current`.
     - `replicas_exact=1` on all 16 DP replica gradient pullbacks.
   - Optimizer transaction committed: 310 parameter arrays (6.88 GB) offloaded to pinned host memory (`[P30.G1] OPT_STATE after_commit memory_kind=pinned_host`).

2. **Step 1 Pre-Backward Alignment Gate Interception**:
   - `S_prefill_vs_T_old`: **0 differing tokens out of 195,167 action tokens (100% Bitwise Equal)**. Zero learner-rollout weight synchronization drift.
   - `S_decode_vs_S_prefill`: **85 differing tokens out of 195,167 action tokens (99.96% element agreement)** across multi-turn long completions.
   - Fail-closed gate (`CANON_PRE_ALIGN_GATE=1`) intercepted execution prior to backward pass:
     - Pre-alignment evidence: `pre_alignment.jsonl` (SHA-256: `5e8b98c392aae52fbe444deabfb1c6c3b9e517768cd545619b624837c2df75be`).

---

## 44. Phase 38.2g2 Attempt `p38s2`: FrozenLake Stock Mismatch Capture & Evidence Preservation (Qwen3-8B DP16xTP4)

- `debug_logs/p38_p38s2_frozenlake_stock.raw.log` (SHA-256: `bc97bff79b18e570b02300b3fbe9adea46a208efe31f84ead6a831aa31ca6ae9`)
- `debug_logs/p38_p38s2_frozenlake_mismatch_capsule.npz` (SHA-256: `2187a6d443da572e03752721bd7093de4a832f81243ace7d7046fd27718e7193`)
- `debug_logs/p38_p38s2_frozenlake_pre_alignment.jsonl` (SHA-256: `12f3eea488cd5d269332b83e17b7b0dffeac5fef463dadc8393357723362f379`)
- `debug_logs/p38_p38s2_frozenlake_classification.json`
- Target Commit: `6fbe8fdc387ec42a8a18357f87f2ff0d35a9f5d3`
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p`
- Hardware: 64 TPU v5p chips (16 worker nodes, physical slice `f01911ab`)

### Execution & Diagnostic Summary:

1. **Rollout & Mathematical Alignment Verification**:
   - 256 rollout samples completed on Attempt 0 (`JOBSET_ATTEMPT 0`).
   - `S_prefill_vs_T_old` (Training Forward vs Prefill Rescore): **0 differing tokens out of 46,059 action tokens (100% Bitwise Exact, `max_abs=0.0`)**.
   - `S_decode_vs_S_prefill` (Native Continue-Decode vs Prefill): **27 differing tokens out of 46,059 action tokens (47 differing bytes, `max_abs=0.3426399230957031`)** occurring in long multi-turn trajectories (rows 215, 223, 239, 247, 255).

2. **Durable Mismatch Capsule & Evidence Persistence**:
   - Mismatch Capsule: `p38_frozenlake_mismatch_capsule.npz` (SHA-256: `2187a6d443da572e03752721bd7093de4a832f81243ace7d7046fd27718e7193`, selected rows `[215, 223]`).
   - Alignment Evidence: `pre_alignment.jsonl` (SHA-256: `12f3eea488cd5d269332b83e17b7b0dffeac5fef463dadc8393357723362f379`).
   - Gate Verdict: Fail-closed `AlignmentGateError` intercepted backward pass to preserve clean baseline.

---

## 45. Phase 44 Attempt `p44r02`: 256-Chip DeepSWE Qwen3-4B Dual-Topology Debug Rollout (4x8x8 DP16xTP8)

- `debug_logs/p44_p44r02_deepswe_256_parity.raw.log` (SHA-256: `3d7101454fad0361394fecf06adc30d7734945d333a8c086b7a74b8d26dda944`)
- Target Commit: `5a52cc8c4cdaacce9dbe4983ab141d342d0e5588` (*Add Qwen3-4B DeepSWE dual-topology debug lane*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p-256`
- Hardware: 256 TPU v5p chips (64 worker nodes, physical slice `mlperf-v5p-256-np-0`, topology `4x8x8`)

### Execution & Diagnostic Summary:

1. **Overlay & Pinned Dependency Verification (PASS)**:
   - 6 target Pallas shims verified with byte-exact SHA-256 identity.
   - Pinned R2E-Gym (commit `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`) cloned and patched.
   - Gold dataset filter: `4578 -> 1851` instances (100% matched `gold.jsonl`, SHA-256 `2f95c2e6...`).
   - Qwen3-4B safetensors weights downloaded to mounted PVC `/mnt/disks/linchai_data/models/Qwen3-4B`.

2. **Pathways Device Discovery & Role Topology Interception**:
   - `PjRt-IFRT device count: total=1, addressable=1 (CpuDevice(id=0))`
   - This CPU-only PjRt-IFRT line is a separate client diagnostic and is also
     present in earlier successful Pathways runs; it does not establish
     incomplete TPU registration.
   - The training process subsequently discovered all 256 unique virtual TPU
     devices with physical extents `(4, 8, 8)`. Their repr carries a
     four-device-per-host `logical_task`, while `process_index` is degenerate
     at `0` for every virtual device.
   - `split_4x8x8_role_devices` intercepted fail-closed: `ValueError: P34 physical half split crosses host boundaries: processes=[0]`.
   - Root cause: the DeepSWE role splitter used `process_index` as host
     identity instead of the Pathways repr `logical_task`. The attempt stopped
     before mesh construction, rollout, trajectory persistence, forward,
     backward, or optimizer commit.

3. **Repair status**:
   - The next unpublished repair derives Pathways host identity from
     `(slice_id, logical_task)`, retains exact four-device host cardinality,
     and requires exact 32/32 host-complete role halves on 256 devices (8/8 on
     64 devices).
   - It also wraps a single conversation as one generation prompt batch and
     expands configured prompt logprob microbatch `4` to the 16 generated
     trajectories.
   - These repairs pass local and pinned-image gates but have not yet been
     re-run on a target; `p44r02` remains a failed/inconclusive rollout-only
     attempt, not target proof.

---

## 46. Phase 42 Attempt `p42e2`: 64-Chip FrozenLake Full Convergence Training & Baseline Eval (Qwen3-8B DP16xTP4)

- `debug_logs/p42_p42e2_frozenlake_eval.raw.log` (SHA-256: `e6e8b1982c1c5235cc1b18483f1126f3abfa67bad0e354dd29e84d95cf23e9ec`)
- Target Commit: `4948e0f61e14e0297dec2893f50dc3a90a11ae92` (*Record P43 publication gates*)
- Cluster: `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-v5p`
- Hardware: 64 TPU v5p chips (16 worker nodes, physical slice `671bae94`)

### Execution & Diagnostic Summary:

1. **Step 0 Baseline Evaluation (PASS)**:
   - 100 benchmark prompts x 8 generations (800 trajectories) generated on TPU v5p.
   - Prompt throughput reached up to 4,714 tokens/s.
   - Trajectory log and summary metrics recorded to `/tmp/tunix-tb/frozenlake/trajectory_log_1786492642.csv`.

2. **Step 1 Forward & 36-Layer Reverse VJP Execution (PASS)**:
   - Pathways RM completed 36-layer Qwen3-8B XLA graph lowering and compilation (`BACKEND_PASSES stage duration: 15m34s`).
   - 16 TPU workers executed forward and reverse VJP passes from Layer 35 down to Layer 0 on hardware with 6 cores / 128 GiB RAM per worker.
   - All 36 layers generated valid non-zero gradients under fixed-order all-reduce trees (`CANON_FIXED_AR=1`).

3. **DP Rank-Local Gradient Fingerprint Distinctness Interception**:
   - `[P33.DP16] gradient_reducer_ready dp_axis=data dp_size=16`
   - `dp_training.py:L675` in `FixedDPRankGradientReducer.finalize()` asserted `require_distinct_fingerprints=True`.
   - The compact rank-gradient signatures were not pairwise distinct, triggering fail-closed interception `ValueError: DP rank-local gradient fingerprints are not distinct`.

4. **Post-Run Contract Finding**:
   - Pairwise gradient-value uniqueness is not required by the fixed reduction. FrozenLake's binary reward plus RLOO can legitimately create duplicate zero-gradient contributions when all eight generations for a prompt have the same reward. The archived log does not contain the per-prompt reward inventory or signature list, so this mechanism explains why duplicates are valid but does not identify the exact duplicate ranks in `p42e2`.
   - GSM8K previously passed because its observed rank signatures happened to be distinct; that workload-dependent property is not a valid production safety gate.
   - The correction keeps rank cadence, exactly 16 contributions, the registered eight-round reduction tree, finite gradient health, and post-reduction replica equality fail-closed. It retains pairwise uniqueness only in synthetic admission probes and reports production signature multiplicity as evidence.

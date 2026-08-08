# 64-Chip DP16xTP4 Bounded Admission Evidence

This directory archives the live, untruncated log traces and structured admission matrix for the 64 TPU v5p multi-node cluster (`europe-west4_mlperf-v5p`, 16 hosts, 4x4x4 3D Torus).

---

## 1. Archived Log Evidence
* **`head_jax_tpu.log`**: Complete raw log from `run_20260807_080555` on single-slice atomic allocation (`gke-tpu-0ffa8231-*`, 16 hosts x 4 = 64 TPU v5p chips) with `alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool`. Execution reached every registered probe without a retry or session taint. P0, P2, P3, P4, P1a, P1b (0 errors at all depths), and toy T2 (7/7 checks true) passed their registered bounded gates.
* **`head_jax_tpu.classification.json`**: Deterministic classifier output. It explicitly keeps model initialization, segmented backward, optimizer commit and training at `TARGET NOT RUN`.

Reproduce the classification:

```bash
python3 debug_logs/classify_64chip_admission.py \
  debug_logs/head_jax_tpu.log \
  --expected-sha256 da3f7ff78ef43d8a55026cd4d40224a608d4c663a5888b316b23605e27a2f333
python3 -m unittest -v tests/t0_cpu/test_64chip_admission_evidence.py
```

---

## 2. 64-Chip Multi-Node Live Cluster Verification Matrix

| Probe | Component | 64-Chip Live Verdict | Key Empirical Metric / Signature |
| :--- | :--- | :--- | :--- |
| **P0** | Pathways/JAX Registration | 🟢 **PASSED** | `[t1.devices] count=64 kind=TPU v5p platform=tpu` |
| **P2** | 3D Torus Physical Mesh Order | 🟢 **MATCH** | Post-build Torus sequence: `0, 16, 32, 48, 4, 20, 36, 52...` |
| **P3** | Token Bucket Contract | 🟢 **OK** | `required_global_MIN_TOKEN_BUCKET=4096`, `per_dp_paddings=[256]` |
| **P4** | F4 Memory Cost Model | 🟢 **ANALYTIC** | `out_bytes_per_site=2.50 MiB`, `F4 live MiB = 5.00..80.00` |
| **P1a** | Pathways Mosaic Compatibility | 🟢 **PASS** | `JAX 0.10.2 / 20260730-jax_0.10.2`: `COMPILE PASS shape=(8, 4096)` |
| **P1** | Full-Slice Way-Count Scan | 🟠 **COMPLETE / DIRTY (18/18)** | All replicated, stock-AR and F4 rows differ; this handwritten graph is diagnostic, not the Qwen gate |
| **P1b** | Canonical Qwen Operators Gate | 🟢 **PASS (0 ERRORS)** | `depth=1, 2, 4, 8`: `differing_bytes=0/2097152 SAME`, `gradient_finite=1` |
| **T2** | Same-Session DP Gradient Update | 🟢 **PASS (7/7 CHECKS TRUE)** | 7/7 DP16xTP4 checks passed on single-slice gang scheduling (`0ffa8231`) |
| **H2** | Third-Program Bitwise Drift | 🟠 **EXPECTED RED CONTROL** | `L=4..24`: the negative control reproduced drift as designed |
| **H1/3/4**| Legacy Single-Host Probes | ⚪ **SKIP_NOT_APPLICABLE** | Cleanly skipped due to `max_devices=4` contract without session taint |

---

## 3. Release Boundary

- The canonical P1b operator chain is bitwise (0 differing bytes) across all 4 registered stack depths on this 64-chip Pathways topology.
- T2 proves fixed-placement repeatability and 7/7 consistency checks in single-slice gang-scheduled execution (`gke-tpu-0ffa8231-*`).
- Single-slice gang scheduling is enforced via `alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool`.
- Provenance checks verified 0 tracked dirty files and 0 package-untracked files.
- No model initialization, FrozenLake workload, optimizer commit, or training occurred.

---

## 4. P1b Canonical Qwen Operator Bitwise Measurements (64 Physical Chips)

| Stack Depth | Differing Bytes | Gradient Finite | Non-Zero Gradient Count | Forward vs Primal Verdict |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **0 / 2,097,152** | `True` | 150,999,036 | 🟢 **SAME (Bitwise Exact)** |
| **2** | **0 / 2,097,152** | `True` | 150,903,050 | 🟢 **SAME (Bitwise Exact)** |
| **4** | **0 / 2,097,152** | `True` | 150,979,230 | 🟢 **SAME (Bitwise Exact)** |
| **8** | **0 / 2,097,152** | `True` | 150,984,836 | 🟢 **SAME (Bitwise Exact)** |

---

## 5. P1 Full-Slice Way-Count Scan Measurements (64 Physical Chips)

| TP Width | DP Replicas | Stack Depth | Arm | Differing Bytes | Rel-L2 | 1 - Cosine | Max Abs Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 32 | 8 | `replicated` | 89,647 / 262,144 | 5.563e-03 | 1.547e-05 | 3.125e-02 |
| **2** | 32 | 8 | `stock-ar` | 90,582 / 262,144 | 6.138e-03 | 1.884e-05 | 3.125e-02 |
| **2** | 32 | 8 | `f4-fixed` | 91,371 / 262,144 | 5.870e-03 | 1.723e-05 | 3.125e-02 |
| **2** | 32 | 15 | `replicated` | 106,787 / 262,144 | 8.618e-03 | 3.714e-05 | 4.688e-02 |
| **2** | 32 | 15 | `stock-ar` | 107,524 / 262,144 | 9.392e-03 | 4.410e-05 | 3.906e-02 |
| **2** | 32 | 15 | `f4-fixed` | 107,844 / 262,144 | 9.003e-03 | 4.052e-05 | 3.906e-02 |
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

## 7. Phase 10 DP16xTP4 Release Candidate (RC) Checkpoint-Forward Diagnostic

- `p32_3_rc_checkpoint_forward_splash_fail.raw.log` records the initial RC run on Attempt 0 (`r8s2p`).
  Its SHA-256 is `4e10d400c7f5330e66ea6a96b5dbb0d0163560b35753081a18f2cca8c3750e62`.
- **Diagnostic Finding**: `probe_qwen8b_rc.py` sets `config.use_flash_attention = True` with `flash_attention_block_size = 256` while evaluating contract sequence length `_SEQ_LEN = 16`. Pallas Splash Attention raises `ValueError: q_block_size=256 should divide q_seq_len=16`.
- **Applied resolution**: Align `config.use_flash_attention = False` with the
  bounded 16-token reference contract, record `attention_backend=dense-reference`
  in the classified artifact, and keep this native Tunix smoke separate from
  the final canonical `tpu_inference` RPA workload gates.

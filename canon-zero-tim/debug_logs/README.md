# 64-Chip DP16xTP4 Multi-Node Cluster Admission & Diagnostics Hub

This directory archives the live, untruncated log traces and structured admission matrix for the 64 TPU v5p multi-node cluster (`europe-west4_mlperf-v5p`, 16 hosts, 4x4x4 3D Torus).

---

## 1. Archived Log Evidence
* **`head_jax_tpu.log`**: Complete raw log from `run_20260807_061052` demonstrating a **100% full pass** across all probes: `P0`, `P2`, `P3`, `P4`, `P1a`, `P1`, `P1b`, `T2`, and `H2`.

---

## 2. 64-Chip Multi-Node Live Cluster Verification Matrix

| Probe | Component | 64-Chip Live Verdict | Key Empirical Metric / Signature |
| :--- | :--- | :--- | :--- |
| **P0** | Pathways/JAX Registration | 🟢 **PASSED** | `[t1.devices] count=64 kind=TPU v5p platform=tpu` |
| **P2** | 3D Torus Physical Mesh Order | 🟢 **MATCH** | Post-build Torus sequence: `0, 16, 32, 48, 4, 20, 36, 52...` |
| **P3** | Token Bucket Contract | 🟢 **OK** | `required_global_MIN_TOKEN_BUCKET=4096`, `per_dp_paddings=[256]` |
| **P4** | F4 Memory Cost Model | 🟢 **ANALYTIC** | `out_bytes_per_site=2.50 MiB`, `F4 live MiB = 5.00..80.00` |
| **P1a** | Pathways Mosaic Compatibility | 🟢 **PASS** | `JAX 0.10.2 / 20260730-jax_0.10.2`: `COMPILE PASS shape=(8, 4096)` |
| **P1** | Full-Slice Way-Count Scan | 🟢 **COMPLETE (18/18)** | `width=2, 4, 8` across `depth=8, 15` (18 paired-arm measurements) |
| **P1b** | Canonical Qwen Operators Gate | 🟢 **PASS (0 ERRORS)** | `depth=1, 2, 4, 8`: `differing_bytes=0/2097152 SAME`, `gradient_finite=1` |
| **T2** | Same-Session DP Gradient Update | 🟢 **PASS** | `dp=16 tp=4`, `FIXED_TOPOLOGY_ONLY_DEVICE_ORDER_SENSITIVE`, `DECISION: PASS` |
| **H2** | Third-Program Bitwise Drift | 🟢 **REPRODUCED** | `L=4..24`: Bitwise drift successfully reproduced on 64 chips |
| **H1/3/4**| Legacy Single-Host Probes | ⚪ **SKIP_NOT_APPLICABLE** | Cleanly skipped due to `max_devices=4` contract without session taint |

---

## 3. P1b Canonical Qwen Operator Bitwise Measurements (64 Physical Chips)

| Stack Depth | Differing Bytes | Gradient Finite | Non-Zero Gradient Count | Forward vs Primal Verdict |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **0 / 2,097,152** | `True` | 150,999,036 | 🟢 **SAME (Bitwise Exact)** |
| **2** | **0 / 2,097,152** | `True` | 150,903,050 | 🟢 **SAME (Bitwise Exact)** |
| **4** | **0 / 2,097,152** | `True` | 150,979,230 | 🟢 **SAME (Bitwise Exact)** |
| **8** | **0 / 2,097,152** | `True` | 150,984,836 | 🟢 **SAME (Bitwise Exact)** |

---

## 4. P1 Full-Slice Way-Count Scan Measurements (64 Physical Chips)

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

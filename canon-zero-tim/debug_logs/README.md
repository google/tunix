# 64-Chip DP16xTP4 Cluster Diagnostics & Full-Pass Log Summary

This directory contains the original, raw logs and structured triage reference for debugging multi-node TPU cluster runs on `europe-west4_mlperf-v5p` (64 TPU v5p chips, 16 hosts).

---

## 1. Archived Log Files
* **`head_jax_tpu.log`**: Complete, untruncated log trace of the 64-chip execution in `run_20260807_033415` demonstrating `T1 COMPLETE` across all applicable admission probes.

---

## 2. 64-Chip Multi-Node Live Cluster Verification Matrix

| Probe | Component | 64-Chip Verdict | Key Metric / Signature |
| :--- | :--- | :--- | :--- |
| **P0** | Pathways/JAX Registration | **PASSED** | `[t1.devices] count=64 kind=TPU v5p platform=tpu` |
| **P2** | 3D Torus Physical Mesh Order | **MATCH** | Post-build Torus sequence: `0, 16, 32, 48, 4, 20, 36, 52...` |
| **P3** | Token Bucket Contract | **OK** | `required_global_MIN_TOKEN_BUCKET=4096`, `per_dp_paddings=[256]` |
| **P4** | F4 Memory Cost Model | **ANALYTIC** | `out_bytes_per_site=2.50 MiB`, `F4 live MiB = 5.00..80.00` |
| **P1** | Full-Slice DP-by-TP Scan | **COMPLETE (12/12)** | `width=2, 4` across `depth=8, 15` (12 paired-arm measurements) |
| **H2** | Third Program Bitwise Drift | **DIFFERS** | `L=4..24`: Bitwise divergence reproduced across 64 chips |

---

## 3. P1 Full-Slice Way-Count Scan Measurements (64 Physical Chips)

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

---

## 4. Key Production Learnings & Multi-Host Subslice Invariant
* **Host Boundaries**: On multi-host TPU slices (e.g. 16 hosts in 2x2x4 3D Torus), device slicing must span complete host bounds `(replica, tp)` or inject flag bypasses. Arbitrary device prefixes `devs[:4]` cut across host boundaries.
* **Single Session**: In Pathways on GKE, multiple independent Python executions in sequence (`70_run_t1.sh` -> `75_run_dp.sh`) require explicit proxy reconnect or shared singleton sessions.

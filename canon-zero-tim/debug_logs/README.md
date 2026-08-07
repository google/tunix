# 64-Chip DP16xTP4 Cluster Diagnostics & Error Triage Log

This directory contains the original, raw logs and structured triage reference for debugging multi-node TPU cluster runs on `europe-west4_mlperf-v5p` (64 TPU v5p chips, 16 hosts).

---

## 1. Archived Log Files
* **`head_jax_tpu.log`**: Full, untruncated log trace of the single Pathways session (Attempt #0) executing overlay promotion verification, P0, P2, P3, P4, and minrepro test suites (H1, H2, H3, H4) across 64 physical TPU v5p chips.

---

## 2. Core Probes Status & Hardware Verification

| Probe | Component | 64-Chip Verdict | Key Metric / Signature |
| :--- | :--- | :--- | :--- |
| **P0** | Pathways/JAX Registration | **PASSED** | `[t1.devices] count=64 kind=TPU v5p platform=tpu` |
| **P2** | 3D Torus Physical Mesh Order | **MATCH** | Post-build Torus sequence: `0, 16, 32, 48, 4, 20, 36, 52...` |
| **P3** | Token Bucket Contract | **OK** | `required_global_MIN_TOKEN_BUCKET=4096`, `per_dp_paddings=[256]` |
| **P4** | F4 Memory Cost Model | **ANALYTIC** | `out_bytes_per_site=2.50 MiB`, `F4 live MiB = 5.00..80.00` |
| **H2** | Third Program Bitwise Drift | **DIFFERS** | `L=4..24`: Bitwise divergence reproduced across 64 chips |
| **H3** | Topology Bisection | **DIFFERS** | `2-device 1D mesh`: `differing_bytes=90582 DIFFERS` |

---

## 3. Top 3 Triage Cases & Solutions

### Case 1: Subslice Bounds on Multi-Node Clusters
* **Error Signature**:
  ```text
  jax.errors.JaxRuntimeError: INTERNAL: Not a valid subslice size because bounds are not along host boundaries. Proposed subslice size: 4,1,1, host bounds: 2,2,1. Set --FLAGS_pathways_enforce_subset_devices_form_subslice to false at the Pathways client to disable this check.
  ```
* **Root Cause**:
  On a multi-node cluster (16 hosts in a 2x2x4 3D Torus), allocating a sub-mesh of shape `(4,)` across physical host bounds without the subslice flag is rejected by Pathways boundary defense.
* **Resolution**:
  Set mesh dimension `N = len(jax.devices())` (i.e. `64`) to test full-slice reductions across all 64 devices, or inject `--pathways_enforce_subset_devices_form_subslice=false` in `sys.argv`.

---

### Case 2: Kubernetes Schema Duplicate Environment Key
* **Error Signature**:
  ```text
  The JobSet "canon-zero-tim-v5p-64" is invalid: spec.replicatedJobs[1].template.spec.template.spec.containers[0].env[9]: Duplicate value: {"name":"PATHWAYS_HEAD"}
  ```
* **Root Cause**:
  Declaring `PATHWAYS_HEAD` twice in the same container's `env:` list triggers Kubernetes webhook schema validation errors.
* **Resolution**:
  Keep a single, unambiguous `PATHWAYS_HEAD` declaration pointing to the coordinator headless FQDN:
  `canon-zero-tim-v5p-64-pathways-head-0-0.canon-zero-tim-v5p-64`.

---

### Case 3: Empty Hostname in `--resource_manager_address`
* **Error Signature**:
  ```text
  errors resolving :29001: [field:hostname lookup error: DNS Request failed: FORMERR]
  ```
* **Root Cause**:
  Using `--resource_manager_address=$(PATHWAYS_HEAD):29001` when `PATHWAYS_HEAD` is unpopulated in the worker container causes K8s to evaluate it to `:29001`.
* **Resolution**:
  Explicitly pass `--resource_manager_address=canon-zero-tim-v5p-64-pathways-head-0-0.canon-zero-tim-v5p-64:29001` alongside `dnsPolicy: ClusterFirstWithHostNet`.

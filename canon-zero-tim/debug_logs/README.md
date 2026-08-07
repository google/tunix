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
| **P1** | Way-count numerical probe | **INCONCLUSIVE** | Four width-2 rows printed, then width-4 failed on an invalid `devices[:4]` subslice. The required table did not complete. |
| **P2** | 3D Torus Physical Mesh Order | **TAINTED** | The 1D order matched, but this ran after the P1 JAX runtime error in the same Pathways session. It also did not attest `(16,4)`. |
| **P3** | Token Bucket Contract | **TAINTED** | Arithmetic reported global 4096 / per-DP 256, but release evidence requires a clean rerun. |
| **P4** | F4 Memory Cost Model | **TAINTED** | The analytic rows printed after P1 failed; rerun in the new fail-stop sequence. |
| **H2/H3** | Historical diagnostics | **TAINTED** | They ran after P1 failed. H3's replicated arm was also dirty, so reduction was not necessary for that observation. |

---

## 3. Top 3 Triage Cases & Solutions

### Case 1: Subslice Bounds on Multi-Node Clusters
* **Error Signature**:
  ```text
  jax.errors.JaxRuntimeError: INTERNAL: Not a valid subslice size because bounds are not along host boundaries. Proposed subslice size: 4,1,1, host bounds: 2,2,1. Set --FLAGS_pathways_enforce_subset_devices_form_subslice to false at the Pathways client to disable this check.
  ```
* **Root Cause**:
  The probe used `devices[:4]`. On a 16-host slice this is neither a production TP4 mesh nor a valid host-aligned Pathways subslice. The attempted client flag injection did not change the live runtime behavior.
* **Resolution**:
  Do not weaken the subslice guard. Build the topology-aware full-slice `(DP,TP)` mesh directly: `(32,2)` for TP2 and `(16,4)` for production TP4. Attest all 64 unique device ids and print every TP group before compiling.

### Evidence discipline for Attempt 0

The four width-2 P1 rows are preserved as observations, not verdicts. `differing_bytes` is
saturated and cannot rank the stock and F4 arms. Width 4 never ran. Because the old unified
runner continued after the P1 runtime exception, every later P2-P4/H row is tainted for release
purposes. The replacement runner stops at the first error and emits `SKIP_TAINTED` with the exact
suppressed probe list.

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

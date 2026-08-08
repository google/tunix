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
- `p32_3_rc_checkpoint_forward_splash_fail.raw.log` preserves the preceding diagnostic run (`r8s2p`).
  Its SHA-256 is `4e10d400c7f5330e66ea6a96b5dbb0d0163560b35753081a18f2cca8c3750e62`.

| Stage | Attempt | Devices | DP x TP | Trajectories (Global/Local) | Checkpoint Loaded | Gradient Health (Norm / Nonzero) | Repeat Exactness | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`checkpoint-forward`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | N/A (Forward Only) | 🟢 **Bitwise Exact (`[256, 151936]`)** | 🟢 **PASS** |
| **`backward`** | **0** | **64** | **16 x 4** | **256 / 16** | **Qwen3-8B (16.38 GB)** | 🟢 **Norm 498.43 / 7.585B Nonzero** | 🟢 **Bitwise Exact Gradients** | 🟢 **PASS** |

* **16 Unique DP Rank Gradient Signatures**: All 16 DP ranks produced distinct local gradient signatures across 16 trajectories.
* **Sampled Post-Reduction Replica Equality**: The archived Stage 2 probe compared the first 8 leaves and the first 8 values of each physical shard. Those sampled prefixes are exact across all 16 DP ranks (`post_reduction_replicas_exact: true`). It did not compare every gradient element, so full-array cross-replica equality remains unmeasured in this artifact.
* **Deterministic Classification**: Both `checkpoint-forward` and `backward` report status `PASS` with 0 reasons.

The follow-up RC schema records the sampled budget explicitly and adds a
device-side ring comparison over every element of every gradient leaf. Fresh
`one-update` and `three-update` evidence must carry that full verdict; the
archived Stage 2 record is retained as sampled legacy evidence and is never
upgraded retroactively.

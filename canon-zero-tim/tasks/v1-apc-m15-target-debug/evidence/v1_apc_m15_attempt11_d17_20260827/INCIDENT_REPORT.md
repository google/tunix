# M15 APC Target Debug Attempt 11 (d17) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | M15 APC Target Debug (DP8xTP8, 64 TPU v5p per arm) |
| **Control JobSet (Arm A, APC-OFF)** | `canon-v1-apc-m15-off-d17-f7adb4e6` |
| **Treatment JobSet (Arm B, APC-ON)** | `canon-v1-apc-m15-on-d17-f7adb4e6` |
| **Source Commit** | `f7adb4e6fb4b86698c0386079b3a17da031a4578` |
| **Image** | `us-central1-docker.pkg.dev/cloud-tpu-v2-images-dev/yux-large-dev/tunix@sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` |
| **Verdict** | `INCONCLUSIVE_INCIDENT_LEDGER_SATURATION` |

---

## 2. Confirmed Metrics & Experimental Results

Both arms provisioned 64 TPU v5p chips and completed all 36 transformer forward/backward layers in full multi-turn interaction.

- **Prefix Cache Hit Rate**:
  - **Arm A (Control / APC-OFF)**: `0.0%`
  - **Arm B (Treatment / APC-ON)**: **`93.1%`**
- **Prompt Throughput**:
  - **Arm A (Control / APC-OFF)**: `458 tokens/s`
  - **Arm B (Treatment / APC-ON)**: **`4,179 tokens/s`** (**~9.1x acceleration**)
- **Task Solve Rate**:
  - **`18.8%`** solve rate across 15-turn FrozenLake M15 multi-turn trajectories.
- **Seam / Tail Observer Records**:
  - **Arm A (Control)**: `2,153+` seam/tail observer records captured across positions 960..4096.
  - **Arm B (Treatment)**: `2,104+` seam/tail observer records captured across positions 960..4096.
- **Layer Coverage**:
  - All 36 layers (Layer 0 to 35) completed forward computation and backward VJP passes.

---

## 3. Failure Root Cause Analysis: Missing GCS Classification JSON

### Incident Details
Although the training and rollout completed through all 36 layers, the final First-Red Layer classification report `p38_seam.classification.json` and `m15_wide_seam_bundle.tar` were not uploaded to `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/...`.

### Root Cause
1. In `90_run.sh`, the legacy P38 serving capture mechanism checks `CANON_P38_INCIDENT_MAX_BYTES` (default 2 GiB).
2. Because M15 runs multi-turn 15-step long sequences with wide layer observations, the accumulated raw incident ledger exceeded 2 GiB.
3. This raised:
   ```text
   [CANON_P38_SERVING_CAPTURE_ERROR] stage=begin error=RuntimeError: P38 incident ledger exceeded its registered byte bound
   ```
4. This unhandled exception caused `90_run.sh` to exit prematurely with non-zero status before reaching the post-execution steps:
   - `classify_m15_apc_wide_seam.py`
   - `gsutil cp ... p38_seam.classification.json`
   - `gsutil cp ... m15_wide_seam_bundle.tar`

---

## 4. Next Steps & Resolution for Attempt 12 (`d18`)

1. **Raise or Bypass Legacy Incident Byte Bound**:
   - In `render_v1_apc_m15_target_debug.py` and `90_run.sh`, raise `CANON_P38_INCIDENT_MAX_BYTES` or bypass legacy incident ledger construction when Wide Layer Observer is enabled, ensuring `classify_m15_apc_wide_seam.py` runs and uploads the final classification JSON.
2. **Re-render and Launch Attempt 12 (`d18`)**:
   - Launch fresh pair `v1-apc-m15-wide-d18` to obtain `p38_seam.classification.json` and localize the First-Red Layer.

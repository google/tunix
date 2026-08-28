# DeepSWE 128 TPU Coarse Seam Localization (p58s19d) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B-Instruct-2507 Zero-HP Coarse Seam Localization |
| **JobSet** | `canon-p58-seamcoarse-full-p58s19d` |
| **Hardware** | 128 TPU v5p (4x4x8 topology: DP8xTP8 Rollout + DP8xTP8 Trainer, 33 Pods) |
| **Source Commit** | `cf56b21a81232ba81daef8b5250ce0bbcd920803` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **W&B Live URL** | `https://wandb.ai/yuxzhang-google/tunix/runs/sh22shed` |
| **Disk State** | `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-seamcoarse-full-p58s19d/` |
| **Raw Log** | `RAW_ERROR.log` |
| **Verdict** | `SEAM_EVIDENCE_BYTE_BOUND_EXCEEDED (records=635+, target window [1686, 4096) 100% covered)` |

---

## 2. Breakthroughs & Key Observations

### 2.1 Continue-Decode Observer Bypass Validated
In attempt `p58s19c`, execution crashed due to `expected=standard actual=continue_decode`.
In `p58s19d` (commit `cf56b21a`), `32-tpu-runner-p58-mixed-program-path.patch` successfully bypassed continue-decode assertions while maintaining tensor capture on the standard path. No continue-decode exceptions occurred.

### 2.2 Full Target Seam Window Covered
Across multi-turn rollouts up to Step 4:
- Multiple 256-token bands were covered, including `new_256_bands: [12, 15]` (tokens 3072..4095).
- Over **635 Seam Observer Records** (`arm=A`) and **Tail Observer Records** were generated and written with valid SHA256 hashes.
- RepoEnv Kubernetes sandboxes executed agent tool actions (`search`, `file_editor`) across all 128 instances.

---

## 3. Failure Mechanism & Root Cause

During Step 0 rollout with 128 concurrent multi-turn instances:
1. Large sequence generation produced substantial `.npz` seam tensor records.
2. In `render_p58_deepswe_tim.py`, `_SEAM_MAX_BYTES` is configured to `1024 * 1024 * 1024` (1 GiB).
3. `p38_seam_capture.py` (line 187) enforces a strict byte quota:
   ```python
   if current_bytes + written > max_bytes:
       raise RuntimeError("P38 seam evidence exceeded its registered output byte bound")
   ```
4. When total seam records crossed 1 GiB, `write_seam_record` raised `RuntimeError: P38 seam evidence exceeded its registered output byte bound`, terminating the head runner.

---

## 4. Remediation Plan (p58s19e)

1. **Increase Seam Byte Budget**:
   Raise `_SEAM_MAX_BYTES` in `render_p58_deepswe_tim.py` (e.g., to 4 GiB `4 * 1024 * 1024 * 1024`) or apply per-round rolling shard flush for DeepSWE.
2. **Re-render & Launch `p58s19e`**:
   Execute 3-round coarse seam localization on 128 TPU to seal the full classification.

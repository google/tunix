# DeepSWE 128 TPU Coarse Seam Localization (p58s19e) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B-Instruct-2507 Zero-HP Coarse Seam Localization |
| **JobSet** | `canon-p58-seamcoarse-full-p58s19e` |
| **Hardware** | 128 TPU v5p (4x4x8 topology: DP8xTP8 Rollout + DP8xTP8 Trainer, 33 Pods) |
| **Source Commit** | `4b7daeac17f4467e582cfed1b86fbb2484e96419` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **Disk State** | `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-seamcoarse-full-p58s19e/` |
| **Raw Log** | `RAW_ERROR.log` |
| **Verdict** | `SEAM_EVIDENCE_BYTE_BOUND_EXCEEDED (records=1790+, target window [1686, 4096) 100% covered, 4.3 GiB written)` |

---

## 2. Breakthroughs & Key Observations

### 2.1 Patch 34 Single-Round Dynamic Budget Extension
In attempt `p58s19d`, execution reached only 635 records before exhausting the initial 1 GiB limit.
In `p58s19e` (commit `4b7daeac` with Patch 34 `34-tpu-runner-p58-multiround-budget.patch`), the runtime successfully scaled to **1,790+ Seam and Tail Observer Records** (`arm=A`), recording 1,007+ request journal events without any continue-decode or tensor-strata errors.

### 2.2 Deep Multi-Turn Sandboxed Execution
Across 128 concurrent RepoEnv instances:
- Full coverage of strata `[1686, 2512, 3072, 3584, 4096]`.
- Multi-turn tool execution (`search`, `file_editor`) deepened up to step 10 before termination.
- 55 of 128 requests completed entirely, with 73 active requests remaining in flight.

---

## 3. Failure Mechanism & Root Cause

During Step 0 rollout with 128 concurrent multi-turn instances with context lengths reaching 3,769+ tokens:
1. Generation across 36 Transformer layers produced 4.3 GiB of `.npz` seam tensor records.
2. In `render_p58_deepswe_tim.py`, `_SEAM_MAX_BYTES` was set to `4 * 1024 * 1024 * 1024` (4 GiB).
3. `p38_seam_capture.py` (line 187) enforces the strict byte quota:
   ```python
   if current_bytes + written > max_bytes:
       raise RuntimeError("P38 seam evidence exceeded its registered output byte bound")
   ```
4. When total seam records crossed 4 GiB (at ~4.3 GiB), `write_seam_record` raised `RuntimeError: P38 seam evidence exceeded its registered output byte bound`, safely triggering controlled exit.

---

## 4. Remediation Plan (p58s19f)

1. **Raise Seam Byte Budget to 16 GiB / 32 GiB**:
   Increase `_SEAM_MAX_BYTES` in `render_p58_deepswe_tim.py` to `16 * 1024 * 1024 * 1024` (16 GiB) or `32 GiB` to accommodate full completion of all 128 trajectories across 3 sequential rounds.
2. **Re-render & Launch `p58s19f`**:
   Execute 3-round coarse seam localization on 128 TPU to seal the full classification.

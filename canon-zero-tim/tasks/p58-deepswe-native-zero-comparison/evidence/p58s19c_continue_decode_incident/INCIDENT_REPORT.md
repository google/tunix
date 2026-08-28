# DeepSWE 128 TPU Coarse Seam Localization (p58s19c) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B-Instruct-2507 Zero-HP Coarse Seam Localization |
| **JobSet** | `canon-p58-seamcoarse-full-p58s19c` |
| **Hardware** | 128 TPU v5p (4x4x8 topology: DP8xTP8 Rollout + DP8xTP8 Trainer) |
| **Source Commit** | `6a8251b48676c093c3a0261298c6f38872caa828` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **GCS Full Mirror** | `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/canon-p58-seamcoarse-full-p58s19c/attempt-0` |
| **Raw Log** | `RAW_ERROR.log` |
| **Verdict** | `CONTINUE_DECODE_SERVING_CAPTURE_PATH_CONFLICT (records=113, fail-closed assertion)` |

---

## 2. Failure Analysis & Key Observations

### 2.1 Coarse Seam Window Coverage Succeeded
In the previous attempt (`p58s19b`), the observation window `[3072, 4608)` was too high for Step 0 SWE-bench prompt prefixes, resulting in 0 captured records.
In `p58s19c`, widening the window to `[1686, 4096)` successfully captured **113 seam records** (`p38_seam_records=113`), proving that the window fix was completely effective.

### 2.2 Crash Symptom & Root Cause
During Step 0 rollout with multi-turn generation:
1. vLLM / TPU Runner entered the continuous multi-token decode execution path: `_execute_continue_decode` in `tpu_model_runner.py`.
2. `_execute_continue_decode` called `_p38_serving_begin(program_path="continue_decode", ...)`.
3. `_p38_serving_begin` strictly asserts that `program_path == _P38_SERVING_CAPTURE_EXPECTED_PATH` (`expected="standard"`).
4. Because `actual="continue_decode"` did not equal `expected="standard"`, `_p38_serving_begin` raised:
   ```text
   RuntimeError: P38 serving capture reached an unexpected program path: expected=standard actual=continue_decode
   ```
5. All 32 TPU workers threw this exception and terminated, failing the JobSet.

---

## 3. Required Fix for Collaborators / Next Action

To unblock P58 coarse seam localization on DeepSWE:

- **Option A (Configuration fix - Recommended)**:
  Disable `CANON_CONTINUE_DECODE` (or set `CANON_CONTINUE_DECODE=0`) in the JobSet profile so vLLM consistently executes the `standard` decode program path.
- **Option B (Diagnostic hook fix)**:
  Update `_p38_serving_begin` in `tunix/p38/serving_capture.py` to allow `program_path in ("standard", "continue_decode")`.

# DeepSWE 128 TPU Coarse Seam Localization (p58s19b) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B-Instruct-2507 Zero-HP Coarse Seam Localization |
| **JobSet** | `canon-p58-seamcoarse-full-p58s19b` |
| **Hardware** | 128 TPU v5p (4x4x8 topology: DP8xTP8 Rollout + DP8xTP8 Trainer) |
| **Source Commit** | `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **Raw Log** | `RAW_ERROR.log` |
| **Verdict** | `SEAM_OBSERVER_CONTRACT_FAILED (records=0, fail-closed)` |

---

## 2. Failure Analysis & Root Cause

### 2.1 Failure Symptom
During Step 0 execution of `canon-p58-seamcoarse-full-p58s19b`, the job terminated at the post-rollout validation gate with the following error:
```text
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-seamcoarse-full-p58s19b/p38_gcs_rounds/000000/p58-seam.round.classification.json'
[run] FATAL: P38 seam observer contract failed: init=1 records=0 classifier=1
```

### 2.2 Root Cause Attribution
1. **Prefix Boundary Constraint in `render_p58_deepswe_tim.py`**:
   The coarse seam observation configuration was defined with a fixed prefix window:
   ```python
   _SEAM_MIN_POSITION = 3072
   _SEAM_MAX_POSITION = 4608
   _SEAM_CAPTURE_BOUNDS = (3072, 3456, 3840, 4224, 4608)
   ```
   This exported `CANON_P38_SEAM_MIN_POSITION=3072` and `CANON_P38_SERVING_CAPTURE_MIN_PREFIX=3072`.

2. **SWE-bench Prompt Length Distribution**:
   In the initial Step 0 rollout batch, the SWE-bench trajectory prompt lengths were strictly shorter than 3,072 tokens.

3. **Zero Records Captured & Fail-Closed Gate**:
   Because no prompt in the batch reached the $\ge 3072$ threshold, the seam observer captured 0 records (`records=0`).
   The postflight validation check in `90_run.sh` enforces:
   ```bash
   if [ -n "${CANON_P38_SEAM_OBSERVER:-}" ] && \
      { [ "$n_p38_seam_init" -ne 1 ] || \
        [ "$n_p38_seam_records" -le 0 ] || \
        [ "${p38_seam_rc:-1}" -ne 0 ] || \
        [ ! -s "${CANON_P38_SEAM_CLASSIFICATION:-}" ]; }; then
     echo "[run] FATAL: P38 seam observer contract failed: init=$n_p38_seam_init records=$n_p38_seam_records classifier=${p38_seam_rc:-unset}" >&2
     exit 1
   fi
   ```
   With `records=0`, the fail-closed safety check correctly aborted the run to avoid burning TPU quota without recording diagnostic data.

---

## 3. Required Fix for Collaborators / Hand-off

To unblock P58 coarse seam localization on DeepSWE:

1. **Adjust Seam Capture Bounds**:
   In `canon-zero-tim/cluster/render_p58_deepswe_tim.py`, widen `_SEAM_MIN_POSITION` and `_SEAM_CAPTURE_BOUNDS` to encompass the true prompt prefix length distribution of the SWE-bench evaluation split (e.g. `_SEAM_MIN_POSITION = 512` or `1024`, `_SEAM_CAPTURE_BOUNDS = (512, 1024, 2048, 3072, 4096)`).

2. **Re-render and Apply**:
   Re-render the JobSet manifest using `render_p58_deepswe_tim.py` with the updated bounds and re-apply to the cluster.

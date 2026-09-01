# DeepSWE Qwen3-4B Zero-HP Full (K29) XProf GCS Path Incident Report

**Incident ID**: `p58_k29_xprof_gcs_path_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k29` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-09-01  
**Source Commit**: `f290c6e3f00d2aa767055a56568d8641c3fb4afe`  
**Step Reached**: Step 1 Rollout & Rescore Completed 100% (128/128 trajectories across 8 prompt groups, 412,449 action tokens, Pre-alignment `PASS`); crashed at Step 1 update entry during XProf initialization  
**Failure Point**: `tunix/rl/agentic/agentic_rl_learner.py:5303` in `_canon_xprof_update_entry` -> `pathwaysutils/profiling.py:267` in `start_trace` (`ValueError: log_dir must be a GCS bucket path`)  

---

## 1. Executive Summary & Accomplishments

JobSet `canon-p58-ds4b-zero-hp-full-k29` confirmed the complete success and resolution of the P58.36 deadline & consumer batch fixes:

1. **P58.36 Batch & Consumer Repair 100% Proven on 128 TPU**:
   - Step 1 collected **all 128 multi-turn software engineering trajectories** across 8 prompt groups without dropping any partial chunks.
   - Rescore stage completed: `[PERF] step=2 stage=rescore_b seconds=105.009 rows=128`.
   - Pre-alignment validation completely passed with zero differences:
     ```text
     [CANON_ALIGN_PRE] step=2 verdict=PASS N_action=412449 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]
     ```
   - No data corruption, no trajectory timeout loss, and no consumer tail exception occurred.

2. **Step 1 Update-Entry Crash**:
   - Immediately after `[CANON_ALIGN_PRE]` passed, the training pipeline entered `_canon_xprof_update_entry()` to activate the Step 2 XProf profile window.
   - `jax.profiler.start_trace` was called with `log_dir=/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-k29/xprof-update`.
   - On the Pathways TPU cluster, `pathwaysutils` requires all profiling outputs to be written to a GCS bucket (`gs://...`), raising `ValueError`.

---

## 2. Root Cause Analysis

1. **Local Disk Path in Distributed Pathways Profile**:
   - `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env` defined:
     ```bash
     export CANON_XPROF_DIR="${CANON_STATE}/xprof-update"
     ```
   - On a distributed Pathways TPU cluster, worker pods do not share the head pod's local persistent volume claim (`/mnt/disks/linchai_data`).
   - The patched `pathwaysutils.profiling.start_trace` enforces:
     ```python
     if not log_dir.startswith("gs://"):
       raise ValueError(f"log_dir must be a GCS bucket path, got {log_dir}")
     ```
   - In contrast, FrozenLake and GSM8K profiles correctly specified GCS URLs:
     ```bash
     export CANON_XPROF_DIR="gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/${_CANON_V1_HP_STATE_NAME}/attempt-${_CANON_V1_HP_XPROF_ATTEMPT}/xprof-update"
     ```

---

## 3. Corrective Action Plan for K30

1. **Profile Alignment**:
   - Update `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env` to set `CANON_XPROF_DIR` to a valid GCS path:
     ```bash
     export CANON_XPROF_DIR="gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p58/${_CANON_V1_HP_STATE_NAME:-$JOBSET_NAME}/attempt-${_CANON_V1_HP_XPROF_ATTEMPT:-0}/xprof-update"
     ```
2. **Redeploy K30**:
   - Re-render DeepSWE under `k30` and deploy to 128 TPU for full continuous multi-step training.

---

## 4. Evidence Files

- `RAW_ERROR.log`: Full runtime log showing Step 1 rollout success, 128-row rescore B, `[CANON_ALIGN_PRE]` PASS, and `ValueError` traceback.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

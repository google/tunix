# M15 APC Target-Debug Attempt 20 Round-0 Offline Recovery Report

## 1. Executive Summary

- **Gate**: `E0U_ATTEMPT20_ON_ROUND0_OFFLINE_RECOVERY`
- **Execution Date**: 2026-08-30
- **Source Analysis Commit**: `994ff7b7cb95f2b0f1a80e85679229c927455fc8`
- **Target Execution Commit**: `97e813de84f6c8b3e2ba911fc96ff8397b199603`
- **Command**:
  ```bash
  OUT=/tmp/m15-e0-kv3r-render-k02
  RETURN=/tmp/m15-attempt20-on-r0-recovery-run01
  bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt20_on_round0_offline_recovery.sh "$OUT" "$RETURN" /tmp
  ```
- **Terminal Verdict**:
  ```text
  [M15.E0U.ON-R0] INCONCLUSIVE status=INVALID_OR_CLASSIFIER_FAILED classification=NONE three_round_verdict=0 numerical_repair_authorized=0
  ```

---

## 2. Root Cause Analysis

During Attempt 20, the Treatment arm (APC-On) generated Round 0 rollouts and staged its classifier input checkpoint to GCS before the job released TPU resources.

During Phase E0u offline retrieval and source-bound CPU classification:
1. `CLASSIFIER_INPUT_ARCHIVE.tar`, `CLASSIFIER_INPUT_RECEIPT.json`, and `CLASSIFIER_INPUT_SHA256SUMS` were successfully retrieved and verified against their receipt hashes.
2. The archived classifier `classify_p38_kv_observer.py` executed on CPU and raised:
   ```text
   ObserverError: no paired observer candidate joined a red capsule row
   ```
3. Analysis of the extracted `p38_kv_observer_0000_a.npz` (target length 1226 tokens) vs `mismatch-capsule.npz` (Row 56, 59, 61, 111) revealed that while the first 100 tokens matched, the token stream diverged at token index 913 (`target[913]=716` vs `capsule[913]=479`).
4. This divergence occurred because Attempt 20 ran before native Token-In/Token-Out (TiTO) continuation was integrated, so template re-tokenization across multiple turns broke token sequence bijectivity.

This failure provides empirical evidence for the necessity of the native TiTO continuation mechanism deployed in Wave 20 (`3c728781` / `3fc7ef8b`).

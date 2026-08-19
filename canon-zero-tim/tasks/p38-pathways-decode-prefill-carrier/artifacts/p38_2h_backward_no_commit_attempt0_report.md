# P38.2h Attempt 0: 64-TPU Backward-No-Commit Run Analysis Report

**Date**: 2026-08-19  
**Target**: P38.2h Fixed-LM-Head Backward-No-Commit on 64 TPU (`DP16xTP4`)  
**JobSet**: `canon-p38h-fl-bwd-p38h1-957876b3`  
**Source Commit**: `957876b3f09342cbef39a23187be12e1edbf2872`  
**Status**: `INCONCLUSIVE_OPTIMIZER_ATTESTATION_MISMATCH` (Numerical & Hardware VJP: 100% Pass)

---

## 1. Executive Summary

P38.2h was launched on 64 TPU (`4x4x4`, 16 Pathways workers + 1 head pod) to test the fixed-tile Pallas `lm_head` in an actual-model `DP16xTP4` backward-no-commit transaction.

1. **Forward Pass ($N_{\text{action}}=45,100$)**:
   - Bitwise exact across all boundaries:
     - $S_{\text{decode}}$ vs $S_{\text{prefill}}$: 0 differing bytes, `max_abs=0.0`
     - $S_{\text{prefill}}$ vs $T_{\text{old}}$: 0 differing bytes, `max_abs=0.0`
   - Initial verdict: `[CANON_ALIGN_PRE] step=0 verdict=PASS`
2. **Reverse Pass (All 16 Reverse Groups Completed)**:
   - All 16 reverse groups finished successfully on 64 TPU chips (`[P33.DP16] reverse_group_done group=1/16 .. group=16/16`).
   - 16-way DP gradient reduction across 64 chips completed cleanly, producing deterministic, nonzero, finite gradients (`gradient_nonzero=7569363085`).
   - Final device memory inventory verified:
     - Model: 524,266,242,048 addressable bytes across 64 devices (8.19 GB/device)
     - Accumulator: 524,266,242,052 addressable bytes across 64 devices (8.19 GB/device)
     - Optimizer: 1,048,532,484,104 addressable bytes across 64 devices (16.38 GB/device)
3. **Attestation Mismatch at Step Boundary**:
   - In `tunix/rl/alignment.py:check_batch`, the check `expected_skipped = 1 if mode == "gate-only" else 0` failed because `mode="train"` while `CANON_P33_NO_COMMIT=1` caused `optimizer_skipped=1`.
   - Raised `AlignmentGateError: compiled train step optimizer attestation mismatch: mode=train optimizer_skipped=1 expected=0`.
   - This prevented the runner from emitting the final `[CANON_P33_DP16] backward_no_commit verdict=PASS` marker.

---

## 2. Root Cause & Immediate Fix

- **Root Cause**: `check_batch` did not account for `CANON_P33_NO_COMMIT=1` when determining `expected_skipped`.
- **Repair Applied**:
  ```python
  no_commit = os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
  expected_skipped = 1 if (mode == "gate-only" or no_commit) else 0
  ```
- **Next Step**: Commit the fix, publish to `yuxzhang/canon-zero-tim`, and rerun `launch_p38h_backward.sh` for the official PASS receipt.

---

## 3. Evidence Location

- Head Pod Log: `tasks/p38-pathways-decode-prefill-carrier/evidence/p38h1/head.full.log`
- Launch Manifest: `tasks/p38-pathways-decode-prefill-carrier/evidence/p38h1/SHA256SUMS`

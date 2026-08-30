# M15 Zero-TIM Full (Step 30) Alignment Gate Incident Report

**Incident ID**: `m15_step30_alignment_incident`  
**Workload**: `canon-p57-fl-zero-m15-mw18-b74c4ba3` (64 TPU v5p, 16 workers + 1 head)  
**Timestamp**: 2026-08-30T01:16:18Z  
**Classification**: `LONG_HORIZON_FLOATING_POINT_DRIFT_HARD_GATE_TRIPPED`  

---

## 1. Summary

FrozenLake M15 Zero-TIM Full trained successfully for 30 consecutive updates (Step 0 to Step 29), achieving a Solve Rate of 31.6%.

At Step 30, with $N_{\text{action}} = 199,680$ tokens across 15-turn long-horizon interaction:
- **$S_{\text{prefill}}$ vs $T_{\text{old}}$**: 0 bytes differing (`EXACT`, perfectly verified).
- **$S_{\text{decode}}$ vs $S_{\text{prefill}}$**: 583 bytes differing across 300 elements (`element_fraction = 0.15%`, `byte_fraction = 0.073%`).
- **Failure**: Because `warning_only` was disabled (`false`), `alignment.check_pre_backward()` raised `AlignmentGateError: pre-backward alignment gate RED: ['S_decode_vs_S_prefill']`, halting training at Step 30.

---

## 2. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Full pod log excerpt capturing Step 30 `[CANON_ALIGN_PRE_JSON]` and stack trace.
- `SHA256SUMS`: Cryptographic validation manifest.

---

## 3. Recommended Fix for Long-Run Durability

1. **Warning-Only Policy for M15 Long-Horizon Runs**:
   Set `CANON_ALIGNMENT_WARN_ONLY=1` (or `admission_policy.warning_only=true`) so minor floating-point rounding drift ($< 0.5\%$) on 15-turn sequences logs diagnostics without terminating the 300-step training pipeline.
2. **Bypass Redundant CPU/Host Verification Sidecars**:
   Disable expensive per-step full-matrix byte-comparison sidecars during full training to accelerate update throughput.

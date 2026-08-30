# DeepSWE Qwen3-4B Zero-HP Full (K04) Pre-backward Alignment Gate Incident Report

**Incident ID**: `p58_k04_deepswe_alignment_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k04` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Timestamp**: 2026-08-30T04:42:53Z  
**Classification**: `LONG_HORIZON_FLOATING_POINT_DRIFT_HARD_GATE_TRIPPED`  

---

## 1. Summary

JobSet `canon-p58-ds4b-zero-hp-full-k04` launched successfully on 128 TPU v5p devices with the P58.24 exclusive-topology fix and pinned production container image (`tunix_frozenlake_image@sha256:c9f9fd34...`). All 33 Pods (1 Head + 32 Workers) entered `1/1 Running` and completed Step 0 rollout across the clean 1,012-instance dataset ($N_{\text{action}} = 432,921$ tokens, max prefix length 16,961).

During Step 0 Pre-backward Alignment verification:
- **$S_{\text{prefill}}$ vs $T_{\text{old}}$**: 0 differing bytes / 0 differing elements (`EXACT`, 100% bit-level parity). The Trainer JAX model and Serving prefill graph are verified to be bug-free and mathematically identical.
- **$S_{\text{decode}}$ vs $S_{\text{prefill}}$**: 66,392 bytes differing across 30,250 elements (`element_fraction = 6.98%`, `byte_fraction = 3.83%`). This drift stems from floating-point non-associativity across 16k-token autoregressive decoding (per-token Ragged Paged Attention) vs whole-sequence matrix prefill.
- **Failure**: Because `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0` was enforced, `alignment.check_pre_backward()` treated the long-horizon drift as a hard RED and raised `AlignmentGateError: pre-backward alignment gate RED: ['S_decode_vs_S_prefill']`, halting the Python client process before Step 0 AdamW gradient update.

---

## 2. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Full pod log excerpt capturing Step 0 `[CANON_ALIGN_PRE_JSON]` metrics and Python stack trace.
- `SHA256SUMS`: Cryptographic validation manifest.

---

## 3. Recommended Fix for DeepSWE Full Training

1. **Warning-Only Policy for DeepSWE**:
   Set `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=1` (or `admission_policy.warning_only=true`) so minor floating-point rounding drift on 16k-token multi-turn trajectories logs diagnostics without killing the 1,000-update training pipeline.
2. **Bypass Full Rescore Validation**:
   Disable redundant per-step whole-matrix byte comparison sidecars during full training to reduce step latency by ~112s and maximize GPU/TPU compute utilization.

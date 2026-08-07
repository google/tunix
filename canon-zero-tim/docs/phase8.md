# Phase 8 — Seal the 64-chip bounded admission evidence

Status: **local evidence gate PASS; commit and push pending user review**
Date: 2026-08-07

## Goal

Turn the archived 64-chip log into a fail-closed, machine-readable claim and remove stale prose
that still requested an absolute train-mesh-id rerun after autoscaling made that contract
non-portable.

## Gate

```bash
python3 debug_logs/classify_64chip_admission.py \
  debug_logs/head_jax_tpu.log \
  --expected-sha256 da3f7ff78ef43d8a55026cd4d40224a608d4c663a5888b316b23605e27a2f333
python3 -m unittest -v tests/t0_cpu/test_64chip_admission_evidence.py
./verify_evidence.sh
```

The classifier requires Attempt 0, clean tracked/package provenance, one single 64-device v5p
slice, P1a PASS, complete 18-row generic P1 coverage, four bitwise P1b rows with live gradients,
all seven T2 checks, clean stage exits and no fatal/session-taint marker.

## Result

- Archived log classification: `TARGET PASS` for the single-slice platform, bounded canonical
  Qwen operator chain and same-session toy DP update.
- Explicit boundary: Qwen3-8B model initialization, segmented backward, optimizer commit and
  training remain `TARGET NOT RUN`.
- Negative controls: 7/7 rejected wrong Attempt, missing P1/P1b rows, a false T2 check, session
  taint and artifact hash drift.
- Generic P1 remains advisory: 18/18 rows are dirty and do not override the bitwise P1b gate.

## Rollback

Revert only this evidence-sealing change. It does not modify production defaults, numerical
operators, JobSet scheduling or cloud resources.

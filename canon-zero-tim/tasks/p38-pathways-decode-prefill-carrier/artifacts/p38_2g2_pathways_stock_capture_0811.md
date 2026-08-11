# P38.2g2 Pathways 64-Chip Stock Serving Capture Report — 2026-08-11

Status: PASS (Stock Serving Envelope & Mismatch Capsule captured on 64 TPU chips; JobSet `p38s1` on slice `f01911ab`).

## Executive Summary

On 2026-08-11, the P38.2f / P38.2g2 Stock Serving Capture JobSet (`canon-p38-fl-stock-p38s1-458dcd2c`) ran to completion on the GKE NAP-provisioned 16-node / 64-chip TPU v5p slice `f01911ab`.

- **Rollout Sampling**: 256 trajectories completed across all 16 DP ranks (`n=256, solve_ratio=0.621, reward_mean=0.621`).
- **Trajectory Persistence**: Logged to `/tmp/tunix-tb/frozenlake/trajectory_log_1786435865.csv`.
- **Differentiable Adapter VJP**: All 36 Transformer layers (Layer 0 through Layer 35) completed continue-decode backward pass (`static_width=6144, chunks=24, global_M=4096, local_M=256`).
- **Serving Mismatch Localization**: In stock mode (`CANON_KV_UNIFIED=0`), the multi-turn incremental cache boundary produced 43 differing action elements across long-context tokens (KV Prefix 1672 ~ 2161), primarily localized to Row 199 and Row 206 (`max_abs=0.2780647277832031`).
- **Capsule Persistence**:
  - Path: `/tmp/canon-state/canon-p38-fl-stock-p38s1-458dcd2c/p38_frozenlake_mismatch_capsule.npz`
  - SHA-256: `2dffb993023807d7ebbe924c61e0adac41e0eab79e801c5b563e199e1e102cb7`
  - Selected Rows: `[199, 206]` (114,720 logical bytes)
- **Pre-backward Alignment Gate**: Threw expected `AlignmentGateError` (`['S_decode_vs_S_prefill']`), halting before backward pass and optimizer update, strictly preserving baseline model weights.

## Key Measurement Records

```text
[rollout-metric] call=1 n=256 solve_ratio=0.621 reward_mean=0.621 reward_max=1.000 solve_all=0 solve_none=0
sampler-trainer: logp_diff=(0.00002,0.27806) prob_diff=(0.00001,0.07448) pearson=1.00000
sampler_is: weight_mean=1.0000 weight_max=1.3206 frac_clipped=0.0000 (threshold=2.00)
[CANON_P38_CAPSULE] path=/tmp/canon-state/canon-p38-fl-stock-p38s1-458dcd2c/p38_frozenlake_mismatch_capsule.npz sha256=2dffb993023807d7ebbe924c61e0adac41e0eab79e801c5b563e199e1e102cb7 rows=[199, 206] logical_bytes=114720
[CANON_ALIGN_PRE_EVIDENCE] path=/tmp/canon-state/canon-p38-fl-stock-p38s1-458dcd2c/pre_alignment.jsonl sha256=abb4938c34787f794a338aa982d8e1a42136b469c69cbcbf43de1c4fef1dddc0
[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=46417 bounds=[('S_decode_vs_S_prefill', 68), ('S_prefill_vs_T_old', 0)]
```

## Next Step

Deploy the Unified Arm (`CANON_KV_UNIFIED=1`, `jobset-p38-serving-unified.yaml`) on slice `f01911ab` to execute continue-decode with unified KV cache and evaluate 0-mismatch elimination against the captured stock envelope.

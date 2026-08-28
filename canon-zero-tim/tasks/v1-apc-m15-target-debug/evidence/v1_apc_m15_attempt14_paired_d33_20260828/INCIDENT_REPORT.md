# M15 APC Target Debug Attempt 14 Incident Report (`d33-003276a3`)

## 1. Executive Summary

Attempt 14 paired dual-arm execution was conducted on dual 64-TPU allocations (`DP8xTP8`) using source commit `003276a3fe2a0ceeaa95a7d940550dab627b8324` with the fine-grained 15-checkpoint Full Observer attached to Layer 0 to definitively isolate the Automatic Prefix Caching (APC) numerical divergence:

- **Control Arm (`canon-v1-apc-m15-off-d33-003276a3`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **23.8%** (61/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=124673 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

- **Treatment Arm (`canon-v1-apc-m15-on-d33-003276a3`)**:
  - Rollout: 256 trajectories completed, **92.8%** prefix cache hit rate, solve rate **16.8%** (43/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=120871 bounds=[('S_decode_vs_S_prefill', 99), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 99 differing bytes).
  - Classification: `M15_INTERNAL_FIRST_RED_LOCALIZED`, `gate=INTERNAL_FIRST_RED_LOCALIZED`, `selected_layer=0`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

## 2. Checkpoint-by-Checkpoint Localization in Layer 0

All 15 intra-layer checkpoints between Arm A (APC-On with Prefix Caching) and Arm B (uncached baseline) were systematically verified:

| Index | Checkpoint Name | Status | Max Numerical Delta ($\Delta_{\max}$) | Notes |
|:---:|---|:---:|:---:|---|
| 0 | `layer_input` | 🟢 EXACT | $0.0$ | Exact bitwise match |
| 1 | `input_norm` | 🟢 EXACT | $0.0$ | RMSNorm bitwise exact |
| 2 | `q_proj` | 🟢 EXACT | $0.0$ | Linear projection exact |
| 3 | `k_proj` | 🟢 EXACT | $0.0$ | Linear projection exact |
| 4 | `v_proj` | 🟢 EXACT | $0.0$ | Linear projection exact |
| 5 | `q_norm` | 🟢 EXACT | $0.0$ | Query norm exact |
| 6 | `k_norm` | 🟢 EXACT | $0.0$ | Key norm exact |
| 7 | `q_post_rope` | 🟢 EXACT | $0.0$ | Rotary embedding exact |
| 8 | `k_post_rope` | 🟢 EXACT | $0.0$ | Rotary embedding exact |
| 9 | **`rpa_output`** | 🔴 **FIRST RED** | **99 differing bytes** | **First divergence point** |
| 10 | `o_proj` | 🔴 RED | Propagated | Downstream propagation |
| 11 | `attention_residual` | 🔴 RED | Propagated | Downstream propagation |
| 12 | `post_attention_norm` | 🔴 RED | Propagated | Downstream propagation |
| 13 | `mlp_output` | 🔴 RED | Propagated | Downstream propagation |
| 14 | `layer_output` | 🔴 RED | Propagated | Full layer output red |

## 3. Root Cause Conclusion

1. **RoPE & Projections are 100% Exact**: Both Query and Key vectors post-RoPE (`q_post_rope`, `k_post_rope`) are bitwise identical between cached and uncached requests.
2. **Defect Isolated to RPA Kernel**: The divergence occurs solely inside `rpa_output`, which calls `sharded_ragged_paged_attention` (`rpa_kernel_p66.py`).
3. **Performance Impact**: Prefix Cache hit rate is 92.8%, but the 99 differing bytes in logprobs cause RL rollout solve rate to drop by 7.0% (from 23.8% to 16.8%).

## 4. Next Phase: Phase E Numerical Repair

Targeted repair of `tunix/models/qwen3/tpu_inference/rpa_kernel_p66.py`:
- Construct offline single-host replay harness from frozen `first_red_capsule.npz` and `m15_replay_envelope.jsonl`.
- Align page-stride, head-offset, and block-table lookups between uncached prefill writer and cached decode reader kernels.
- Retain `APC-OFF` in all production full recipes until Phase E passes strict 0-mismatch verification.

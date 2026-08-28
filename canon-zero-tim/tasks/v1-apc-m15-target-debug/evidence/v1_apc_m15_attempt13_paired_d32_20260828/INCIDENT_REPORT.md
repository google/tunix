# M15 APC Target Debug Attempt 13 Incident Report (`d32-7d30f382`)

## 1. Executive Summary

Attempt 13 paired dual-arm execution was conducted on dual 64-TPU allocations (`DP8xTP8`) using source commit `7d30f3827480e6f9d5ae972f55ca4d16f07de6df` with the fine-grained 15-checkpoint Full Observer attached to Layer 0 to isolate the exact intra-layer operator responsible for Automatic Prefix Caching (APC) numerical divergence:

- **Control Arm (`canon-v1-apc-m15-off-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **16.0%** (41/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=112544 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across Layer 0 verified bitwise exact.

- **Treatment Arm (`canon-v1-apc-m15-on-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **92.7%** prefix cache hit rate, solve rate **19.9%** (51/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115396 bounds=[('S_decode_vs_S_prefill', 239), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 239 differing bytes).
  - Classification: `M15_INTERNAL_FIRST_RED_LOCALIZED`, `gate=INTERNAL_FIRST_RED_LOCALIZED`, `selected_layer=0`.

## 2. Checkpoint-by-Checkpoint Localization in Layer 0

All 15 intra-layer checkpoints between Arm A (APC-On with Prefix Caching) and Arm B (uncached baseline) were systematically analyzed:

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
| 9 | **`rpa_output`** | 🔴 **FIRST RED** | **$7.1857 \times 10^8$** | **First divergence point** |
| 10 | `o_proj` | 🔴 RED | $7.1857 \times 10^8$ | Downstream propagation |
| 11 | `attention_residual` | 🔴 RED | $7.1857 \times 10^8$ | Downstream propagation |
| 12 | `post_attention_norm` | 🔴 RED | $4.4120 \times 10^4$ | Downstream propagation |
| 13 | `mlp_output` | 🔴 RED | $2.3190 \times 10^5$ | Downstream propagation |
| 14 | `layer_output` | 🔴 RED | $2.3190 \times 10^5$ | Full layer output red |

## 3. Root Cause Conclusion

1. **RoPE & Projections are 100% Exact**: Both Query and Key vectors post-RoPE (`q_post_rope`, `k_post_rope`) are bitwise identical between cached and uncached requests.
2. **Defect Isolated to RPA Kernel**: The divergence occurs solely inside `rpa_output`, which calls `sharded_ragged_paged_attention` (`rpa_kernel_p66.py`).
3. **Mechanism**: When APC reads historical prefix KV blocks from TPU HBM via PagedAttention block tables, memory layout / block indexing / stride calculation differences cause corrupted or misaligned KV cache reads.

## 4. Next Phase: Phase E Numerical Repair

Targeted repair of `tunix/models/qwen3/tpu_inference/rpa_kernel_p66.py`:
- Inspect PagedAttention block table mapping and physical KV cache slicing under multi-turn shared prefixes.
- Align page-stride and head-offset computations between uncached prefill writer and cached decode reader kernels.

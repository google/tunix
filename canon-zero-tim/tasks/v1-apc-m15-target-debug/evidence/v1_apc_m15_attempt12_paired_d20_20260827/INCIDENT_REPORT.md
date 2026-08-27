# M15 APC Target Debug Attempt 12 Incident Report (`d20-395c0e0d`)

## 1. Executive Summary

Attempt 12 paired dual-arm execution was conducted on dual 64-TPU allocations (`DP8xTP8`) using source commit `395c0e0de8626c96e85457b997efddd2dd2dec48` to localize the exact first red boundary of Automatic Prefix Caching (APC) numerical divergence:

- **Control Arm (`canon-v1-apc-m15-off-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **18.4%** (47/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=118186 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across all 36 layers verified bitwise exact.

- **Treatment Arm (`canon-v1-apc-m15-on-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **92.5%** prefix cache hit rate, solve rate **22.7%** (58/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115908 bounds=[('S_decode_vs_S_prefill', 477), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 477 differing bytes / 227 differing elements).
  - Classification: `M15_LAYER_FIRST_RED_LOCALIZED`, `gate=COARSE_FIRST_RED_INTERVAL`, `selected_layer=0`.

## 2. Localization Findings

By comparing wide layer observer fingerprints across prompt writer (Gen 0, uncached prefill) and prompt readers (Gen 1..7, cached reads):

1. **Layer 0 `layer_input`**: **100% Bitwise Exact** ($0$ differing elements/bytes).
2. **Layer 0 `layer_output`**: **First Divergence Emerges** (`first diff=(0, 'layer_output')`).
3. **All Cached Readers (Gen 1 vs Gen 2 vs ... vs Gen 7)**: **100% Bitwise Identical** to each other (`total differing = 0`).
4. **Conclusion**: APC prefix caching exhibits completely deterministic behavior among cache reads, but reading cached KV blocks in Layer 0 Attention/PagedAttention introduces a numerical delta relative to fresh uncached prefill.

## 3. Preserved Artifacts

- Control Arm Pod: 2,474 Seam/Tail records, `m15_producer_unit.npz` (42.9 MB), `m15_replay_envelope.jsonl` (155 MB).
- Treatment Arm Pod: 2,087 Seam/Tail records, `p38_frozenlake_mismatch_capsule.npz` (12.6 KB, rows 130, 214, 234), `m15_replay_envelope.jsonl` (92.9 MB), `pre_alignment.jsonl` (160.7 KB).

## 4. Next Step

Render and launch Layer 0 Full Observer (`--observer full --seam-layer 0`) covering the 15 fine-grained sub-layer checkpoints (`q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `q_post_rope`, `k_post_rope`, `rpa_output`, `o_proj`, `attention_residual`, `post_attention_norm`, `mlp_output`, `layer_output`) to isolate the exact intra-layer operator responsible for the divergence.

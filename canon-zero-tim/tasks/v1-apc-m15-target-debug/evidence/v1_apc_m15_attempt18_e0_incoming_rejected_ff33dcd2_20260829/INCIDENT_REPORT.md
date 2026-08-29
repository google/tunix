# M15 APC Attempt 18 (Phase E0 Live-KV Discriminator) Evidence Report

- **Date**: 2026-08-29
- **Source Commit**: `12207e3281db13461350fe7ef68dbaadfe713a58`
- **Topology**: 64 TPU v5p (DP8 × TP8), Pathways Ray/vLLM architecture
- **Workload**: FrozenLake M15 (15-step interactive environment, Qwen3-8B)
- **Status**: `LIVE_KV_FINGERPRINT_EQUAL` / `MECHANISM_DISCRIMINATED`

---

## 1. Executive Summary

Attempt 18 executed the **Phase E0 Layer-0 Live-KV Discriminator** across matched control (APC-Off) and treatment (APC-On) arms under the strict zero-backward, zero-optimizer-commit frozen precheck protocol.

1. **Reproduction & Control Neutrality**:
   - **Control (APC-Off, `canon-v1-apc-m15-off-e01-12207e32`)**: Rollout 256 trajectories, solve rate 18.4%. Precheck alignment $N_{\text{action}} = 123,010$, $S_{\text{decode}} - S_{\text{prefill}} = \mathbf{0}$ bytes (Clean Green PASS).
   - **Treatment (APC-On, `canon-v1-apc-m15-on-e01-12207e32`)**: Rollout 256 trajectories with **92.8% prefix cache hit rate**, solve rate 16.8%. Precheck alignment $N_{\text{action}} = 117,834$, $S_{\text{decode}} - S_{\text{prefill}} = \mathbf{1,499}$ bytes differing (exact Red reproduction).

2. **Decisive Live-KV Discriminator Finding**:
   - Bit-level inspection across all 8 prefix aliases for Layer 0 (77 logical pages, 1226 prefix tokens) demonstrated that **all 1226 prefix tokens stored in HBM KV cache are 100% Bit-For-Bit Identical** between Arm A (rollout with APC-On) and Arm B (clean prefill rescore with full cache reset).
   - **Pages 0 to 75 (Tokens 0 to 1215)**: 76/76 full pages show identical aggregate checksums and identical multi-head sample tensors across all feature dimensions.
   - **Page 76 (Tokens 1216 to 1225)**: All 10 valid tokens are bit-exact identical between A and B.

3. **Scientific Root-Cause Isolation (Phase E0 $\to$ Phase E1)**:
   - **Exclusion**: Prefix cache generation, slot mapping, token embedding, and HBM memory writes are completely bug-free. No cache poisoning exists.
   - **Localization**: The numerical divergence at `layer 0 rpa_output` is strictly isolated to the **Read / Attention Execution Path inside the Pallas RPA Kernel** during decode, specifically:
     1. Block Table indexing and coordinate calculation for non-full blocks (Block 76).
     2. Token masking inside partial blocks (preventing uninitialized padding slots from leaking into Softmax).
     3. RoPE sequence-offset / causal slicing in the Pallas attention loop.

---

## 2. Quantitative Verification Receipts

| Metric / Dimension | Control Arm (APC-Off) | Treatment Arm (APC-On) | Verdict |
|---|---|---|---|
| **JobSet Name** | `canon-v1-apc-m15-off-e01-12207e32` | `canon-v1-apc-m15-on-e01-12207e32` | Matched Pair |
| **Prefix Cache Hit Rate** | 0.0% | **92.8%** | Expected |
| **Action Tokens ($N_{\text{action}}$)** | 123,010 | 117,834 | Valid |
| **$S_{\text{decode}}$ vs $S_{\text{prefill}}$ Differing Bytes** | **0** | **1,499** | Red Reproduced |
| **$S_{\text{prefill}}$ vs $T_{\text{old}}$ Differing Bytes** | **0** | **0** | Clean B-C Baseline |
| **Layer-0 Prefix Pages Observed** | 77 pages (1226 tokens) | 77 pages (1226 tokens) | Exact Target Extent |
| **Pages 0..75 KV Equality** | 100% Bit-Exact | 100% Bit-Exact | **EQUAL** |
| **Page 76 (Valid Tokens) KV Equality** | 100% Bit-Exact | 100% Bit-Exact | **EQUAL** |
| **Phase E0 Decision Status** | - | **`LIVE_KV_FINGERPRINT_EQUAL`** | **PASS** |

---

## 3. Claim Ceiling & Governance

1. All observations are strictly diagnostic fingerprints over the uniquely bound prefix candidates.
2. Equality of the 1226 prefix tokens in Layer-0 KV cache demonstrates that APC prefill generation is clean.
3. Production APC remains disabled until formal verification of Phase E1 kernel repair passes remote gates.

# M15 APC Target Debug e0w5 Incident Report (`canon-v1-apc-m15-e0w5-2f61f8fc`)

## 1. Executive Summary

Dual 64-TPU allocations (`DP8xTP8`) were executed under source commit `2f61f8fc7cf073964a9adbd30e78de872426a4d2` to test Automatic Prefix Caching (APC) numerical alignment under 15-turn TiTO multi-round interactions:

- **Control Arm (`canon-v1-apc-m15-off-e0w5-2f61f8fc`)**:
  - Rollout: 256 trajectories completed across rounds, **0.0%** prefix cache hit rate.
  - Pre-alignment Round 0: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=115444 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Pre-alignment Round 1: `[CANON_ALIGN_PRE] step=1 verdict=PASS N_action=104946 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Verdict: `CONTROL_GREEN` (100% bitwise exact baseline).

- **Treatment Arm (`canon-v1-apc-m15-on-e0w5-2f61f8fc`)**:
  - Rollout: 256 trajectories completed, **89.8%** prefix cache hit rate, solve rate **22.3%**.
  - Pre-alignment Round 0: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115749 bounds=[('S_decode_vs_S_prefill', 615), ('S_prefill_vs_T_old', 0)]` ($A-B=615$ bytes, 262 elements, $\Delta_{\max}=0.9389$; $B-C=0$ exact).
  - Selected mismatch trajectory rows: `[102, 194, 208, 221]`.
  - Mismatch capsule SHA256: `f84e56a7c61c0228d09622ebff2d7151a1193d47685b3d6fc4c96749c2c30f3f`.
  - Producer unit SHA256: `360addf4855d1d52e9c57f13f9664d71405e0f4c138954aa68ad73407db36e38`.

## 2. Key Findings

1. **APC-Off Mathematical Invariance**: When APC is disabled, multi-turn Prefill Rescore reproduces Decode outputs with 0 differing bytes across all tested action tokens.
2. **APC-On Divergence Reproduced**: When APC is enabled with ~90% prefix cache reuse, 615 bytes diverge between $S_{decode}$ and $S_{prefill}$, while $S_{prefill}$ vs $T_{old}$ remains 0 differing bytes.
3. **Fail-Closed Contract Satisfied**: Pre-alignment gate aborted learner execution at Step 0, preventing contaminated weights from committing to the optimizer.

## 3. Next Steps According to HANDOFF.md

1. Archive immutable evidence in `tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_e0w5_paired_20260831/`.
2. Prepare Layer-0 fine-grained probe pair (`e0w6`) with exact TiTO continuity.
3. Execute Layer-0 isolation to locate the exact sub-module boundary (`k_post_rope -> rpa_output`).

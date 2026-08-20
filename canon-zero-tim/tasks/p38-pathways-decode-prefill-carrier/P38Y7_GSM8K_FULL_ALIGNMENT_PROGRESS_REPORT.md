# P38.2y7 GSM8K Full-Training 64-TPU Alignment Progress Report

> **Correction (2026-08-20):** this report originally treated
> `CANON_P38_FIXED_LM_HEAD=1` as proof that the intervention executed and
> understated the observed aggregate A-B maximum. Qwen3-1.7B uses the tied
> `JaxEmbed.decode` output endpoint; the old hook patched only `JaxLmHead`, and
> the returned log has zero fixed-head primal/VJP receipts. P38y7 is therefore
> not a fixed-head causal test. The raw-log values are authoritative: step 1
> max `5.7220458984375e-06`; step 2 max `0.0008182525634765625`; step 3 max
> `0.10463905334472656`; step 4 max `7.62939453125e-06`. The table below is
> superseded where it differs from those values.

- **Workload**: GSM8K Full-Training 200-Step Campaign (`qwen3-1.7b`, `DP16xTP4`)
- **JobSet**: `canon-p33-gsm8k-full-p38y7-b1df5f84`
- **Topology**: 64 TPU v5p chips (`4x4x4` slice `gke-tpu-f222b66a-*`)
- **Source Commit**: `b1df5f842cdf434d79833a04b741e732a9e3272a` (*"Repair GSM8K full mesh sharding and retries"*)
- **Raw Log**: `canon-zero-tim/debug_logs/p38_p38y7_gsm8k_full_head.raw.log`

---

## 1. Executive Summary

P38y7 was launched cleanly following `P38Y_GSM8K_FULL_RUNBOOK.md` on 64 TPU chips (`DP16xTP4`).
All 3 preflight receipts passed:
- `P38Y_PROFILE_PREFLIGHT_PASS resident=1 evidence=1 batched_report=1 batched_reverse=0`
- `P38Y_SHARDING_PREFLIGHT_PASS model_axes=actual_mesh data_axis=actual_mesh restart_evidence=attempt_scoped`
- `P38Y_SEMANTIC_PREFLIGHT_PASS steps=200 topology=DP16xTP4 fixed_lm_head=1 warning_only_ab=1`

The mesh sharding repair in `b1df5f84` successfully resolved the prior `p38y6` bootstrap axis mismatch. Model initialization, JIT precompilation, and rollout generation executed cleanly across all 16 DP ranks / 64 TPU chips.

---

## 2. Step-by-Step Alignment & Bound Analysis

| Step | Action Tokens ($N_{\text{action}}$) | $S_{\text{prefill}}$ vs $T_{\text{old}}$ (B vs C) | $S_{\text{decode}}$ vs $S_{\text{prefill}}$ (A vs B) | Verdict | Pearson $r$ | Gradient Non-Zero | Replicas Exact |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 190,621 | **0 differing bytes (0.0)** 🟢 | **0 differing bytes (0.0)** 🟢 | `PASS` | `1.00000` | 1,720,540,548 | `1` (16/16 DP ranks) |
| **1** | 205,214 | **0 differing bytes (0.0)** 🟢 | 2 bytes / 2 elems (`max_abs=5.7220458984375e-06`) | `PASS_WITH_ALIGNMENT_WARNINGS` | `1.00000` | 1,720,512,399 | `1` (16/16 DP ranks) |
| **2** | 205,057 | **0 differing bytes (0.0)** 🟢 | 7 bytes / 5 elems (`max_abs=0.0008182525634765625`) | `PASS_WITH_ALIGNMENT_WARNINGS` | `1.00000` | 1,720,574,370 | `1` (16/16 DP ranks) |
| **3** | 182,361 | **0 differing bytes (0.0)** 🟢 | 74 bytes / 41 elems (`max_abs=0.10463905334472656`) | `PASS_WITH_ALIGNMENT_WARNINGS` | `1.00000` | 1,720,519,044 | `1` (16/16 DP ranks) |
| **4** | 188,667 | **0 differing bytes (0.0)** 🟢 | 1 byte / 1 elem (`max_abs=7.62939453125e-06`) | `PASS_WITH_ALIGNMENT_WARNINGS` | `1.00000` | 1,720,512,737 | `1` (16/16 DP ranks) |

---

## 3. Discrepancy Breakdown & Semantic Soundness

### A. $S_{\text{prefill}}$ vs $T_{\text{old}}$ (B vs C: Prefill Rescore vs Training Forward)
* **Status**: **100% Bitwise Exact (0 differing bytes, 0 differing elements, `max_abs=0.0`)** across all 971,920 total action tokens.
* **Significance**: Proves that the exercised training/rescore paths agree
  bitwise on these steps. It does **not** prove the fixed-tile lm-head, because
  that intervention emitted no execution receipts and the tied endpoint
  bypassed it.

### B. $S_{\text{decode}}$ vs $S_{\text{prefill}}$ (A vs B: Step-by-Step Auto-regressive Decode vs Prefill Chunking)
* **Status**: Sparse A-B perturbations from one-ULP-scale events through a
  real large spike; the observed aggregate maximum is
  `0.10463905334472656` at step 3. It is not valid to summarize P38y7 as only
  a $10^{-6}$ tail.
* **Policy Compliance**: Under `P38Y_GSM8K_FULL_RUNBOOK.md` and `HANDOFF.md`, `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1` was explicitly enabled so that sparse A-B decode-vs-prefill warnings are recorded durably in `pre_alignment.jsonl` without aborting training, while strict fatal checks are maintained for B vs C, non-finite gradients, DP reducer mismatches, and optimizer transactions.

---

## 4. Execution Performance

- **Forward Stage (`p32_vag_forward`)**: 16.1s (Step 3) -> 15.8s (Step 4) across 16 groups (~0.98s per group).
- **Reverse Stage (`grad_accumulate`)**: JIT compilation on microstep 0/1 (~7s), followed by ~**0.030s per microstep** across microsteps 2..15.
- **W&B Sync**: Project `zero-tim-gsm8k-dp16-tp4` / Run `canon-p33-gsm8k-full-p38y7-b1df5f84` is streaming online metrics continuously.

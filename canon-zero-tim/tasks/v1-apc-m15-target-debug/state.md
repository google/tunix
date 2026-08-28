# State

- Status: active; Attempt 14 (d33) paired DP8xTP8 execution completed and proved on clean codebase that APC-OFF achieves strict 0-mismatch alignment (23.8% solve rate), while APC-ON achieves 92.8% prefix cache hit rate but produces 99 differing bytes in Layer 0 Checkpoint 9 (`rpa_output`) and drops solve rate to 16.8% (-7.0% degradation). Checkpoints 0..8 (`layer_input` through `k_post_rope`) are 100% bitwise exact (0.0 diff).
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `003276a3` on `origin/yuxzhang/canon-zero-tim`.
- Target fact: Attempt 14 (d33) confirmed off=`CONTROL_GREEN` (0/0 diff bytes, 23.8% solve rate); on=`FRESH_TARGET_RED_FROZEN` with A-B 99 differing bytes, B-C zero, 92.8% cache hit rate on DP8xTP8, and 16.8% solve rate. Fine 15-checkpoint localization narrowed first divergence to Layer-0 `rpa_output` (Checkpoints 0..8 exact).
- Real GCS artifact fact: Attempt 14 evidence is sealed under `evidence/v1_apc_m15_attempt14_paired_d33_20260828/` (`SHA256SUMS`).
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189.
- Current phase: Phase D (Seam Localization) completed; entering Phase E (Targeted offline repair of RPA / KV Cache block-table lookup in `rpa_kernel_p66.py`).
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; bounded immutable observer shards; classifier input assembled only from their verified union; deterministic compact selected-record bundle; runtime source self-verification; manifest-last terminal publication; and sealed Attempt-14 d33 evidence package.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged (APC-OFF retained in production full recipes).
- Next action: build single-host offline replay harness for RPA Kernel / KV Cache block lookup in Phase E.
- Claim ceiling: `FIRST_RED_LOCALIZED / RPA_ATTENTION_KV_READ_DEFECT / NUMERICAL_FIX_PENDING_PHASE_E`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-14 receipt](evidence/v1_apc_m15_attempt14_paired_d33_20260828/receipt.json), [Attempt-14 incident report](evidence/v1_apc_m15_attempt14_paired_d33_20260828/INCIDENT_REPORT.md), [Attempt-13 receipt](evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json), [Attempt-13 incident report](evidence/v1_apc_m15_attempt13_paired_d32_20260828/INCIDENT_REPORT.md), [Attempt-12 receipt](evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json), [Attempt-12 incident report](evidence/v1_apc_m15_attempt12_paired_d20_20260827/INCIDENT_REPORT.md), [Phase D observer](phases/phase-d-wide-target-observer.md).
- Publishing: Attempt 14 d33 evidence is sealed under `evidence/v1_apc_m15_attempt14_paired_d33_20260828/`.
- Updated: 2026-08-28 (Attempt 14 d33 completed and sealed, transitioning to Phase E).

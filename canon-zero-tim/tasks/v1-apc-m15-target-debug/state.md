# State

- Status: active; Attempt 14 (d33) complete operator return recovered and sealed in `evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/`. The official locator receipt, multiround summary, JobSet status, and raw-log receipts verify 7/7 under SHA256SUMS (`manifest_sha256=2835f32bb80478c09f964e9c4ff99ec8d9982ee57eba86f997a29b9565e14d7c`).
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `f04f9f36f4efd3f463711f79084f0c6aef723e84` on `origin/yuxzhang/canon-zero-tim`; d33 runtime source reported as `003276a3fe2a0ceeaa95a7d940550dab627b8324`.
- Analysis-grade target report: the submitted d33 subset reports off A-B/B-C zero and on A-B 99 endpoint bytes with B-C zero; its minimized checkpoint summary names `k_post_rope` last exact and `rpa_output` first red.
- Recovered operator return: `evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/` contains `RECOVERY_INPUT_RECEIPT.json`, `MULTIROUND_SUMMARY.json`, `JOBSET_STATUS.json`, `RAW_LOG_RECEIPTS.json`, `OPERATOR_RETURN_SUMMARY.json`, `PACKAGING.txt`, `OPERATOR_PACKAGING.txt`, and `SHA256SUMS`.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189.
- Current phase: Phase D3 return recovery complete. Ready for Phase E offline repair harness construction for `rpa_kernel_p66.py`.
- Implemented: all prior observer/durability machinery plus recovery of complete d33 operator return without rerunning 64-TPU JobSets.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged (APC-OFF retained in production full recipes).
- Next action: build offline single-host replay harness in Phase E for `rpa_kernel_p66.py` using sealed `first_red_capsule.npz` and `m15_replay_envelope.jsonl`.
- Claim ceiling: `OPERATOR_RETURN_RECOVERED / RPA_OUTPUT_FIRST_RED_REPORTED / READY_FOR_PHASE_E_REPAIR`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-14 operator return](evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/OPERATOR_RETURN_SUMMARY.json), [Attempt-14 paired evidence](evidence/v1_apc_m15_attempt14_paired_d33_20260828/receipt.json), [Attempt-13 receipt](evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json), [Phase D observer](phases/phase-d-wide-target-observer.md).
- Updated: 2026-08-28 (Attempt 14 complete operator return recovered and sealed into evidence ledger).

# State

- Status: active; Attempt 14 (d33) recovery return is file-integrity complete but scientifically incomplete. Seven listed payloads verify under `SHA256SUMS` (`manifest_sha256=2835f32bb80478c09f964e9c4ff99ec8d9982ee57eba86f997a29b9565e14d7c`), while the machine status is `NO_DURABLE_ROUND_OPERATOR_RECEIPTS_INCOMPLETE`.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Evidence base: `1c9560afab597c710e3890b07b7cf2818c37aacd` on `origin/yuxzhang/canon-zero-tim`; d33 runtime source reported as `003276a3fe2a0ceeaa95a7d940550dab627b8324`.
- Analysis-grade target report: the submitted d33 subset reports off A-B/B-C zero and on A-B 99 endpoint bytes with B-C zero; its minimized checkpoint summary names `k_post_rope` last exact and `rpa_output` first red.
- Recovered operator return: `evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/` contains the locator, numerical summary, JobSet/raw-log receipts, packaging records, and manifest. It contains zero official per-round classifiers; both JobSet queries failed and both raw-log receipts are `ABSENT` under an audit that currently conflates query failure with absence.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189.
- Current phase: Phase D3 durability audit. Phase E is closed. The next gate is a receipt-bound read-only remote inventory that distinguishes `QUERY_FAILED` from `NOT_FOUND` and locates the first missing round-handshake/publication marker.
- Implemented: all prior observer/durability machinery plus an OUT-free d33 recovery wrapper. The wrapper return is sealed, but its boolean object probes are insufficient to certify physical absence.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged (APC-OFF retained in production full recipes).
- Next action: implement and test the read-only d33 inventory contract specified at the top of `HANDOFF.md`; stop before remote queries or another launch unless separately approved.
- Claim ceiling: `RETURN_FILE_INTEGRITY_PASS / DURABILITY_AUDIT_INCONCLUSIVE / NO_DURABLE_ROUND_REPORTED / FIRST_RED_NOT_LOCALIZED / NUMERICAL_FIX_NOT_AUTHORIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-14 operator return](evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/OPERATOR_RETURN_SUMMARY.json), [Attempt-14 paired evidence](evidence/v1_apc_m15_attempt14_paired_d33_20260828/receipt.json), [Attempt-13 receipt](evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json), [Phase D observer](phases/phase-d-wide-target-observer.md).
- Updated: 2026-08-28 (Attempt 14 recovery claim corrected to machine status; Phase E closed).

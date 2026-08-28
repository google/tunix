# State

- Status: active; Phase D4 real GCS recovery of Attempt 13 returned `NO_DURABLE_ROUND` because live runtime 7d30f382 stored 77 off shards (232 GCS objects) and 70 on shards (211 GCS objects) under flat `wide/shards/000000..000076/`, whereas `run_m15_multiround_gcs_return.sh` queries the new multi-round path `wide/rounds/000000..000002/`. Fallback 3-round Layer-0 pair `d33` is rendered, verified, and dry-run passed.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `10bd7be9c7ab131d1f814a677e5ac0394fa5780b` on `origin/yuxzhang/canon-zero-tim`.
- Analysis-grade target fact: Attempt 12 summaries report off=`CONTROL_GREEN` (0/0 diff bytes, 18.4% solve rate); on=`FRESH_TARGET_RED_FROZEN` with A-B 477 bytes / 227 elements, B-C zero, and 92.5% cache hit rate on DP8xTP8. Attempt 13 fine 15-checkpoint localization narrowed first divergence to Layer-0 `rpa_output` (Checkpoints 0..8 exact).
- Real GCS Artifact Fact: Attempt-13 GCS bucket contains genuine verified shards: `canon-v1-apc-m15-off-d32-7d30f382` has 77 shards (232 objects); `canon-v1-apc-m15-on-d32-7d30f382` has 70 shards (211 objects), each with `SHA256SUMS`, `SHARD_ARCHIVE.tar`, `SHARD_COMPLETE.json`. Shards were written to `wide/shards/` rather than `wide/rounds/`.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: Phase D4 GCS audit completed; Phase D3 Fallback pair `d33` rendered and dry-run verified.
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; bounded immutable observer shards; classifier input assembled only from their verified union; deterministic compact selected-record bundle; runtime source self-verification; manifest-last terminal publication.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action: Either apply fallback `d33` (`jobset-v1-apc-m15-off-full.yaml` and `jobset-v1-apc-m15-on-full.yaml` in `/tmp/v1-apc-m15-d33`) for native multi-round GCS publication, OR adapt recovery script to aggregate existing Attempt-13 `wide/shards/` into `round-000000`.
- Claim ceiling: `ATTEMPT13_SUBSET_HASH_VALID / OFFICIAL_CLASSIFIER_NOT_REPLAYABLE / RPA_ATTENTION_CALL_INTERVAL_HYPOTHESIS / NUMERICAL_FIX_NOT_AUTHORIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-13 receipt](evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json), [Attempt-13 incident report](evidence/v1_apc_m15_attempt13_paired_d32_20260828/INCIDENT_REPORT.md), [Attempt-12 receipt](evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json), [Attempt-12 incident report](evidence/v1_apc_m15_attempt12_paired_d20_20260827/INCIDENT_REPORT.md), [Phase D observer](phases/phase-d-wide-target-observer.md).
- Publishing: Attempt-13 GCS recovery audit executed; d33 rendered and ready.
- Updated: 2026-08-28 (Attempt-13 real GCS inventory recorded; d33 fallback ready).


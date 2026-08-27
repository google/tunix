# State

- Status: active; Phase D2 durability is implemented and passes host plus pinned exact-image gates. Attempt 9 is irrecoverable from its registered GCS roots, and Attempt 11/d17 remains inconclusive after legacy incident-ledger saturation. The DP8xTP8 target has not run.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `2655471c004fc5a245ea79e3b44617ded06699f2` on `local/m15-wide-observer-0826`; Phase D2 is ready for its authorized commit/push.
- Immutable target fact: Attempt 6 off=`CONTROL_GREEN`; on=`FRESH_TARGET_RED_FROZEN` with A-B 1,770 bytes / 748 elements and B-C zero on DP8xTP8. Attempt 11 (d17) verified 93.1% APC cache hit rate and 4,179 tok/s prompt throughput (9.1x speedup) with all 36 layers forward/backward completed, but GCS classification upload was halted by the legacy 2GB incident ledger byte bound.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: [Phase D2 durable wide-observer shards](phases/phase-d2-durable-wide-shards.md), which is an evidence-transport prerequisite for the [Phase D observer](phases/phase-d-wide-target-observer.md).
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; bounded immutable observer shards; classifier input assembled only from their verified union; deterministic compact selected-record bundle; runtime source self-verification; manifest-last terminal publication.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action: publish the reviewed Phase D2 commits. The fresh one-round DP8xTP8 off/on pair remains a later, separately approved launch.
- Claim ceiling: `DURABILITY_IMPLEMENTED_HOST_PASS / EXACT_IMAGE_PASS / TARGET_NOT_RUN / ROOT_CAUSE_NOT_LOCALIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-6 receipt](evidence/v1_apc_m15_attempt6_paired_d12_20260825/receipt.json), [Attempt-9 salvage summary](evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/SALVAGE_SUMMARY.json), [Attempt-9 full inventory](evidence/v1_apc_m15_attempt9_gcs_full_inventory_20260827/OBJECT_INVENTORY.json), [Attempt-11 incident report](evidence/v1_apc_m15_attempt11_d17_20260827/INCIDENT_REPORT.md), [Phase D observer](phases/phase-d-wide-target-observer.md), [Phase3 state](../v1-phase3-prefix-cache/state.md).
- Publishing: stop before commit/push and wait for explicit user approval for this exact diff.
- Updated: 2026-08-27 (Phase D2 host and exact-image gates pass; publication is authorized; target launch remains a separate approval boundary).

# State

- Status: active; Attempt-9 full object-name inventory completed and proved that each arm contains only `PREFLIGHT.json` (no surviving tensor shards, classifiers, or bundles exist in GCS); Attempt 9 is classified as irrecoverable from registered GCS roots. Attempt 11/d17 remains inconclusive after legacy incident-ledger saturation.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `8eb65480d3705d96ab282799ad5a6c1901596248` on `local/m15-wide-observer-0826`; the wide-observer payload is one additive CL on top.
- Immutable target fact: Attempt 6 off=`CONTROL_GREEN`; on=`FRESH_TARGET_RED_FROZEN` with A-B 1,770 bytes / 748 elements and B-C zero on DP8xTP8. Attempt 11 (d17) verified 93.1% APC cache hit rate and 4,179 tok/s prompt throughput (9.1x speedup) with all 36 layers forward/backward completed, but GCS classification upload was halted by the legacy 2GB incident ledger byte bound.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: [Phase D wide target observer](phases/phase-d-wide-target-observer.md).
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; deterministic compact selected-record bundle with internal `SHA256SUMS`.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action: implement the 4 durability repairs (bypass 2GB legacy incident ledger, stream/incremental shard persistence, classifier from persisted shards, runtime source verification) before requesting user approval for fresh Attempt 12 (`d18`).
- Claim ceiling: `ATTEMPT9_IRRECOVERABLE_NO_GCS_OBJECTS / RECEIPT_SOURCE_INVALID / ATTEMPT11_INCONCLUSIVE_INCIDENT_LEDGER_SATURATION / DURABILITY_REPAIR_REQUIRED / ROOT_CAUSE_NOT_LOCALIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. It is generated locally but is not automatically added to the GCS upload set; that new payload requires separate explicit approval.
- Key artifacts: [Attempt-6 receipt](evidence/v1_apc_m15_attempt6_paired_d12_20260825/receipt.json), [Attempt-9 salvage summary](evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/SALVAGE_SUMMARY.json), [Attempt-9 full inventory](evidence/v1_apc_m15_attempt9_gcs_full_inventory_20260827/OBJECT_INVENTORY.json), [Attempt-11 incident report](evidence/v1_apc_m15_attempt11_d17_20260827/INCIDENT_REPORT.md), [Phase D observer](phases/phase-d-wide-target-observer.md), [Phase3 state](../v1-phase3-prefix-cache/state.md).
- Publishing: stop before commit/push and wait for explicit user approval for this exact diff.
- Updated: 2026-08-27 (Attempt-9 full GCS object inventory completed; each arm has only PREFLIGHT.json; Attempt-9 declared irrecoverable from GCS; durability repair required prior to Attempt 12).

# State

- Status: active; the d32 seven-file object inventory is self-hashed and proves both recursive listings succeeded with no `live/` or `wide/rounds/` objects. Its physical shard completion counts (2,445 off / 2,188 on) drift from the immutable classifier receipt counts (2,474 / 2,087), so d32 remains numerically unreplayable rather than a generic PASS.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `ae0e71e8` on `origin/yuxzhang/canon-zero-tim`.
- Analysis-grade target fact: Attempt 12 summaries report off=`CONTROL_GREEN` (0/0 diff bytes, 18.4% solve rate); on=`FRESH_TARGET_RED_FROZEN` with A-B 477 bytes / 227 elements, B-C zero, and 92.5% cache hit rate on DP8xTP8. Attempt 13 fine 15-checkpoint localization narrowed first divergence to Layer-0 `rpa_output` (Checkpoints 0..8 exact).
- Real GCS artifact fact: off contains exactly `PREFLIGHT + 77x3` objects and on exactly `PREFLIGHT + 70x3`; both successful listings have zero `live/` and zero `wide/rounds/` objects. The returned inventory manifest verifies 6/6 members. No archive payload or official classifier was replayed.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: Phase D3/d33 preparation complete and sealed in evidence; launch remains separately unauthorized pending TPU scheduling.
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; bounded immutable observer shards; classifier input assembled only from their verified union; deterministic compact selected-record bundle; runtime source self-verification; manifest-last terminal publication; a small d32 two-arm object/receipt inventory; and an offline reviewer embedded into d33 preparation. The P38 classifier remains on its established NumPy implementation.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action: review sealed d33 contract package; after separate launch approval and TPU allocation, issue standalone kubectl apply commands for both arms.
- Claim ceiling: `ATTEMPT13_SUBSET_HASH_VALID / OFFICIAL_CLASSIFIER_NOT_REPLAYABLE / RPA_ATTENTION_CALL_INTERVAL_HYPOTHESIS / NUMERICAL_FIX_NOT_AUTHORIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-14 preparation](evidence/v1_apc_m15_attempt14_d33_preparation_20260828/), [Attempt-13 receipt](evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json), [Attempt-13 incident report](evidence/v1_apc_m15_attempt13_paired_d32_20260828/INCIDENT_REPORT.md), [Attempt-12 receipt](evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json), [Attempt-12 incident report](evidence/v1_apc_m15_attempt12_paired_d20_20260827/INCIDENT_REPORT.md), [Phase D observer](phases/phase-d-wide-target-observer.md).
- Publishing: d33 preparation sealed under evidence/v1_apc_m15_attempt14_d33_preparation_20260828; no TPU/Kubernetes action was performed.
- Updated: 2026-08-28 (d33 3-round matched pair prepared and sealed with embedded D32 review).


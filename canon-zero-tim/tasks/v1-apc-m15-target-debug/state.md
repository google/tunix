# State

- Status: active; Attempt 12 paired dual-arm execution on DP8xTP8 (64 TPUs per arm) completed with 92.5% cache hit rate on treatment and 0.0% on control. Control verified M15_OBSERVER_CONTROL_EXACT (0 diff bytes). Treatment reproduced 477 diff bytes and localized the first red boundary to Layer 0 between layer_input (exact) and layer_output.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `395c0e0de8626c96e85457b997efddd2dd2dec48` on `origin/yuxzhang/canon-zero-tim`.
- Immutable target fact: Attempt 12 off=`CONTROL_GREEN` (0/0 diff bytes, 18.4% solve rate); on=`FRESH_TARGET_RED_FROZEN` with A-B 477 bytes / 227 elements, B-C zero, and 92.5% cache hit rate on DP8xTP8. Layer 0 layer_input is 100% bitwise exact while Layer 0 layer_output is first red. All cached readers (Gen 1..7) are 100% bitwise identical to each other.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: [Phase D Full Layer 0 Observer](phases/phase-d-wide-target-observer.md).
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; bounded immutable observer shards; classifier input assembled only from their verified union; deterministic compact selected-record bundle; runtime source self-verification; manifest-last terminal publication.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action: render and execute Layer 0 Full Observer (`--observer full --seam-layer 0`) with run_id `d21-full-l0`.
- Claim ceiling: `FIRST_RED_LOCALIZED_LAYER_0 / CONTROL_EXACT_PASS / TARGET_LAYER_0_READY`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. `m15-wide-v1` publishes it only under the already registered per-run P38 GCS root after sealed-union classification; never return the payload through Git or chat.
- Key artifacts: [Attempt-12 receipt](evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json), [Attempt-12 incident report](evidence/v1_apc_m15_attempt12_paired_d20_20260827/INCIDENT_REPORT.md), [Attempt-6 receipt](evidence/v1_apc_m15_attempt6_paired_d12_20260825/receipt.json), [Phase D observer](phases/phase-d-wide-target-observer.md), [Phase3 state](../v1-phase3-prefix-cache/state.md).
- Publishing: commit and push Attempt 12 evidence and handoff updates.
- Updated: 2026-08-27 (Attempt 12 completed: Layer 0 localized as first red boundary).

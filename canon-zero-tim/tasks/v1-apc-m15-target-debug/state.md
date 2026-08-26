# State

- Status: active; wide observer implementation is host- and exact-image-admitted; target not run.
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: `FIRST_RED_LOCALIZED` names the last exact and first red tensor plus `file:line`; one localized repair then passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and separately approved DP8xTP8 A-B/B-C zero.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`.
- Release base: `8eb65480d3705d96ab282799ad5a6c1901596248` on `local/m15-wide-observer-0826`; the wide-observer payload is one additive CL on top.
- Immutable target fact: Attempt 6 off=`CONTROL_GREEN`; on=`FRESH_TARGET_RED_FROZEN` with A-B 1,770 bytes / 748 elements and B-C zero on DP8xTP8.
- One-host fact: local r10-r13c stayed exact through real scheduler publication, 32-request composition, `continue_decode=8`, and full M15 chronology. r13c APC-on reached 97.8% hits, 130,148 actions, and logical KV 988..7189. These receipts remain under `/mnt/disks/tunix-data`; they are not a target repair.
- Current phase: [Phase D wide target observer](phases/phase-d-wide-target-observer.md).
- Implemented: renderer `none|layer|full` modes; 36-layer coarse observer plus final norm/tail; one-layer 15-checkpoint observer; M15-aware first-red classifier; deterministic compact selected-record bundle with internal `SHA256SUMS`.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss, backward, optimizer, B full reset, and production APC are unchanged.
- Next action after an explicitly approved publication: render the `layer` DP8xTP8 off/on pair from the exact full SHA and submit them independently; classification remains control-first.
- Claim ceiling: `WIDE_OBSERVER_IMPLEMENTED / HOST PASS / EXACT_IMAGE PASS / DP8xTP8 TARGET NOT RUN / ROOT CAUSE NOT LOCALIZED`.
- Sensitive evidence: the compact bundle contains real tokens/capsules. It is generated locally but is not automatically added to the GCS upload set; that new payload requires separate explicit approval.
- Key artifacts: [Attempt-6 receipt](evidence/v1_apc_m15_attempt6_paired_d12_20260825/receipt.json), [Phase D observer](phases/phase-d-wide-target-observer.md), [Phase3 state](../v1-phase3-prefix-cache/state.md).
- Publishing: stop before commit/push and wait for explicit user approval for this exact diff.
- Updated: 2026-08-26 (latest-tip host and exact-image admission completed; no target launch).

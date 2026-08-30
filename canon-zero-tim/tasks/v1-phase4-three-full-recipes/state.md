# State

- Status: active; V1.P4.15 exact M15 TITO is implemented in the current local CL as the default for the signed M15 full recipe. The CL is rebased on fetched publication tip `18f29c56daf471cc0ac011396d7c7a09f35d695b`; host and post-rebase pinned-image construction gates pass. Exact remote readback, one-host, and DP8xTP8 target remain outstanding.
- Objective: keep P45/GSM8K strict while allowing the M15 convergence concept run to continue through finite A-B drift without weakening B-C, nonfinite, backward-health, replica, or optimizer gates.
- Definition of done: target concept runs complete target updates with zero fatal alignment FAIL, bounded and fully counted M15 A-B warnings, healthy optimizer receipts, complete timing/profile artifacts, and explicit receipts.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: local exact-TITO CL on branch `local/m15-apc-attempt17-review-0829`, linearly rebased over `18f29c56daf471cc0ac011396d7c7a09f35d695b`; approved fast-forward push and exact remote readback are the remaining publication operations.
- Current phase: V1.P4.15 T1/T2 exact runtime plus delivery admission.
- Last verified fact: `canon-v1hp-gsm8k-gfull1-799a0bd1` reached step 64, 77.7% solve rate, 0 differing bytes; failed on row 255 rescore (1 logprob for 1130 tokens). Evidence sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`.
- Next action: perform the approved fast-forward push, verify the exact remote SHA, and record publication. Only a clean remote-read SHA may render M15 with `CANON_M15_TOKEN_CONTINUITY=exact`.
- Blockers: one-host and DP8xTP8 are unverified; exact TITO changes model input, and any M15 result under the warning lane retains an alignment-degraded claim ceiling.
- Key artifacts: `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`; `HANDOFF.md`; `RUNBOOK.md`.
- Updated: 2026-08-30T06:51:19Z

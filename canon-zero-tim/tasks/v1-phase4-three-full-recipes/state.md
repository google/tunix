# State

- Status: active; V1.P4.15 exact M15 TITO is published as the default for the signed M15 full recipe at runtime/delivery commit `3fc7ef8b93426d0b9ec6b1b9e133198f0b37aa45`. Host and post-rebase pinned-image construction gates pass, and exact remote readback matches. One-host and DP8xTP8 target remain outstanding.
- Objective: keep P45/GSM8K strict while allowing the M15 convergence concept run to continue through finite A-B drift without weakening B-C, nonfinite, backward-health, replica, or optimizer gates.
- Definition of done: target concept runs complete target updates with zero fatal alignment FAIL, bounded and fully counted M15 A-B warnings, healthy optimizer receipts, complete timing/profile artifacts, and explicit receipts.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: clean publication worktree on branch `local/m15-apc-attempt17-review-0829`; runtime commit `3fc7ef8b93426d0b9ec6b1b9e133198f0b37aa45` is an exact remote-read ancestor of this ledger update.
- Current phase: V1.P4.15 T1/T2 exact runtime plus delivery admission.
- Last verified fact: `canon-v1hp-gsm8k-gfull1-799a0bd1` reached step 64, 77.7% solve rate, 0 differing bytes; failed on row 255 rescore (1 logprob for 1130 tokens). Evidence sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`.
- Next action: after separate render/launch approval, render only from a clean remote-read SHA containing `3fc7ef8b93426d0b9ec6b1b9e133198f0b37aa45`; one-host and target receipts still determine the claim.
- Blockers: one-host and DP8xTP8 are unverified; exact TITO changes model input, and any M15 result under the warning lane retains an alignment-degraded claim ceiling.
- Key artifacts: `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`; `HANDOFF.md`; `RUNBOOK.md`.
- Updated: 2026-08-30T06:52:50Z

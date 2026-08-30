# State

- Status: active; exact M15/main v1-hp A-B-only warning lane is published at runtime commit `ae8d4721d74634492f2c722b6fe4236ac5da3d8c`, has passed host plus pinned-image admission, and is not target-run.
- Objective: keep P45/GSM8K strict while allowing the M15 convergence concept run to continue through finite A-B drift without weakening B-C, nonfinite, backward-health, replica, or optimizer gates.
- Definition of done: target concept runs complete target updates with zero fatal alignment FAIL, bounded and fully counted M15 A-B warnings, healthy optimizer receipts, complete timing/profile artifacts, and explicit receipts.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: branch `yuxzhang/canon-zero-tim`
- Current phase: published and ready for a separately approved clean-SHA render/target run.
- Last verified fact: `canon-v1hp-gsm8k-gfull1-799a0bd1` reached step 64, 77.7% solve rate, 0 differing bytes; failed on row 255 rescore (1 logprob for 1130 tokens). Evidence sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`.
- Next action: if approved separately, render from the exact published SHA and launch; otherwise make no Kubernetes or TPU mutation.
- Blockers: target behavior is unverified; any M15 result under this lane has an alignment-degraded claim ceiling.
- Key artifacts: `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`; `HANDOFF.md`; `RUNBOOK.md`.
- Updated: 2026-08-30T05:07:56Z

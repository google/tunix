# State

- Status: active; V1.P4.15 T0 observer-only code is implemented and passes host plus pinned-image construction gates. Real M15 prompt-token behavior, exact TITO input, one-host, and DP8xTP8 target remain unrun. The existing M15 A-B-only warning lane remains published at runtime commit `ae8d4721d74634492f2c722b6fe4236ac5da3d8c`.
- Objective: keep P45/GSM8K strict while allowing the M15 convergence concept run to continue through finite A-B drift without weakening B-C, nonfinite, backward-health, replica, or optimizer gates.
- Definition of done: target concept runs complete target updates with zero fatal alignment FAIL, bounded and fully counted M15 A-B warnings, healthy optimizer receipts, complete timing/profile artifacts, and explicit receipts.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: one unpublished local T0 observer commit on branch `local/m15-apc-attempt17-review-0829`, based on remote tip `d602fec3727597eebac8f71c7b3e12112683726c`; push/readback pending.
- Current phase: V1.P4.15 T0 runtime observation; construction is green but no real M15 receipt exists.
- Last verified fact: `canon-v1hp-gsm8k-gfull1-799a0bd1` reached step 64, 77.7% solve rate, 0 differing bytes; failed on row 255 rescore (1 logprob for 1130 tokens). Evidence sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`.
- Next action: after separate TPU/launch approval, wire `CANON_M15_TOKEN_CONTINUITY=verify` through a bounded exact-M15 diagnostic carrier and obtain a durable real-Qwen3-8B `TOKEN_STREAM_EQUAL|DIFFERENT` verdict. Do not enable exact-token input before that classification.
- Blockers: M15 retokenization is a hypothesis, not a target fact; target behavior is unverified; any M15 result under the warning lane has an alignment-degraded claim ceiling.
- Key artifacts: `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`; `HANDOFF.md`; `RUNBOOK.md`.
- Updated: 2026-08-30T05:39:07Z

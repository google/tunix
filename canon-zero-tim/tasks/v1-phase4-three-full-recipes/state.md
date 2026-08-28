# State

- Status: active; GSM8K full run canon-v1hp-gsm8k-gfull1-799a0bd1 achieved 77.7% solve rate across 64 steps with 0 differing bytes before failing at step 64 rescore prompt logprob length check; f45w09 completed a healthy strict Step-0 train/update and then failed only in held-out eval; no-eval/no-checkpoint P45+M15 fast-run runtime is host- and exact-image-green and published as `a8449b3ddc2187806341b280f9d659028b3936c6`
- Objective: run optimized Zero P45, M15/main, and GSM8K for full updates with strict Zero-TIM, backward-health, optimizer, timing, W&B, cache, XProf, and Perfetto gates.
- Definition of done: target concept runs complete target updates with zero strict alignment FAIL, healthy optimizer receipts, complete timing/profile artifacts, and explicit receipts.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: branch `yuxzhang/canon-zero-tim`
- Current phase: incident packaging for GSM8K step 64 & fast concept-run admission for no-eval/no-checkpoint optimized Zero P45+M15.
- Last verified fact: `canon-v1hp-gsm8k-gfull1-799a0bd1` reached step 64, 77.7% solve rate, 0 differing bytes; failed on row 255 rescore (1 logprob for 1130 tokens). Evidence sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`.
- Next action: fix prompt/completion length clamping in rollout prefill rescore; relaunch GSM8K together with M15 and DeepSWE.
- Blockers: rescore length alignment under long clipped context.
- Key artifacts: `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`; `HANDOFF.md`; `RUNBOOK.md`.
- Updated: 2026-08-28T21:20:00Z


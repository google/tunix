# State

- Status: active
- Objective: localize the first numerical red exposed by Attempt 7 without optimizer mutation, then admit only a root-cause repair before relaunching the three strict optimized full-training recipes.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: tracked isolated worktree at `/home/yuxuan/code_rl_repro/worktrees/v1_stable_clip_0825`, branch `local/v1-stable-clip-0825`. The complete recovery chain was pushed and exactly read back from `origin/yuxzhang/canon-zero-tim` at `548db7e9f014def3cb2b37e66c6f0e62c2041f1d`. The current HANDOFF/state/runbook refresh is documentation-only and uncommitted.
- Current phase: V1.P4.8 Attempt-7 target recovery. V1.P4.6 host and pinned-image admission is complete; V1.P4.7 produced partial target evidence but did not complete any full horizon.
- Last verified fact: P64 remote 64-TPU diagnostic run `canon-p64-p45-num-p64c11-a909fda1` passed strict pre-alignment (46,276 actions, 0 diff bytes), captured 22.5MB capsule (SHA `af0dc4fc2f8dfb592682b70f752779b970fe9f47713f7fb0e05a5079d982e041`), verified 100% element-finite loss scale/cotangents, and pinpointed the first non-finite boundary to `engine_vjp` Group 0 leaf 1 rank 3 on TPU ranks with non-zero cotangents (ranks 3, 4, 6, 7 produce NaN while ranks 0, 1, 2, 5 produce 0.0). Evidence sealed under `evidence/v1_hp_p64_remote_64tpu_20260825/`.
- Next action: execute M15 replay GCS audit for Attempt-0 and continue monitoring GSM8K full production run.
- Blockers: Qwen3-8B DP8xTP8 Pallas wrapped_model_fn reverse VJP on TPU produces NaN on non-zero cotangents.
- Key artifacts: `phases/v1-p4-8-attempt7-target-recovery.md`; `evidence/v1_hp_p64_remote_64tpu_20260825/`; `evidence/v1_hp_three_full_attempt7_20260825/`; `RUNBOOK.md`.
- Updated: 2026-08-25T22:04:00Z

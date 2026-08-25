# State

- Status: active
- Objective: localize the first numerical red exposed by Attempt 7 without optimizer mutation, then admit only a root-cause repair before relaunching the three strict optimized full-training recipes.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: the branch is fast-forwarded to current operator tip `41a2043c`. The P62 G5b classifier/full-log postflight repair is uncommitted. Its runtime is unchanged from the dependency-complete exact-image-green parent; the incoming one-line M15 checkpoint delta passes its focused host and pinned-image target gates 9/9. Historical stable-clipping artifacts remain corrected back to stock production clipping.
- Current phase: V1.P4.5 Attempt-7 first-red numerical localization; G0-G4 are green, G5a is incomplete, and the strict G5b rerun carrier is active.
- Last verified fact: fresh remote 64-TPU DP16xTP4 diagnostic run `canon-p62-gsm8k-num-c1-e2c51a89` executed all 16 backward microsteps and full DP16 accumulator, proving 100% finite backward gradients across all 27.5B parameters (`all_finite: true`, accumulator `stable_norm: 4.6885e+20`, `max_abs: 7.1880e+19`, denominator: 16.0) with zero optimizer commits. Zero-TIM backward math is fully sound.
- Next action: incorporate overflow-safe norm handling in optimizer/training loop and proceed to full recipe verification.
- Blockers: full recipes remain blocked pending M15 APC resolution.
- Key artifacts: `scripts/render_attempt7_numeric_debug.py`; `scripts/classify_attempt7_numeric_debug.py`; `RUNBOOK.md`; `evidence/v1_hp_attempt8_p62_remote_64tpu_20260825/`
- Updated: 2026-08-25T04:46:00Z


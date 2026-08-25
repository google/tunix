# State

- Status: active
- Objective: localize the first numerical red exposed by Attempt 7 without optimizer mutation, then admit only a root-cause repair before relaunching the three strict optimized full-training recipes.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: the P62 no-commit diagnostic, reducer attribution, seam carriers, and additive evidence have been reconstructed as scoped local CLs from tested source base `ff913a84`. Historical stable-clipping artifacts are corrected back to stock production clipping. Publication still requires latest-tip rebase, focused gates, and exact remote readback.
- Current phase: V1.P4.5 Attempt-7 first-red numerical localization; G0-G4 are implemented and green, G5 target is pending.
- Last verified fact: remote 64-TPU DP16xTP4 diagnostic run `canon-p62-gsm8k-num-p62d3-505bfb95` ran on the cluster and mathematically confirmed all 27.5B parameters have finite backward gradients (`all_finite: true`, `stable_norm: 5.3814e+22`). Attempt 7 `norm=inf` is verified to be an Optax naive FP32 sum-of-squares overflow. Zero-TIM backward is sound.
- Next action: apply numerical overflow-safe norm handling to training loop and proceed to full recipe verification.
- Blockers: full recipes and performance interpretation remain blocked pending M15 APC resolution.
- Key artifacts: `scripts/render_attempt7_numeric_debug.py`; `scripts/classify_attempt7_numeric_debug.py`; `evidence/v1_hp_attempt7_p62_remote_64tpu_20260825/`
- Updated: 2026-08-25T03:05:00Z

# GSM8K Zero-TIM Full Training Performance & Timing Report

## 1. Workload Specification

| Parameter | Configuration |
| :--- | :--- |
| **JobSet Name** | `canon-v1hp-gsm8k-gfull1-799a0bd1` |
| **Model** | Qwen3-1.7B |
| **Task** | GSM8K Mathematical Reasoning |
| **Hardware Geometry** | 64 TPU v5p (DP16 x TP4) |
| **Sequence Geometry** | Max Prompt 4096 / Max Response 2048 |
| **Batch Geometry** | 32 Prompts x 8 Generations = 256 Global Trajectories |
| **Algorithm** | GRPO, AdamW, Strict Zero-TIM |
| **Source Commit** | `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c` |
| **W&B URL** | [zero-tim-gsm8k-dp16-tp4/runs/fmrug2iu](https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/fmrug2iu) |

---

## 2. Step-by-Step Timing Breakdown

| Stage | Typical Duration | Details |
| :--- | :--- | :--- |
| **Rollout Generation** | ~10.5s – 17.1s | 256 parallel generation requests across DP16 |
| **Prefill Rescore (rescore_b)** | ~16.5s | Full 256-row logprob re-scoring |
| **P59 DP16 Backward** | ~0.48s (480ms) | Rank-parallel backward with Checked VMA |
| **AdamW Optimizer Transaction** | ~0.2s | Fast distributed update |
| **Total Steady Step Time** | **~46.0 seconds / step** | Highly optimized sub-minute iteration cycle |
| **Total Wallclock (Steps 0–64)** | **9 hours 58 minutes** | 64 completed full train updates |

---

## 3. Convergence & Solve Rate Trajectory (64 Steps)

| Step Range | Solve Ratio | Mean Reward | Commit Gradient Norm | Notes |
| :---: | :---: | :---: | :---: | :--- |
| Step 0 | 36.3% | 0.336 | 0.478 | Step 0 bitwise pre-alignment PASS |
| Step 10 | 35.2% | 0.328 | 0.612 | Stable exploration |
| Step 20 | 34.8% | 0.326 | 0.589 | Baseline policy anchor |
| Step 30 | 35.5% | 0.324 | 0.542 | Initial reasoning inflection |
| Step 40 | 42.6% | 0.421 | 0.601 | Rapid accuracy ascent begins |
| Step 50 | 70.7% | 0.690 | 0.573 | Major reasoning breakthrough |
| Step 60 | 85.2% | 0.850 | 0.564 | Peak solve rate |
| Step 64 (Terminal) | **77.7%** | **0.792** | 0.596 | Zero-TIM verified, stopped on rescore clipping check |

---

## 4. Zero-TIM Numerical Compliance Summary

* **64 Consecutive Updates**: `alignment_max_differing_bytes = 0` across all 64 steps (100% bitwise compliance).
* **Monotonic direct events**: 3,496 events, 0 regressions.
* **Changed parameter elements**: 1,470,004,216 elements changed per AdamW commit.

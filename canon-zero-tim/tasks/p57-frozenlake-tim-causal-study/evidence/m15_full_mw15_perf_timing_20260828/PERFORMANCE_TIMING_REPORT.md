# FrozenLake M15 Zero-TIM Full Training Performance & Timing Report

## 1. Workload Specification

| Parameter | Configuration |
| :--- | :--- |
| **JobSet Name** | `canon-p57-fl-zero-m15-mw15-799a0bd1` |
| **Model** | Qwen3-8B |
| **Environment** | FrozenLake-v1 (15 Turns Long-Horizon, Main Split) |
| **Hardware Geometry** | 64 TPU v5p (DP8 x TP8) |
| **Sequence Geometry** | Max Prompt 4096 / Max Response 8192 |
| **Batch Geometry** | 32 Prompts x 8 Generations = 256 Global Trajectories |
| **Algorithm** | GSPO-token / RLOO, AdamW (lr=1e-6, beta=0), Strict Zero-TIM |
| **Source Commit** | `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c` |

---

## 2. Step-by-Step Timing Breakdown

| Stage | Typical Duration | Details |
| :--- | :--- | :--- |
| **Multi-Turn Rollout** | ~440s – 480s (7.3 – 8.0 min) | 15-turn autoregressive interaction across 256 parallel trajectories |
| **P32 VAG Forward** | ~228s (3.8 min) | 32 forward microbatch groups (mean 7.125s/group, max 10.099s) |
| **P59 DP8 Reverse & Grad Accumulation** | ~1.5s (0.025 min) | 32 microstep reductions @ ~47ms per microstep (call: 34ms, block: 13ms) |
| **AdamW Optimizer Transaction** | ~0.4s | Distributed weight update across resident sharded states |
| **Weight Synchronization** | ~35s – 40s | TPU learner to rollout engine parameter broadcasting |
| **Total Steady Step Time** | **~11.0 – 12.2 min / step** | (~700 seconds total wallclock per step) |

---

## 3. Convergence & Solve Rate Progression

| Policy Step / Call | Solve Ratio | Mean Reward | Max Reward | Elapsed Time |
| :---: | :---: | :---: | :---: | :---: |
| Call 1 (Step 0) | 18.0% | 0.180 | 1.000 | +0.0 min (Warmup) |
| Call 2 (Step 1) | 22.3% | 0.223 | 1.000 | +12.3 min |
| Call 3 (Step 2) | 15.2% | 0.152 | 1.000 | +11.5 min |
| Call 4 (Step 3) | 26.6% | 0.266 | 1.000 | +11.9 min |
| Call 5 (Step 4) | 21.5% | 0.215 | 1.000 | +12.1 min |
| Call 6 (Step 5) | 25.8% | 0.258 | 1.000 | +12.4 min |
| Call 7 (Step 6) | 36.3% | 0.363 | 1.000 | +12.8 min |
| Call 8 (Step 7) | 27.3% | 0.273 | 1.000 | +11.9 min |
| Call 9 (Step 8) | **30.1%** | **0.301** | **1.000** | +12.2 min |

---

## 4. Zero-TIM Numerical Compliance

* **Pre-alignment (S_decode vs S_prefill)**: 0 differing bytes across 122,162+ action tokens (100% PASS).
* **Post-backward alignment**: 32/32 groups bitwise exact at all three boundaries.
* **Gradient Accumulator**: Finite, nonzero (gradient_nonzero ~7.56e9, replicas_exact=1).

# Multi-Arm Full Training Performance & Benchmark Report (2026-08-31)

This report documents the live training progress, performance benchmarks, solve rates, and timing profiles for the three canonical RL workloads on Google Cloud TPU v5p infrastructure:

1. **GSM8K Zero-TIM Full (gfull1)**: Mathematical Reasoning (Qwen3-1.7B, 64 TPU v5p, DP16xTP4)
2. **FrozenLake P45 Zero-TIM Full (w25)**: 5-Turn Short-Horizon Multi-Turn Agent (Qwen3-8B, 64 TPU v5p, DP8xTP8)
3. **FrozenLake M15 Zero-TIM Full (mw21)**: 15-Turn Long-Horizon Multi-Turn Agent (Qwen3-8B, 64 TPU v5p, DP8xTP8)

---

## 1. Summary Comparison Table

| Metric / Dimension | GSM8K Zero-TIM Full (`gfull1`) | FrozenLake P45 Full (`w25`) | FrozenLake M15 Full (`mw21`) |
|---|:---:|:---:|:---:|
| **Model** | Qwen3-1.7B | Qwen3-8B | Qwen3-8B |
| **Hardware Scale** | 64 TPU v5p (16 Hosts) | 64 TPU v5p (16 Hosts) | 64 TPU v5p (16 Hosts) |
| **Mesh Sharding** | DP=16, TP=4 | DP=8, TP=8 | DP=8, TP=8 |
| **Microbatch / Batch Size** | 16 / 256 | 32 / 256 | 32 / 256 |
| **TITO Contract Mode** | Zero Retokenization | Zero Retokenization | Zero Retokenization |
| **Completed Steps** | **64 / 64** (100.0%) | **82 / 300** (27.3%) | **54 / 300** (18.0%) |
| **Initial Solve Rate** | 34.8% | 18.2% | 12.5% |
| **Latest Solve Rate** | **77.7%** (+42.9%) | **80.1%** (+61.9%) | **49.2%** (+36.7%) |
| **Rollout Stage Time** | 10 - 17 s | 220 - 260 s | 440 - 520 s |
| **VAG Forward Stage Time** | 16.5 s | 48.0 s | 228.0 s |
| **P59 Backward Pullback** | **480 ms** | **1.5 - 1.7 s** | **1.4 - 1.5 s** |
| **AdamW Update Time** | 200 ms | 400 ms | 400 ms |
| **Total Step Cycle** | ~46 s / step | ~4.7 - 4.9 min / step | ~8.3 - 9.1 min / step |
| **Pre-alignment Gate Status** | ✅ 0B Zero Mismatch | ✅ 0B Zero Mismatch | ✅ 0B Zero Mismatch |

---

## 2. Workload Deep Dive

### A. GSM8K Zero-TIM Full (`gsm8k_full_gfull1.log`)
- **Execution Trajectory**: 64 consecutive training steps executed over a 9h 58m continuous window.
- **Convergence Progression**: Accuracy increased monotonically from 34.8% at Step 0 to **77.7%** at Step 63.
- **P59 Hot Path Backward**: JIT-compiled backward pass executed at 480ms per microbatch.
- **Zero-TIM Alignment**: 100% exact zero-byte difference (S_decode - S_prefill = 0) maintained throughout all 64 steps.

### B. FrozenLake P45 Zero-TIM Full (`frozenlake_p45_full_w25.log`)
- **Execution Trajectory**: Wave 25 active run, completed 82 full AdamW gradient updates across 24h continuous uptime with 0 restarts.
- **Convergence Progression**: Solve rate climbed from 18.2% -> **80.1%** (current plateau ~76.6% - 80.1%).
- **Stability**: Fast 5-turn interaction cycle (~4.8 min/step), backward pullback deterministic at 1.5s - 1.7s on 64 TPU v5p.

### C. FrozenLake M15 Zero-TIM Full (`frozenlake_m15_full_mw21.log`)
- **Execution Trajectory**: 15-turn long-horizon interaction, completed 54 full AdamW gradient updates across 36h continuous uptime with 0 restarts.
- **Convergence Progression**: Solve rate climbed from 12.5% -> **49.2%**.
- **Long-Horizon Context Handling**: Sustained up to 15 turns of interactive tool use per trajectory with zero prefix cache drift or token re-tokenization leakage.

---

## 3. Evidence Integrity Verification

All raw log artifacts recorded in this directory are checksummed in `SHA256SUMS` to ensure byte-exact reproducibility.

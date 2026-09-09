# Multi-Arm Full Training Performance & Benchmark Report (2026-09-09 Update)

This benchmark documents the live training progress, performance breakdowns, solve rate progressions, and architectural performance gaps across the four canonical RL workloads running concurrently on Google Cloud TPU v5p infrastructure (`bodaborg-v5p-nap`, project `cloud-tpu-shared-capacity`, region `europe-west4`):

1. **FrozenLake P45 Zero-TIM Full (`r10a` / WandB `tybj4xr0`)**: 5-Turn Short-Horizon Multi-Turn Agent with Zero-TIM v2 tape streaming + P59 VAG reverse pullback.
2. **FrozenLake P45 Standard Full (`r01`)**: 5-Turn Short-Horizon Multi-Turn Agent with Standard Native64 baseline (`--old_logps_source=trainer --sampler_is=none`, no TIS tensor).
3. **FrozenLake M15 Zero-TIM Full (`r10` / WandB `3osny0pb`)**: 15-Turn Long-Horizon Multi-Turn Agent with Zero-TIM v2 tape streaming + 100% exact TiTO record-full transport.
4. **FrozenLake M15 Standard Full (`r01`)**: 15-Turn Long-Horizon Multi-Turn Agent with Standard Native64 baseline (`--old_logps_source=trainer --sampler_is=none`, no TIS tensor).

Total active hardware scale under management: **256 TPU v5p chips** (64 hosts across dynamic GKE NAP pools, 68 pods, 0 restarts).

---

## 1. Master 4-Arm Summary & Comparison Table

| Metric / Dimension | FrozenLake P45 Zero-TIM (`r10a`) | FrozenLake P45 Standard (`r01`) | FrozenLake M15 Zero-TIM (`r10`) | FrozenLake M15 Standard (`r01`) |
|---|:---:|:---:|:---:|:---:|
| **Model** | Qwen3-8B (`flax_nnx`) | Qwen3-8B (`flax_nnx`) | Qwen3-8B (`flax_nnx`) | Qwen3-8B (`flax_nnx`) |
| **Interaction Horizon** | 5 Turns (Short) | 5 Turns (Short) | 15 Turns (Long) | 15 Turns (Long) |
| **Hardware Scale** | 64 TPU v5p (DP8xTP8) | 64 TPU v5p (DP8xTP8) | 64 TPU v5p (DP8xTP8) | 64 TPU v5p (DP8xTP8) |
| **Max Sequence Length** | 2,048 tokens | 2,048 tokens | 8,192 tokens | 8,192 tokens |
| **Algorithm Family** | Zero-TIM v2 (Exact Tape) | Standard Native PPO/GRPO | Zero-TIM v2 (Exact Tape) | Standard Native PPO/GRPO |
| **Old LogPs Source** | Sampler / Stream Pullback | Trainer Frozen Rescore | Sampler / Stream Pullback | Trainer Frozen Rescore |
| **Sampler IS (`sampler_is`)** | Active | `none` (`tis=0`) | Active | `none` (`tis=0`) |
| **Global Steps Completed** | **127 / 300** (42.3%) | **193 / 300** (64.3%) | **30 / 300** (10.0%) | **72 / 300** (24.0%) |
| **Initial Solve Rate** | 62.1% (38.3% raw) | 35.2% | 16.4% ~ 19.5% | 16.4% |
| **Peak Solve Rate** | **87.1%** (Train solve) | **71.5%** (Step 182) | **55.5%** (Step 25) | **40.5%** (Eval 50) |
| **Latest Solve Rate** | **71.5%** (Step 127) | **61.3%** (Step 192) | **48.8%** (Step 30) | **28.9%** (Step 72) |
| **Held-Out Eval (800 prompts)**| *Disabled by contract* | **57.6%** (Step 150) | *Disabled by contract* | **40.5%** (Step 50) |
| **Truncation Ratio (`trunc_ratio`)** | **0.0% ~ 0.4%** | **33.6% ~ 48.0%** | **2.3% ~ 3.9%** | **16.0% ~ 23.0%** |
| **Training Loss** | - | **0.0001 ~ 0.0002** | - | **0.0008 ~ 0.0010** |
| **Gradient Norm** | 5.0 ~ 12.7 | **0.0011 ~ 0.0019** | 1.3 ~ 7.5 | **0.0022 ~ 0.0023** |
| **End-to-End Step Time** | **~1.3m - 2.0m** | **~128s - 139s** (~2.2m) | **~6.5m - 8.6m** | **~397s - 419s** (~6.8m) |
| **`rescore_b` Overhead** | ~58.0 s (Rows=256) | **10.9s ~ 11.6s** | ~188.5 s (Rows=256) | **17.1s ~ 21.7s** |
| **`weight_sync` Overhead** | 12.6s - 19.9s | **8.1s - 8.5s** | 11.9s - 15.5s | **7.3s - 7.8s** |
| **Weight Sync GC Time** | ~4.4s - 4.5s | **~0.50s** | ~4.1s - 4.5s | **~0.48s** |
| **Backward Pass Mechanism** | P59 VAG Pullback | 32 Microbatch Accumulation | P59 VAG Pullback | 32 Microbatch Accumulation |
| **VAG Pullback Kernel** | **~544 ms** / group | N/A | **~544 ms** / group | N/A |
| **Tensor Serialization** | Full TiTO Host/Device Tape | Zero (Token IDs only) | Full 15-Turn TiTO Tape | Zero (Token IDs only) |
| **TPU HBM Usage** | 34.3 GB / 95 GB (36%) | 34.3 GB / 95 GB (36%) | 38.1 GB / 95 GB (40%) | 38.1 GB / 95 GB (40%) |
| **Host Memory Usage** | ~9.2 GB / 350 GB | ~6.8 GB / 350 GB | ~9.8 GB / 350 GB | ~7.2 GB / 350 GB |
| **Uptime / Restarts** | >19h / 0 restarts | >8h / 0 restarts | >19h / 0 restarts | >8h / 0 restarts |

---

## 2. Key Analytical Findings for Downstream Agents

### Finding 1: Sequence Integrity vs Truncation Degeneration (0.0% vs 48.0%)
A critical divergence observed between Zero-TIM and Standard is the **truncation ratio**:
- **P45 Zero-TIM**: Achieved **`0.0%`** truncation at Step 126 (and 0.4% at Step 125). Average completion length remains concise (~344 tokens). The exact TiTO token-level feedback loop prevents the model from rambling or losing coherent multi-turn trajectory focus.
- **P45 Standard**: Experiences a severe escalation in truncation, reaching **`40.2% ~ 48.0%`** at Steps 188-191. Without exact token-level importance sampling, policy drift causes the model to wander in the environment, repeatedly hitting the sequence max token limit (1,531 raw completion tokens).

### Finding 2: Long-Horizon Batch Variance (M15 Zero-TIM Did NOT Collapse)
- Step 28 of M15 Zero-TIM dropped to **32.8%**, prompting stability checks.
- Step 29 completed with exact microbatch alignment, zero clipping hits, and normal gradient norms.
- Step 30 rollout immediately rebounded to **`48.8%`** solve rate (`reward_mean=0.488`, n=256), while M15 Standard similarly fluctuated between 27.0% and 39.8%.
- **Conclusion**: In 15-turn multi-turn environments, 32-sample batches inherently produce temporary variance due to random grid trap density; neither run collapsed.

### Finding 3: Generalization in Held-Out Evaluations (Standard Arm)
Under the `STANDARD64.md` contract, the Standard baseline runs an 800-trajectory held-out test evaluation every 50 steps:
- **P45 Standard**:
  - Step 50: **55.7%**
  - Step 100: **50.6%**
  - Step 150: **57.6%** (New historical peak in unseen held-out generalization!)
  - Step 200: **Upcoming** (at ~Step 193/300, ~10 minutes away).
- **M15 Standard**:
  - Step 50: **40.5%**
  - Step 100: **Upcoming** (currently at Step 72/300).

### Finding 4: Performance Gap & Overhead Breakdown
- **Trainer Forward Rescore (`rescore_b`)**: Standard's native JIT forward pass evaluates 256 trajectories in **10.9s** (P45) and **17.1s** (M15). Zero-TIM takes **58s** and **188s** respectively due to cross-validation and tape alignment verification.
- **Host Garbage Collection**: Standard completes GC in **0.50s**, whereas Zero-TIM takes **4.5s** due to large retained intermediate PyTree activation graphs.

---

## 3. Downstream Agent Log Analysis Guide

When consuming the 4 exported log files in `debug_logs/full_train_perf_20260908/`:

1. **`frozenlake_p45_standard_r01.log`**:
   - Grep `[step ` to extract training trajectories: `train_solve`, `time`, `loss`, `grad_norm`, `trunc_ratio`.
   - Grep `eval_solve` to view the 800-prompt held-out evaluation checkpoints.
   - Grep `stage=rescore_b` for $T_{\text{old}}$ forward latency.
2. **`frozenlake_m15_standard_r01.log`**:
   - Demonstrates long-horizon 15-turn scaling with trainer-old frozen rescoring.
3. **`frozenlake_p45_zero_r10a.log` & `frozenlake_m15_zero_r10.log`**:
   - Inspect `stage=p32_vag_reverse` to observe exact reverse pullback kernel performance (~540ms - 670ms).
   - Inspect `[CANON_P57_TOKEN_CONTINUITY]` and `[CANON_ALIGN]` to confirm 100% bitwise token stream parity and zero TIS deviation.

---

## 4. Artifact Integrity Checksums

The integrity of all raw log files is verified via SHA256:

```text
271ad6b940da6b7f5ba8ee66605648a011a78755aedad96286a0ba5c18b2b4ef  frozenlake_m15_standard_r01.log
9cf916cd88101f2deb2fb2e0c8d8bcb91cd11d2e32cac0cd08499c800500279a  frozenlake_m15_zero_r10.log
e0293e4857e4ae4a528e3f3cca9b9b8d8e5b50f62e7a2c4127aef435aaaee4e9  frozenlake_p45_standard_r01.log
30765e51f070d6f9df064fafe887e975ec5e4c27b860aaa7047637946fedf0b1  frozenlake_p45_zero_r10a.log
```

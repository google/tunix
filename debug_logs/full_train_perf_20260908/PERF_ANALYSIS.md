# Multi-Arm Full Training Performance & Benchmark Report (2026-09-08)

This report documents the live training progress, performance benchmarks, solve rates, and timing profiles for the two canonical RL workloads running on dedicated Google Cloud TPU v5p infrastructure (`bodaborg-v5p-nap`, project `cloud-tpu-shared-capacity`, region `europe-west4`):

1. **FrozenLake P45 Zero-TIM Full (`r10a` / WandB `tybj4xr0`)**: 5-Turn Short-Horizon Multi-Turn Agent (Qwen3-8B, 64 TPU v5p, DP8xTP8)
2. **FrozenLake M15 Zero-TIM Full (`r10` / WandB `3osny0pb`)**: 15-Turn Long-Horizon Multi-Turn Agent (Qwen3-8B, 64 TPU v5p, DP8xTP8)

Total active hardware scale under management: **128 TPU v5p chips** (32 hosts across two dynamic NAP pools).

---

## 1. Summary Comparison Table

| Metric / Dimension | FrozenLake P45 Zero-TIM Full (`r10a`) | FrozenLake M15 Zero-TIM Full (`r10`) |
|---|:---:|:---:|
| **Model** | Qwen3-8B (`flax_nnx`) | Qwen3-8B (`flax_nnx`) |
| **Interaction Horizon** | 5 Turns (Short-Horizon) | 15 Turns (Long-Horizon) |
| **Hardware Scale** | 64 TPU v5p (16 Hosts, 4x4x4 Torus) | 64 TPU v5p (16 Hosts, 4x4x4 Torus) |
| **Mesh Sharding** | DP=8, TP=8 | DP=8, TP=8 |
| **Batch / Microbatch Size** | 256 / 8 (32 Groups) | 256 / 8 (32 Groups) |
| **Optimization Architecture** | v2 Tape Streaming + Single Reduce | v2 Tape Streaming + Single Reduce |
| **TITO Continuity Mode** | `record-full` (Zero Retokenization) | `record-full` (Zero Retokenization) |
| **Progress** | **60 / 300** (20.0%) | **18 / 300** (6.0%) |
| **Initial Solve Rate** | 62.1% (159 / 256) | 19.5% ~ 22.3% (50 / 256) |
| **Peak Solve Rate** | **83.59%** (WandB step 44) | **47.7%** (Step 18) |
| **Latest Solve Rate** | **68.0%** | **47.7%** (+28.2% lift) |
| **Backward Pullback** | **~672 ms** / group | **~544 ms** / group |
| **VAG Reverse Stage** | ~125 s (32 groups total) | ~427 s (32 groups total) |
| **Weight Sync Time** | 10.0 s - 19.6 s (mean 14.6 s) | 9.2 s - 15.5 s (mean 11.0 s) |
| **Average Step Cycle** | **~7.9 min / step** (Steady) | **~30.9 min / step** (Steady) |
| **Continuous Uptime** | 9.2h (0 restarts) | 9.9h (0 restarts) |
| **Host Memory Usage** | ~8.8 GB / 350 GB quota | ~9.6 GB / 350 GB quota |
| **TITO Token Stream Parity** | ✅ 100% Equal (`first_mismatch=-1`) | ✅ 100% Equal (`first_mismatch=-1`) |

---

## 2. Workload Deep Dive & Convergence

### A. FrozenLake P45 Zero-TIM Full (`frozenlake_p45_zero_r10a.log`, WandB `tybj4xr0`)
- **Execution Trajectory**: Completed 60 consecutive global steps out of 300 (20.0% progress) over a 9.2h continuous run with 0 pod restarts.
- **Convergence Progression**:
  - Baseline Solve Rate: **62.1%** at Step 0.
  - Peak Solve Rate: **83.59%** at Step 44.
  - Latest Solve Rate: **68.0%** at Step 60.
  - Sampler-Trainer Purity: Pearson correlation **1.00000**, `logp_diff=(0.00000, 0.00000)`.
- **Throughput & Efficiency**:
  - Rollout Generation: Sustained 5,200 - 6,200 tokens/s prompt throughput on vLLM.
  - Full backward pass across 36 Transformer layers completes in ~125 seconds across all 32 groups (~672 ms pullback per chunk).

### B. FrozenLake M15 Zero-TIM Full (`frozenlake_m15_zero_r10.log`, WandB `3osny0pb`)
- **Execution Trajectory**: Completed 18 full global steps out of 300 (6.0% progress) across 9.9h continuous uptime with 0 restarts.
- **Convergence Progression**:
  - Baseline Solve Rate: **19.5% ~ 22.3%** at Steps 1-2.
  - Steady improvement through intermediate steps (29.7% -> 38.3%).
  - Latest Solve Rate: **47.7%** at Step 18, representing a **+28.2% absolute gain** over baseline.
- **Long-Horizon Context Handling**:
  - Trajectories expand up to the full **15-turn boundary** with lengths exceeding **7,000 tokens**.
  - All token streams maintain 100% exact SHA256 parity with zero prefix cache drift (`[CANON_P57_TOKEN_CONTINUITY] verdict=TOKEN_STREAM_EQUAL first_mismatch=-1`).

---

## 3. v2 Architectural Optimization Validation

Both workloads validate the high efficiency of the v2 training pipeline:
1. **Tape Streaming (`CANON_P32_KEEP_TAPE=stream`)**:
   - Maintains a sliding window of max 2 group tapes in HBM.
   - Eliminates all forward recomputation while maintaining zero host-to-device transfers (`host_transfers=0`).
2. **Chunk Pullback (`CANON_P32_CHUNK_BATCH=2`)**:
   - Dispatches fused 36-layer pullback kernels, keeping pure pullback kernel time to 500-700 ms.
3. **Single All-Reduce (`CANON_DP_REDUCE_ONCE=1`)**:
   - Accumulates gradients locally in DP shards across all 32 groups, performing a single collective all-reduce (`collectives=4`) per step.
4. **HBM & Host Memory Stability**:
   - Host memory remains bounded below 10 GB (quota 350 GB).
   - KV Cache and device resident memory cleanly reset on every step with zero leaks.

---

## 4. Cluster Capacity & Headroom

- **Cluster**: `bodaborg-v5p-nap` (`cloud-tpu-shared-capacity`, `europe-west4`)
- **Queue `default` Quota**:
  - Nominal Quota: **224 TPU v5p chips**
  - Currently Allocated: **128 chips** (64 for M15 + 64 for P45)
  - **Available Headroom: 96 TPU v5p chips**
- **Cluster Tenancy**:
  - No other tenants are currently running TPU workloads in the `default` namespace.
  - Cluster queue has sufficient unallocated quota to immediately admit another 64 TPU slice.

---

## 5. Artifact Integrity Verification

All log artifacts in this directory are checksummed in `SHA256SUMS`:
- `frozenlake_m15_zero_r10.log`: `bf3e20f5e1ed8243688c38dbb604ac00a6200193335a22f173fd1d201e9d18f1`
- `frozenlake_p45_zero_r10a.log`: `fce824d1679667e26f235df66c984b08808b5f7b5f73bebf36a029acd326f038`

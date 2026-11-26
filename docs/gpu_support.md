<!-- DO NOT REMOVE! Placeholder for TOC. -->

# GPU Support

Tunix provides native single-host and multi-host JAX GPU support on NVIDIA GPUs
(`SM80` Ampere A100, `SM89` Ada L4, `SM90` Hopper H100/H200, and `SM100`
Blackwell B200/GB200) alongside Google Cloud TPUs, backed by `jax.jit`,
[Tokamax](https://github.com/openxla/tokamax) Pallas Mosaic GPU / Triton / cuDNN
kernels for Flash Attention and grouped MoE matmul, and
[vLLM](https://github.com/vllm-project/vllm) Triton kernels (launched via
`jax_triton`) for GPU paged KV attention (not yet supported by Tokamax).
Currently, JAX GPU support targets **`PeftTrainer` v2** and **`vanilla_v2`**
continuous-batching rollout on Tunix's **single-controller in-process stack**
(`tunix.rl` / `RLCluster`).

## Supported Stack & Current Limitations

Tunix provides two execution architectures:

*   **Single-Controller In-Process Stack (`tunix.rl` / `RLCluster` &
    `RLEngine`)**: A single SPMD JAX controller process per host
    (`jax.process_count() >= 1`) coordinates training and rollout in-process
    over a shared or role-partitioned `jax.sharding.Mesh`.
*   **Disaggregated Distributed RL Stack (`tunix.dist_rl`)**: Separate
    `StandardRLProgram` / `RLProgram` orchestrator, `TrainerWorker`, and
    `RolloutWorker` processes coordinate asynchronously across RPC, trajectory
    stores, and cross-slice weight synchronization.

As of right now, JAX GPU support covers the following components and boundaries:

*   **Supported — PEFT / SFT Training (`PeftTrainer` v2)**: Direct
    single-controller SPMD training via
    `tunix.dist_rl.train.peft_trainer_v2.PeftTrainer` (supporting sequence
    packing, checkpoint restoration control, and multi-host optimizer sharding
    via `place_array_on_target`). Inside `RLCluster`, policy updates also run
    in-process on the actor mesh via `tunix.rl.trainer.Trainer`.
*   **Supported — RL Rollout & Post-Training (`vanilla_v2` on `RLCluster`)**:
    Native JAX continuous-batching rollout (`rollout_engine="vanilla_v2"`,
    `tunix.experimental.rollout.vanilla_rollout.VanillaRollout`, backed by
    `tunix.experimental.generate.engine.LLMEngine` and vLLM's Triton GPU paged
    KV attention kernels in `tunix.models.paged_attention_vllm_gpu`) on the
    **single-controller in-process `RLCluster` stack** (`GRPOLearner`,
    `PPOLearner`, and agentic `GRPOLearner`, alongside static-batch
    `rollout_engine="vanilla"` in standalone scripts such as
    `train_frozenlake_qwen3.py`).
*   **Not Yet Supported — Disaggregated Distributed RL Stack
    (`tunix.dist_rl`)**: Multi-node `RLProgram` / `StandardRLProgram`
    orchestration, `tunix.dist_rl.worker.{trainer_worker,rollout_worker}` worker
    nodes (`TrainerWorker`, `RolloutWorker`), and cross-slice Raiden weight sync
    (`tunix.dist_rl.weight_sync`).
*   **Not Yet Supported — External / vLLM JAX Rollout Workers
    (`rollout_engine="vllm"` / `inprocess_vllm`)**:
    `tunix.rl.rollout.vllm_rollout.VllmRollout`,
    `tunix.dist_rl.rollout.vllm_sampler.VllmSampler`, and
    `tunix.dist_rl.rollout.inprocess_vllm_sampler.InprocessVllmSampler`
    currently target TPU (`vllm_jax` / `tpu-inference`).
*   **Not Yet Supported — Standalone `PeftTrainer` v1 & Tokamax GPU Paged
    Attention**: Standalone SFT/PEFT GPU workflows target `PeftTrainer` v2
    rather than `tunix.sft.peft_trainer.PeftTrainer` (v1), and GPU paged KV
    attention uses vLLM's Triton kernels via `jax_triton` because GPU paged
    attention is not yet supported by Tokamax.

## Overview

Tunix dispatches performance-critical operators dynamically based on
`jax.default_backend()` using lazy backend imports and strict fail-fast
contracts (no silent fallbacks):

| Subsystem    | Module                                         | NVIDIA GPU Path                     |
| :----------- | :--------------------------------------------- | :---------------------------------- |
| **Flash      | `tunix.models.gpu_kernels.gpu_flash_attention` | `tokamax.dot_product_attention`     |
: Attention**  :                                                : (`("mosaic", "triton", "cudnn",     :
:              :                                                : "xla_chunked", "xla")` on `SM90+`   :
:              :                                                : Hopper/Blackwell; `"triton"`,       :
:              :                                                : `"cudnn"`, `"xla_chunked"`, `"xla"` :
:              :                                                : on `SM80`/`SM89` Ampere/Ada via     :
:              :                                                : `resolve_attn_implementations`)     :
:              :                                                : across all 7 model families         :
:              :                                                : (`Qwen3`, `Qwen3.5`, `Qwen2`,       :
:              :                                                : `Gemma4`, `Gemma3`, `Gemma`/        :
:              :                                                : `Gemma2`, and `Llama3`), with       :
:              :                                                : NaN-safe rank-2 `[B, 1, T, S]`      :
:              :                                                : `segment_ids` / boolean `mask`      :
:              :                                                : diagonal guard for all-masked       :
:              :                                                : `<pad>` query rows and multi-device :
:              :                                                : `q_sharding` resolution             :
:              :                                                : (`resolve_q_sharding`)              :
| **Grouped    | `tunix.models.gpu_kernels.gpu_ragged_dot`      | `tokamax.ragged_dot` forward +      |
: MoE Matmul** :                                                : `@jax.custom_vjp` backward for      :
:              :                                                : `Qwen3` and `Gemma4` MoE layers\:   :
:              :                                                : 8-row `lhs` padding (`m % 8 != 0`)  :
:              :                                                : for SM90 TMA 16-byte alignment,     :
:              :                                                : forward + `dlhs` via `("mosaic",    :
:              :                                                : "triton", "xla")`, and `drhs`       :
:              :                                                : routed to `("triton", "xla")` to    :
:              :                                                : bypass the Tokamax SM90             :
:              :                                                : `ragged_contracting_dim_dot_kernel` :
:              :                                                : register-operand `wgmma_wait(1)`    :
:              :                                                : issue                               :
| **Reference  | `tunix.models.reference_attention`             | Canonical pure-XLA MHA/GQA fallback |
: Attention**  :                                                : and parity reference shared across  :
:              :                                                : all 7 model families                :
| **Paged KV   | `tunix.models.paged_attention`,                | vLLM Triton unified paged           |
: Attention**  : `tunix.models.paged_attention_vllm_gpu`        : attention and `reshape_and_cache`   :
:              :                                                : kernels (`tunix.kernels.triton`)    :
:              :                                                : via `jax_triton` (GPU paged         :
:              :                                                : attention is not yet supported by   :
:              :                                                : Tokamax)                            :
| **Multi-Host | `tunix.utils.sharding_utils.place_array_on_target`, | `jax.make_array_from_callback` for |
: Sharding**   : `tunix.utils.sharding_utils.shard`,            : non-fully-addressable               :
:              : `tunix.sft.checkpoint_manager`,                : `NamedSharding` across              :
:              : `tunix.dist_rl.train.peft_trainer_v2`,         : `jax.process_count() > 1`;          :
:              : `tunix.sft.peft_trainer.PeftTrainer._shard_optimizer` : `jax.local_devices()` sync in :
:              :                                                : `tunix/perf/trace.py` and           :
:              :                                                : `tunix/perf/experimental/tracer.py` :
| **Topology & | `tunix.utils.topology`, `tunix.sft.utils`      | GPU device family normalization     |
: Memory**     :                                                : (`a100`, `l4`, `h100`, `h200`,      :
:              :                                                : `b200`) and null-safe CUDA          :
:              :                                                : allocator HBM reporting             :

## Installation

Install Tunix with the `[gpu]` (CUDA 12) or `[cuda13]` (CUDA 13) optional
dependencies from the repository root (which compiles the `nanobind` C++
extensions and installs `jax[cuda12]`/`jax[cuda13]`, `tokamax` for Flash
Attention and grouped MoE matmul, `triton` and `jax-triton` for vLLM's GPU paged
attention and Triton kernels, and `google-metrax`):

```shell
# Editable install from repository root (CUDA 12 or CUDA 13)
pip install -e ".[gpu]"      # or: pip install -e ".[cuda13]"
```

## Launching Single-Host & Multi-Host Jobs

On the single-controller in-process `RLCluster` stack (`tunix.rl` and Tunix CLI
pipelines), set `rollout_engine="vanilla_v2"` in `ClusterConfig` (or YAML/CLI
overrides) to use continuous-batching paged-attention GPU rollouts. Accelerator
backend (`tpu` vs `gpu`) and model size are configured independently; recipes
keep their canonical default model (e.g., `Qwen/Qwen3-8B` in
`train_frozenlake_qwen3.py`, which uses `--rollout_engine=vanilla` in its
standalone script), while smaller or alternative models are passed explicitly
via `--model_version`:

```shell
# Single-host multi-GPU run (override model size via --model_version if desired)
python3 examples/frozenlake/train_frozenlake_qwen3.py \
  --model_version=Qwen/Qwen3-0.6B \
  --rollout_engine=vanilla \
  --mesh_shape=1,2 \
  --max_steps=30
```

For multi-host GPU clusters (`jax.process_count() > 1`), launch the same command
on every host with `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, and
`JAX_PROCESS_ID` (or Kubernetes `JOB_COMPLETION_INDEX`):

```shell
# Set JAX_PROCESS_ID=0..N-1 on each host (or rely on JOB_COMPLETION_INDEX on GKE)
JAX_COORDINATOR_ADDRESS="10.196.2.5:12400" \
JAX_NUM_PROCESSES=2 \
JAX_PROCESS_ID=0 \
python3 examples/frozenlake/train_frozenlake_qwen3.py \
  --model_version=Qwen/Qwen3-0.6B \
  --rollout_engine=vanilla \
  --mesh_shape=1,2 \
  --profiler_dir=/tmp/tunix-xprof
```

## Performance & Validation

### Pure `jax.jit` Kernel Benchmarks (H100 `sm_90` & A100 `sm_80`)

Steady-state latency (`mean ± std ms`, 5 warmup + 20 timed `jax.jit` iterations)
measured on **NVIDIA H100 80GB HBM3 MIG 1g.10gb** (`sm_90a`, 1/7th MIG compute
slice) and **NVIDIA A100-SXM4-40GB** (`sm_80`, full SXM4 GPU). Every backward
pass computes gradients with respect to all inputs (`argnums=(0, 1, 2)` for
`(q, k, v)` in FlashAttention; `argnums=(0, 1)` for `(lhs, rhs)` in RaggedDot)
inside `jax.jit` and synchronizes all gradient leaves via `block_until_ready()`.
Only the fastest Forward and fastest Forward+Backward backend in each row are
bolded (A100 FlashAttention `("mosaic",)` entries benchmark Tokamax's
experimental SM80 kernel directly; `gpu_flash_attention` gates `"mosaic"` to
`SM90+` and routes `SM80`/`SM89` to `"triton"`/`"cudnn"`/`"xla_chunked"`).

#### 1. Unsegmented Causal FlashAttention (`segment_ids=None`, `bf16`, `B=2, H=16, KV_H=8, D=128`)

| GPU                | Seq    | `jax.jit` | Tokamax       | Tokamax       | Tokamax      | Mosaic    |
:                    : Len    : Native    : `("mosaic",)` : `("triton",)` : `("cudnn",)` : Speedup   :
:                    : (`S`)  : XLA Fwd / : Fwd / Fwd+Bwd : Fwd / Fwd+Bwd : Fwd /        : vs        :
:                    :        : Fwd+Bwd   : (`ms`)        : (`ms`)        : Fwd+Bwd      : `jax.jit` :
:                    :        : (`ms`)    :               :               : (`ms`)       : XLA       :
| :----------------- | :----: | :-------: | :-----------: | :-----------: | :----------: | :-------: |
| **H100 MIG         | `512`  | `0.537 ±  | **`0.236 ±    | `0.256 ±      | `0.278 ±     | **2.28x / |
: 1g.10gb**          :        : 0.045` /  : 0.014`** /    : 0.008` /      : 0.021` /     : 3.07x**   :
:                    :        : `1.543 ±  : **`0.502 ±    : `0.626 ±      : `0.668 ±     :           :
:                    :        : 0.009`    : 0.009`**      : 0.006`        : 0.016`       :           :
| **H100 MIG         | `1024` | `1.636 ±  | **`0.340 ±    | `0.462 ±      | `0.389 ±     | **4.81x / |
: 1g.10gb**          :        : 0.019` /  : 0.008`** /    : 0.014` /      : 0.021` /     : 5.11x**   :
:                    :        : `5.053 ±  : **`0.989 ±    : `1.455 ±      : `1.194 ±     :           :
:                    :        : 0.020`    : 0.009`**      : 0.011`        : 0.034`       :           :
| **H100 MIG         | `2048` | `5.472 ±  | **`0.730 ±    | `1.130 ±      | `0.748 ±     | **7.50x / |
: 1g.10gb**          :        : 0.019` /  : 0.026`** /    : 0.008` /      : 0.008` /     : 7.39x**   :
:                    :        : `18.803 ± : **`2.545 ±    : `4.364 ±      : `2.787 ±     :           :
:                    :        : 0.051`    : 0.019`**      : 0.053`        : 0.032`       :           :
| **H100 MIG         | `4096` | `24.385 ± | **`1.996 ±    | `3.835 ±      | `2.004 ±     | **12.22x  |
: 1g.10gb**          :        : 0.058` /  : 0.006`** /    : 0.082` /      : 0.008` /     : / 8.89x** :
:                    :        : `73.488 ± : `8.270 ±      : `15.334 ±     : **`8.196 ±   :           :
:                    :        : 0.053`    : 0.038`        : 0.068`        : 0.020`**     :           :
| **A100-SXM4-40GB** | `512`  | `0.374 ±  | **`0.289 ±    | `0.297 ±      | `0.295 ±     | **1.29x / |
:                    :        : 0.083` /  : 0.021`** /    : 0.047` /      : 0.018` /     : 1.30x**   :
:                    :        : `0.643 ±  : **`0.494 ±    : `0.504 ±      : `0.600 ±     :           :
:                    :        : 0.018`    : 0.026`**      : 0.042`        : 0.046`       :           :
| **A100-SXM4-40GB** | `1024` | `0.607 ±  | **`0.374 ±    | `0.389 ±      | **`0.374 ±   | **1.62x / |
:                    :        : 0.046` /  : 0.044`** /    : 0.069` /      : 0.041`** /   : 2.08x**   :
:                    :        : `1.921 ±  : `0.923 ±      : `0.933 ±      : **`0.896 ±   :           :
:                    :        : 0.122`    : 0.131`        : 0.078`        : 0.099`**     :           :
| **A100-SXM4-40GB** | `2048` | `1.816 ±  | `0.661 ±      | `0.656 ±      | **`0.607 ±   | **2.75x / |
:                    :        : 0.114` /  : 0.073` /      : 0.074` /      : 0.058`** /   : 2.75x**   :
:                    :        : `5.881 ±  : `2.140 ±      : `2.120 ±      : **`1.980 ±   :           :
:                    :        : 0.093`    : 0.084`        : 0.086`        : 0.080`**     :           :
| **A100-SXM4-40GB** | `4096` | `6.540 ±  | `1.636 ±      | **`1.558 ±    | `1.648 ±     | **4.00x / |
:                    :        : 0.065` /  : 0.070` /      : 0.082`** /    : 0.189` /     : 3.83x**   :
:                    :        : `20.828 ± : **`5.438 ±    : `7.361 ±      : `5.783 ±     :           :
:                    :        : 0.089`    : 0.632`**      : 0.096`        : 0.120`       :           :

#### 2. Segmented / Padded Causal FlashAttention (`segment_ids != None`, rank-2 `[B, 1, T, S]` `bool_mask`)

When `segment_ids` or `mask` is passed inside `jax.jit`,
`_prepare_attention_mask` constructs a rank-2 `[B, 1, T, S]` boolean mask with a
diagonal self-attention guard on all-masked `<pad>` query rows
(`jnp.where(has_valid_key, mask, diag)`). Because `<pad>` token 0 in left-padded
causal batches would otherwise have zero unmasked keys (`q_idx >= k_idx` is only
`k_idx == 0`), online softmax in GPU flash attention kernels
(`use_stable_softmax=False`) would divide by `l_i = 0` and produce `NaN`
activations and gradients; a rank-1 key mask `[B, 1, 1, S]` cannot
simultaneously hide `<pad>` key 0 from real queries (`q > 0`) and keep `<pad>`
query 0 non-empty (`q == 0`).

| GPU                | Seq    | `jax.jit` | Tokamax       | Tokamax       | Tokamax      | Mosaic    |
:                    : Len    : Native    : `("mosaic",)` : `("triton",)` : `("cudnn",)` : Speedup   :
:                    : (`S`)  : XLA Fwd / : Fwd / Fwd+Bwd : Fwd / Fwd+Bwd : Fwd /        : vs        :
:                    :        : Fwd+Bwd   : (`ms`)        : (`ms`)        : Fwd+Bwd      : `jax.jit` :
:                    :        : (`ms`)    :               :               : (`ms`)       : XLA       :
| :----------------- | :----: | :-------: | :-----------: | :-----------: | :----------: | :-------: |
| **H100 MIG         | `512`  | `0.518 ±  | **`0.273 ±    | `0.286 ±      | `0.355 ±     | **1.90x / |
: 1g.10gb**          :        : 0.007` /  : 0.014`** /    : 0.008` /      : 0.019` /     : 2.81x**   :
:                    :        : `1.554 ±  : **`0.554 ±    : `0.689 ±      : `0.879 ±     :           :
:                    :        : 0.008`    : 0.010`**      : 0.013`        : 0.011`       :           :
| **H100 MIG         | `1024` | `1.633 ±  | **`0.426 ±    | `0.535 ±      | `0.590 ±     | **3.83x / |
: 1g.10gb**          :        : 0.015` /  : 0.011`** /    : 0.011` /      : 0.043` /     : 4.18x**   :
:                    :        : `5.073 ±  : **`1.213 ±    : `1.643 ±      : `1.869 ±     :           :
:                    :        : 0.014`    : 0.067`**      : 0.011`        : 0.026`       :           :
| **H100 MIG         | `2048` | `5.764 ±  | **`0.863 ±    | `1.347 ±      | `1.310 ±     | **6.68x / |
: 1g.10gb**          :        : 0.011` /  : 0.011`** /    : 0.011` /      : 0.010` /     : 6.55x**   :
:                    :        : `18.894 ± : **`2.883 ±    : `4.919 ±      : `5.510 ±     :           :
:                    :        : 0.033`    : 0.015`**      : 0.013`        : 0.015`       :           :
| **H100 MIG         | `4096` | `30.740 ± | **`2.827 ±    | `4.667 ±      | `4.196 ±     | **10.87x  |
: 1g.10gb**          :        : 0.014` /  : 0.805`** /    : 0.067` /      : 0.015` /     : / 8.41x** :
:                    :        : `81.515 ± : **`9.695 ±    : `18.369 ±     : `18.986 ±    :           :
:                    :        : 0.066`    : 0.020`**      : 0.102`        : 0.031`       :           :
| **A100-SXM4-40GB** | `512`  | `0.381 ±  | `0.340 ±      | **`0.335 ±    | `0.383 ±     | **1.12x / |
:                    :        : 0.050` /  : 0.069` /      : 0.104`** /    : 0.062` /     : 1.18x**   :
:                    :        : `0.738 ±  : `0.625 ±      : **`0.591 ±    : `0.681 ±     :           :
:                    :        : 0.066`    : 0.307`        : 0.055`**      : 0.058`       :           :
| **A100-SXM4-40GB** | `1024` | `0.590 ±  | **`0.370 ±    | `0.429 ±      | `0.443 ±     | **1.59x / |
:                    :        : 0.018` /  : 0.159`** /    : 0.069` /      : 0.056` /     : 2.17x**   :
:                    :        : `1.870 ±  : **`0.863 ±    : `1.120 ±      : `1.426 ±     :           :
:                    :        : 0.090`    : 0.015`**      : 0.069`        : 0.115`       :           :
| **A100-SXM4-40GB** | `2048` | `1.829 ±  | **`0.724 ±    | `0.755 ±      | `0.725 ±     | **2.53x / |
:                    :        : 0.113` /  : 0.091`** /    : 0.063` /      : 0.027` /     : 2.33x**   :
:                    :        : `5.846 ±  : **`2.512 ±    : `2.642 ±      : `3.410 ±     :           :
:                    :        : 0.037`    : 0.762`**      : 0.099`        : 1.301`       :           :
| **A100-SXM4-40GB** | `4096` | `6.439 ±  | **`1.841 ±    | `2.283 ±      | `2.088 ±     | **3.50x / |
:                    :        : 0.208` /  : 0.128`** /    : 0.068` /      : 0.067` /     : 3.18x**   :
:                    :        : `20.926 ± : **`6.584 ±    : `8.981 ±      : `8.185 ±     :           :
:                    :        : 0.097`    : 0.430`**      : 0.094`        : 0.970`       :           :

#### 3. Grouped MoE Matmul (`gpu_ragged_dot`, `bf16`, `G=8, K=1024, N=1024`)

| GPU                | Tokens | `jax.jit`        | Tokamax       | Tokamax       | Mosaic    |
:                    : (`M`)  : `lax.ragged_dot` : `("mosaic",)` : `("triton",)` : Speedup   :
:                    :        : Fwd / Fwd+Bwd    : Fwd / Fwd+Bwd : Fwd / Fwd+Bwd : vs        :
:                    :        : (`ms`)           : (`ms`)        : (`ms`)        : `jax.jit` :
:                    :        :                  :               :               : XLA       :
| :----------------- | :----: | :--------------: | :-----------: | :-----------: | :-------: |
| **H100 MIG         | `1024` | `0.437 ± 0.015`  | `0.285 ±      | **`0.284 ±    | **1.53x / |
: 1g.10gb**          :        : / `1.127 ±       : 0.018` /      : 0.034`** /    : 2.01x**   :
:                    :        : 0.012`           : **`0.560 ±    : `0.575 ±      :           :
:                    :        :                  : 0.014`**      : 0.057`        :           :
| **H100 MIG         | `4096` | `1.174 ± 0.047`  | `0.407 ±      | **`0.345 ±    | **2.88x / |
: 1g.10gb**          :        : / `3.712 ±       : 0.016` /      : 0.017`** /    : 4.13x**   :
:                    :        : 0.068`           : `0.899 ±      : **`0.800 ±    :           :
:                    :        :                  : 0.017`        : 0.014`**      :           :
| **A100-SXM4-40GB** | `1024` | `0.375 ± 0.029`  | `0.358 ±      | **`0.301 ±    | **1.05x / |
:                    :        : / `0.736 ±       : 0.084` /      : 0.053`** /    : 1.72x**   :
:                    :        : 0.059`           : `0.427 ±      : **`0.401 ±    :           :
:                    :        :                  : 0.036`        : 0.056`**      :           :
| **A100-SXM4-40GB** | `4096` | `0.607 ± 0.013`  | **`0.333 ±    | `0.350 ±      | **1.82x / |
:                    :        : / `1.891 ±       : 0.035`** /    : 0.119` /      : 3.18x**   :
:                    :        : 0.079`           : `0.594 ±      : **`0.575 ±    :           :
:                    :        :                  : 0.053`        : 0.093`**      :           :

### Multi-Host FrozenLake GRPO Validation

Validated end-to-end across 2 GPU hosts (`process_count=2`,
`Mesh('fsdp': 1, 'tp': 2)`, 30 steps of `Qwen/Qwen3-0.6B` on FrozenLake GRPO):

*   **Convergence**: `rewards/train/solve_ratio` improved from `0.8369` (Step 1)
    to `0.9995-1.0000` (Steps 4–30), with held-out `rewards/eval/solve_ratio`
    reaching `0.9900` and surrogate loss converging from `+1.960` to `+0.104`.
*   **Throughput & Memory**: Steady-state step latency reached `1.574s/step`
    (`1,300.7 tok/s`), with per-host peak HBM flat at `8.35 GiB` across all 30
    steps (zero memory leaks) and fused flash attention invoked across all 28
    decoder layers (`pallas_flash_attention_fwd_res`, `0.579 ms` mean latency).

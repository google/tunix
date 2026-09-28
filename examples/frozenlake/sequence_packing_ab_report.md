# Sequence packing and completion-only lm_head: trainer performance A/B

Gemma4-E2B agentic GRPO on FrozenLake (`examples/frozenlake/train_frozenlake.py`),
single v6e-8 host, branch `tokamax_seq_pack_val`. Runs executed 2026-09-26 to 2026-09-28.

## TL;DR

- **Sequence packing** (`max_seq_token_per_tpu=4096`) cuts trainer calls per step from 64 to 35.7 on average.
  The average falls from 50 early to 28–30 late, as completions get shorter.
  Device training time per step drops **42%** (25.8 s to 15.0 s).
  The cost of a single call does not drop (404 ms vs 419 ms), because every call is still an `[8, 4096]` forward+backward.
- In packed mode the lm_head used to run on every row position.
  **Restricting it to loss-masked positions** (`max_logp_positions_per_packed_row=2048`) cuts per-call time **11%** (419 ms to 372 ms).
  That puts packing at **−51% device training time per step** vs the unpacked baseline.
- The same idea on the **unpacked** path (`logp_gather_bucket_size=1024`) cuts per-call time **10%** overall (436 ms to 391 ms) and 14% in late training.
  Two limits apply:
  (a) one K per call (SPMD), so it only helps calls whose 8 generations all fit in 1024 loss-masked tokens: 33% of calls early, 85% late;
  (b) the second compiled program does not fit in HBM at `compute_logps_chunk_size=1024`, so this arm needs chunk 512, which is itself 8% slower.
- **End-to-end step time is not improved by any variant.** Step time here is bound by rollout and weight sync (~29 s/step, on the critical path).
  The trainer mostly overlaps with rollout.
  Packing also adds a ~3 s tail after the last rollout, because it waits for the full mini-batch before flushing the last chunks.
- Numerics are unchanged by the lm_head gather: identical `logp_diff` in paired runs, and unit tests check logps, entropy and gradient equality.
  Eval solve rates are within run-to-run noise (0.79–0.83 final).
  Both packed arms show a slightly higher sampler/trainer `logp_diff` (0.0106–0.0119 vs 0.0095–0.0098 unpacked).

## Setup

| Item | Value |
|---|---|
| Hardware | 1× TPU v6e-8 (8 chips, 31.25 GB HBM each); rollout and trainer share the chips |
| Model | `google/gemma-4-E2B-it`: 35 layers (28 sliding-window-512 + 7 global), 8 q heads / 1 kv head, head_dim 256, vocab 262144 |
| Trainer mesh | `(fsdp=8, tp=1)`, so one row per device per call |
| Rollout | vLLM (vllm-tpu / tpu-inference 0.29.0) server mode, mesh `(dp=2, tp=4)`, `hbm_utilization=0.2` |
| Algorithm | GRPO, `loss_algo=gspo-token`, RLOO advantages, `sampler_is="token"`, beta=0 |
| Batch | 64 prompts × 8 generations = 512 sequences per step, one optimizer update per step |
| Lengths | `max_prompt_length=2048`, `max_response_length=2048` |
| Common trainer config | tokamax splash attention (block 256, with PR openxla/tokamax#1446 disjoint-segment tile pruning), remat, `train_micro_batch_size=1` (1 prompt group = 8 rows per call), `logps_in_consumer=True`, `exact_token_continuity=False` |
| Steps / seed | 100 steps per arm, seed 42 |
| Software | jax/jaxlib 0.11.0, libtpu 0.0.44, local tokamax at `133416e` |

### Arms

| Arm | Packing | lm_head positions | `compute_logps_chunk_size` | Run | W&B |
|---|---|---|---|---|---|
| `nopack` | off | all 2048 completion slots | 1024 | `gemma4_fl_nopack_ab_20260926_051923` | [2zi57w1u](https://wandb.ai/linchai-google/tunix-frozenlake/runs/2zi57w1u) |
| `pack4096` | 4096 tokens/row | all 4095 row positions | 1024 | `gemma4_fl_pack4096_ab_20260926_141559` | [9g2vbv22](https://wandb.ai/linchai-google/tunix-frozenlake/runs/9g2vbv22) |
| `pack4096k2048` | 4096 tokens/row | loss-masked only, K=2048 | 1024 | `gemma4_fl_pack4096k2048_ab_20260927_005401` | [ysdjglx3](https://wandb.ai/linchai-google/tunix-frozenlake/runs/ysdjglx3) |
| `nopack_c512` | off | all 2048 completion slots | 512 | `gemma4_fl_nopack_c512_20260928_081850` | [6dbjg59d](https://wandb.ai/linchai-google/tunix-frozenlake/runs/6dbjg59d) |
| `nopackg1024_c512` | off | loss-masked only, K bucketed to {1024, full} | 512 | `gemma4_fl_nopackg1024_c512_20260927_235926` | [f0g7m9eu](https://wandb.ai/linchai-google/tunix-frozenlake/runs/f0g7m9eu) |

The first three arms form one comparable group (chunk 1024) and the last two form another (chunk 512).
The unpacked-gather arm could not run at chunk 1024 (see [Unpacked gather: HBM limit](#unpacked-gather-hbm-limit)), so it has its own chunk-512 control.

## Method

- Per-step timings come from the Tunix perf-v2 Perfetto trace
  (`gs://linchai-bucket-dev/perfetto/<run>/`).
  - **Device train time** is the sum of the device-timeline `peft_train` spans in a global step.
    **Per-call time** is that sum divided by the number of `train_step` calls.
  - **Tail** is the time from the end of the step's last rollout to the end of its last `peft_train` span: the trainer work left on the critical path.
  - **Weight sync** is the `weight_sync` span.
- Step 0 is excluded from every aggregate because it includes compilation. Aggregates cover steps 1–99; medians are used where eval steps (every 10) and the JAX-profiler window skew means.
- Global step time and training metrics come from W&B (`perf/train/global_step_time`, `sampler_trainer/train/logp_diff_mean`, `rewards/*`).

## Results

### Trainer cost

| Metric (steps 1–99) | nopack | pack4096 | pack4096k2048 | nopack_c512 | nopackg1024_c512 |
|---|---|---|---|---|---|
| `train_step` calls / step | 64 | 35.7 | 34.1 | 64 | 64 |
| Device time / call | 404 ms | 419 ms | **372 ms** | 436 ms | **391 ms** |
| Device train time / step | 25.8 s | 15.0 s | **12.7 s** | 27.9 s | **25.0 s** |
| vs its baseline | — | −42% | **−51%** | — | **−10%** |

Per-call time by training phase. Completions shorten from ~1000 to ~500 tokens over training.

| Steps | nopack | pack4096 | pack4096k2048 | nopack_c512 | nopackg1024_c512 |
|---|---|---|---|---|---|
| 1–19 | 402 ms | 417 ms | 371 ms | 435 ms | 413 ms (−5%) |
| 40–59 | 404 ms | 418 ms | 370 ms | 439 ms | 388 ms (−12%) |
| 80–99 | 403 ms | 420 ms | 375 ms | 437 ms | 375 ms (−14%) |

Device train time per step by phase:

| Steps | nopack | pack4096 | pack4096k2048 | nopack_c512 | nopackg1024_c512 |
|---|---|---|---|---|---|
| 1–19 | 25.8 s | 18.6 s | 16.6 s | 27.8 s | 26.4 s |
| 40–59 | 25.9 s | 14.8 s | 11.2 s | 28.1 s | 24.8 s |
| 80–99 | 25.8 s | 12.4 s | 10.6 s | 27.9 s | 24.0 s |

Packing efficiency (`pack4096`): 512 sequences packed into 50 chunks at step 1 and 28–30 chunks by step 90. The dummy (padding) ratio fell from 27% to 10–13%, and sequences per row rose from 1.4 to 2.2.

### End-to-end

| Metric (steps 1–99) | nopack | pack4096 | pack4096k2048 | nopack_c512 | nopackg1024_c512 |
|---|---|---|---|---|---|
| Global step time (median) | 237 s | 250 s | 219 s | 227 s | 212 s |
| Total wall time | 7.64 h | 7.79 h | 7.27 h | 7.39 h | 6.98 h |
| Tail after last rollout (median) | 0.34 s | 3.38 s | 3.14 s | 0.34 s | 0.35 s |
| Weight sync | 29.7 s | 29.5 s | 27.5 s | 28.7 s | 28.0 s |
| Mean completion length | 689 | 716 | 671 | 675 | 648 |

End-to-end differences are dominated by rollout.
The runs happened at different times, and their trajectories diverge, so mean completion lengths differ by up to ~10% (648–716).
They should not be attributed to the trainer changes.
The trainer-side effect that reaches the critical path is the tail: packing adds ~3 s per step because `pack_sequences` holds sequences until the mini-batch boundary before emitting its last chunks.

### Numerics

| Metric (steps 1–99) | nopack | pack4096 | pack4096k2048 | nopack_c512 | nopackg1024_c512 |
|---|---|---|---|---|---|
| `logp_diff_mean` (sampler vs trainer) | 0.0095 | 0.0106 | 0.0119 | 0.0098 | 0.0098 |
| Train solve ratio (mean) | 0.758 | 0.752 | 0.757 | 0.762 | 0.766 |
| Eval solve ratio (step 0 → final) | 0.576 → 0.800 | 0.569 → 0.791 | 0.576 → 0.825 | 0.565 → 0.808 | 0.565 → 0.826 |

- The lm_head gather does not change numerics.
  In the paired unpacked runs, `logp_diff` is identical (0.0098 vs 0.0098).
  In the packed pair, step 0 (same weights, same data) gives the same value (0.0107 vs 0.0107).
  The unit tests below assert equality of logps, entropy and gradients.
- Both packed arms have a ~10–25% higher `logp_diff` than the unpacked arms throughout training.
  The magnitude is still small, but it points to a small numerical difference in the segment-aware forward path (tokamax splash with multiple segments per row). It may be worth a separate look.

## Analysis

### Why packing alone does not make a call cheaper

Every call is an `[8, 4096]` forward+backward in both modes, so per-call cost is shape-bound:

1. **Attention is a small share and mostly not prunable.**
   A microbenchmark of tokamax splash fwd+bwd per row gave the following times.

   | Layer type | All real tokens | Baseline-like row | Packed row |
   |---|---|---|---|
   | Global (causal) | 1.99 ms | 1.41 ms | 0.9–1.1 ms |
   | Local (window 512) | 0.77 ms | 0.74 ms | 0.70–0.74 ms |

   The baseline-like row is `pad | 1000 prompt | 600 completion | pad`; the packed row holds 2–5 sequences.
   PR #1446's disjoint-segment tile pruning does work, per row: shard_map puts one row on each device, so the vmap batch is 1.
   But only the 7 global layers benefit.
2. **Dense layers and the lm_head scale with positions, not tokens.** Projections, MLP and PLE run on all 4096 positions regardless of padding.
3. **The packed lm_head ran on every row position.**
   The unpacked path slices the 2048 completion slots.
   The packed path had to keep all 4095 positions because prompts and completions interleave, and it discarded ~70% of the logits via `completion_mask`.

### Completion-only lm_head

Positions whose target token has `completion_mask > 0` are gathered into a static `[B, K]` buffer before the chunked lm_head. Results are scattered back, and other positions return 0, which the loss masks anyway.

- **Packed** (`max_logp_positions_per_packed_row=K`).
  The packer adds a third FFD bin constraint: loss-masked tokens per row ≤ K.
  This guarantees the static gather never drops a token.
  With K=2048 the constraint barely binds: 34.1 vs 35.7 calls per step, with mean completion lengths within 7%.
  - Saving: 47 ms per call, which is ~23 ms per 1024 positions.
  - That is ~4× a bare matmul+log_softmax microbenchmark, because the real path also computes entropy, temperature scaling, fp32 casts and remat recompute.
- **Unpacked** (`logp_gather_bucket_size=B`).
  K is the micro-batch's largest per-row loss-masked count, rounded up to a multiple of B, or no gather when that covers all 2048 slots.
  It rides on a static `TrainExample.logp_positions` field, so each distinct K compiles once.
  The per-call distribution is bimodal:

  | Steps | Calls gathered (K=1024) | Gathered call | Full call |
  |---|---|---|---|
  | 1–19 | 33% | 370 ms | 433 ms |
  | 40–59 | 67% | 365 ms | 434 ms |
  | 80–99 | 85% | 363 ms | 443 ms |

  A gathered call is ~16% cheaper.
  SPMD means all 8 devices share one K, so a single long or clipped generation in the group disables the gather for that call.

### Unpacked gather: HBM limit

At `compute_logps_chunk_size=1024` the unpacked arm OOMed with both bucket 512 and bucket 1024: `RuntimeProgramAllocationFailure: Attempting to allocate 164.42M ... 162.38M free`.
A second resident `train_step` program needs ~164 MB, and the unpacked baseline has only ~2 MB less than that to spare.
Switching the chunk to 512 frees enough lm_head temporaries, but a chunk-512 call is ~8% slower (436 vs 404 ms).
Net result: `nopackg1024_c512` (391 ms) beats the chunk-1024 baseline (404 ms) by only ~3%, while `pack4096k2048` reaches 372 ms with 44% fewer calls.

### Remaining per-call cost

Summing the microbenchmarked components gives ~100 ms per call, far below the measured 370–440 ms.
The remainder does not scale with token count; collective latency under FSDP is the working hypothesis.
It was not verified, because the JAX profiler captures in these runs contained host-plane data only and no TPU device plane.
Because this overhead is paid per call, **most of packing's saving comes from fewer calls**, not from cheaper calls.

## Recommendations

1. Use `max_seq_token_per_tpu` together with `max_logp_positions_per_packed_row` (K = `max_response_length` is sufficient here). This gives −51% trainer device time with no numerical change.
2. The unpacked `logp_gather_bucket_size` is only worth enabling when HBM headroom allows a second compiled `train_step`. Otherwise its prerequisite, a smaller chunk size, eats most of the gain.
3. For end-to-end speed in this setup, look at rollout and weight sync (~29 s/step on the critical path) before further trainer work.
   Packing's ~3 s tail could be reduced by emitting the final chunks earlier.
4. Before optimizing the per-call fixed cost, re-capture a JAX profile with the TPU device plane enabled to confirm what it is.

## Code changes (branch `tokamax_seq_pack_val`)

| Commit | Change |
|---|---|
| `49647be4`, `5aba8b56`, `a4f12205` | Recipe flags: `--max_seq_token_per_tpu`, `--max_steps`, `--splash_impl`, `--train_micro_batch_size`, `--compute_logps_chunk_size`, Perfetto/profiler, `DISABLE_TRAJECTORY_LOG` |
| `d8422e9f` | `AgenticRLConfig.logps_in_consumer`: serialize old-logp computation with `train_step` at micro batch 1. Without it the producer ran logps concurrently with training and the unpacked baseline OOMed. |
| `283aa18a` | Perfetto export via the perf-v2 tracer. The agentic learner emits no v1 spans. |
| `8ff22f1c` | Import `tpu_inference` before jax. tpu-inference ≥ 0.29 preloads `tpu_sync`, whose bundled XLA protos otherwise double-register and abort. |
| `0340b3b9` | Packed completion-only lm_head (`max_logp_positions_per_packed_row`) plus the packer cap and tests |
| `94867fe3` | Unpacked completion-only lm_head (`logp_gather_bucket_size`, static `TrainExample.logp_positions`) and tests |

### Tests

New tests:
- `tests/rl/common_test.py`: gathered vs full-row logps, entropy and gradients, packed and unpacked, with exact and loose K.
- `tests/rl/packing_test.py`: per-row loss-token cap and validation.
- `tests/rl/rl_utils_test.py`: bucket helper and config helpers.

`common`, `packing`, `rl_utils`, `algo_core`, `rl_learner`, `agentic_rl_learner` and `agentic_grpo_learner` all pass.
`rl_cluster_test` has 3 sglang failures, identical on the base commit (`sgl_jax` not installed).
Test files must run in separate processes; together they conflict on the CPU device count, which is pre-existing.

## Caveats

- One seed per arm. The arms ran sequentially on the same host, so rollout-dependent quantities are not paired.
- Environment issues found along the way:
  - Main's Gemma4 vLLM weight mapping (`0e459822`) requires tpu-inference ≥ 0.28. On 0.25, weight sync silently transferred nothing: vLLM kept dummy weights, `logp_diff` sat at ~15 and reward was 0. That run was discarded.
  - `exact_token_continuity=True` (main's default) crashes on budget-filling trajectories. The Gemma4 parser's per-turn `"\n"` is not counted against `max_response_length`, giving 2049/2048.
- The JAX profiler windows (trainer iterations 640–642) contain no TPU device plane. In the unpacked baseline the window also spanned rollout, which slowed one step.

## Reproduce

Scripts live outside the repo, in `~/.tmp/tokamax_seq_pack_val/`: `launch.sh <arm> <steps> <tag> [extra args]`.
The equivalent direct invocation (plus `WANDB_*`, `DISABLE_TRAJECTORY_LOG=1`, `PYTHONPATH=.`) is:

```bash
python3 examples/frozenlake/train_frozenlake.py --max_steps 100 --seed 42 \
  --splash_impl tokamax --train_micro_batch_size 1 --compute_logps_chunk_size 1024 \
  --perf_trace_dir gs://<bucket>/perfetto/<run> \
  [--max_seq_token_per_tpu 4096 [--max_logp_positions_per_packed_row 2048]]
# unpacked gather arm:
#   --compute_logps_chunk_size 512 --logp_gather_bucket_size 1024
```

The per-step analysis script (`per_step.py`) reads the Perfetto trace with `perfetto.trace_processor`.

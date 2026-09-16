# Launch configuration diff: `74x7dx50` (635 s/step) vs `ocl0zt4v` (3005 s/step)

Both runs are DeepSWE Qwen3-4B on the same cluster family. `74x7dx50` is the
run remembered as "deepswe 4b native, ~600 s/step"; it is confirmed at
**635.15 s/step mean over 64 steps**.

wandb's Python `config` is empty for every DeepSWE run — the trainer never calls
`wandb.config.update`, and it also ignores `CANON_WANDB_PROJECT` / `RUN_NAME` /
`GROUP`. The wandb **agent**, however, records raw `argv` in
`wandb-metadata.json`, and that is the only faithful record of how each run was
launched. This file diffs those two argv lists in full.

## 1. The launch flags are effectively identical

69 argv entries vs 68. Parsed into 63 vs 62 flags.

**Flags present in only one run:**

| run | flag |
|---|---|
| `74x7dx50` only | `--max_to_keep=8`, `--save_interval_steps=8` |
| `ocl0zt4v` only | `--seed=42` |

**Flags present in both with different values:**

| flag | `74x7dx50` | `ocl0zt4v` |
|---|---|---|
| `--ckpt_dir` | `.../canon-p58-ds4b-native-full-p58f14/checkpoints` | `none` |
| `--metric_logger_dir` | `.../canon-p58-ds4b-native-full-p58f14/metrics` | `.../canon-p58-ds4b-zero-hp-full-k34/metrics` |

**Everything else is byte-identical**, including every flag that could plausibly
affect rollout throughput:

```
--rollout_mesh_dp=8      --rollout_mesh_tp=8
--train_mesh_dp=8        --train_mesh_tp=8
--rollout_split_fraction=0.5        <- 128 chips split 64 rollout / 64 train
--rollout_vllm_max_num_seqs=16      <- x8 DP = 128 padded request slots
--max_num_batched_tokens=256        <- x8 DP = the 2048-token padded program
--max_concurrency=128
--vllm_utilization=0.6
--max_prompt_length=4096            --max_response_length=16384
--max_turns=50                      --num_generations=16
--batch_size=8                      --mini_batch_size=8
--train_micro_batch_size=8          --compute_logps_micro_batch_size=8
--rollout_micro_batch_size=1
--temperature=1.0 --top_k=0 --top_p=1.0
--advantage_estimator=rloo          --loss_agg_mode=sequence-mean-token-scale
--beta=0.0 --epsilon=0.2 --epsilon_high=0.28 --off_policy_steps=0
--learning_rate=1e-6 --b1=0.9 --b2=0.99 --weight_decay=0.01 --max_grad_norm=1.0
--enable_remat=True --remat_policy=decoder --no-optimizer-offload
--model_version=Qwen3-4B-Instruct-2507
--dataset_name=R2E-Gym/R2E-Gym-Subset
--dataset_revision=2e8108ff942f24fcb5686badfaf7f9a8808566d5
--per_turn_timeout_secs=300 --episode_timeout_secs=3000 --step_timeout_secs=600
--reward_timeout_secs=600 --cleanup_timeout_secs=300
--rollout_batch_timeout_secs=3600
```

> [!IMPORTANT]
> `--rollout_split_fraction=0.5` explains the `devices=64` seen in both logs:
> a 128-chip slice is split into 64 rollout chips and 64 training chips. The
> geometry, the scheduler window and the padded program shape are **the same in
> both runs**. The 4.7x step-time difference is therefore not a configuration
> difference.

## 2. What actually differs

| | `74x7dx50` (fast) | `ocl0zt4v` (slow) |
|---|---|---|
| started | 2026-08-23T19:26:17Z | 2026-09-01T23:20:06Z |
| **source commit** | **`24b1bbcf`** | **`c25f6f2d`** |
| host node | `gke-mlperf-v5p-cpu-np-ebb0f94d-zmls` | `gke-mlperf-v5p-canon-cpu-pool-f08a88e0-3rd5` |
| node vCPU as seen by agent | **64** | **16** |
| entrypoint | `examples/deepswe/canonical_entrypoint.py` | same |
| step time | 635.15 s (n=64) | 3004.67 s (n=22) |

The two commits are 343 apart on the same branch:

```
24b1bbcf  2026-08-23  Increase DeepSWE P58 tolerance with maxRestarts 3 and Pathways keepalives
c25f6f2d  2026-09-01  p58 deepswe: align Pathways Head pod resource requests for canon-cpu-pool scheduling
```

## 3. The head-pod CPU hypothesis, and why it is rejected

The node vCPU dropping 64 -> 16, together with a slow-run commit whose subject is
literally about head-pod resource requests, is a strong circumstantial lead: a
starved host driver would look exactly like "the same device program takes 4x
longer".

It was tested directly against the currently running bd07 head pod
(`canon-p58-128s-zsoptt-full-timbd07-pathways-head-0-0-l2wp5`) and **rejected**:

| measurement | value | reading |
|---|---|---|
| container `jax-tpu` cpu request / limit | 16 / 24 cores | ceiling is 24 |
| cgroup `cpu.max` | `2400000 100000` | 24 cores, confirmed |
| cgroup cumulative `usage_usec` | 4 948 987 497 us over 82 425 x 100 ms periods | **0.60 cores average** |
| cgroup `nr_throttled` / `nr_periods` | 9 / 82 425 | **0.011%** — effectively never throttled |
| `kubectl top` instantaneous | jax-tpu 523m, pathways-proxy 480m, pathways-rm 177m | ~1.2 cores of 24 |
| **busiest single thread** (5 s /proc sample) | **0.136 cores** | no thread anywhere near saturated |

The last row matters most: an aggregate of 0.6 cores would still be consistent
with one GIL-bound thread pegged at 100%, which is why the per-thread sample was
taken. It is not. The busiest thread of the `jax-tpu` container uses 13.6% of one
core, with three active threads totalling 0.23 cores.

**Conclusion: the head container is not CPU-starved, in aggregate or per-thread.
It is blocked waiting.** The node-vCPU difference recorded by the wandb agent
describes the *node*, not the container's cgroup ceiling, and does not explain
the gap.

> [!NOTE]
> Caveats on this rejection. The 5 s thread sample is a single snapshot and could
> have landed in a rollout-idle moment, though the cumulative cgroup counters
> over the whole run agree with it. The measurement covers the `jax-tpu`
> container only; `pathways-proxy` (480m instantaneous) brokers the actual
> dispatch to the 32 workers and was not thread-profiled. bd07 is not
> `ocl0zt4v`, so this rejects the hypothesis for the *current* configuration
> rather than retroactively for 2026-09-01.

## 4. What remains open

The gap is now narrowed to a **343-commit source range**, `24b1bbcf..c25f6f2d`,
with configuration and geometry held constant. Nothing in this package
identifies which commit is responsible; that has not been investigated.

Specific unexamined candidates:

* whether the fixed `2048` / `128` padding (`MIN_TOKEN_BUCKET`,
  `pad_per_rank=256`) was already in force on 2026-08-23 — `74x7dx50` predates
  the `[ENGINE_STEP]` instrumentation, so its padded shape is not directly
  recorded anywhere in this package;
* `tpu_inference` / `vllm-tpu` kernel differences between the 2026-08-23 image
  and the current one (`requirements.txt` for both runs is included here and has
  not yet been diffed);
* the `pathways-proxy` dispatch path.

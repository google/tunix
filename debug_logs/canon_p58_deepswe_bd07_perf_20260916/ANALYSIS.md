# canon-p58 DeepSWE bd07 — rollout performance evidence package

**Date**: 2026-09-16
**JobSet**: `canon-p58-128s-zsoptt-full-timbd07` (namespace `trellis`, queue `multislice-queue`)
**Commit under test**: `a5bda4cb0c634189541cf1c9d0f995a7c356b975` ("Segment DeepSWE actor logprob compilation", P78)
**Image**: `tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16`
**Head pod**: `canon-p58-128s-zsoptt-full-timbd07-pathways-head-0-0-l2wp5`
**Arm**: `CANON_P58_TIM_ARM=zero`, `CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM=treatment`
**Topology**: v5p `4x4x8` (128 chips), trainer mesh DP8xTP8, rollout mesh `devices=64`
**wandb**: `yuxzhang-google/tunix/tlxfco1e` (name `2026-09-15_22-15-21`)

This package answers two questions:

1. Did P78 fix the bd06 Pathways compile ceiling, and what does the training step
   cost now?
2. A historical run (`74x7dx50`, 2026-08-23) reached ~635 s/step. Why are we at
   ~2470 s/step on nominally the same geometry?

---

## 1. Step accounting (measured, not modelled)

Boundaries are `[CANON_P34_DP8] update_step_committed` timestamps from the head
pod log; the step-1 lower edge is the container entrypoint timestamp.

| step | window (UTC) | wall |
|---|---|---:|
| 1 | 22:08:38 -> 23:15:33 | **4016 s** (66.9 min) — includes startup + first compile |
| 2 | 23:15:33 -> 23:56:43 | **2470 s** (41.2 min) — steady state |
| 3 | 23:56:43 -> in progress at time of capture | — |

Trainer-side stages, summed per step from `[PERF] stage=... seconds=...`:

| stage | step 1 | step 2 | change |
|---|---:|---:|---|
| `segmented_value_and_grad` | 712.5 | **118.0** | -83% |
| `trainer_old` (P78 segmented old-logprob) | 227.0 | **121.7** | -46% |
| `rescore_b` | 91.7 | 80.4 | -12% |
| `optimizer_transaction` | 52.3 | **1.1** | -98% |
| `weight_sync` | 0.0 | 7.7 | — |
| `grad_accumulate` | 5.9 | 0.0 | — |
| **trainer subtotal** | **1089.3** | **335.0** | **-69%** |
| **residual (rollout + orchestration)** | **2926.7** (73%) | **2135.0** (86%) | — |

> [!IMPORTANT]
> After warmup the trainer costs **335 s of a 2470 s step (14%)**.
> `optimizer_transaction` settles to **1.1 s**, matching the 0.8 s steady state of
> the older `ocl0zt4v` run. **P78 resolved the compile ceiling and the trainer is
> no longer a meaningful cost.** All remaining headroom is in rollout.

P78 receipts are intact — see `raw/p78_actor_logps_receipts.log`. No `DATA_LOSS`,
no traceback, no fatal in the captured window.

---

## 2. The historical comparison

`74x7dx50` (wandb name `2026-08-23_19-26-17`, 64 steps) really did run fast:

```
perf/train/global_step_time : n=64  mean=635.15  min=426.81  max=960.56
```

`ocl0zt4v` (wandb name `2026-09-01_23-20-05`, 22 steps):

```
perf/train/global_step_time : n=22  mean=3004.67  min=2168.04  max=3728.10
```

### 2.1 Hypotheses tested and rejected

Each row is a measured comparison, not an argument.

| candidate explanation | `74x7dx50` (fast) | slow run / bd07 | verdict |
|---|---:|---:|---|
| fast run discards incomplete trajectories | `incomplete_prompt_ratio` mean **0.84** | **0.75** | **rejected** — fast is higher |
| fast run generates less | `mean_raw_length` mean **9854.7** | **7190.2** | **rejected** — fast is longer |
| smaller batch | `trajectory_solve_ratio` denominator **128** | **128** | rejected — identical |
| prefix caching was on | hit rate **0.0%** | **0.0%** | rejected — off in both |
| less wasted prefill | prefill/decode token ratio **40.0x** | **17.1x** | **rejected** — fast is worse |
| not running under Pathways | "Pathways" x260 in log | x9 | rejected — both Pathways |
| bigger rollout mesh | `devices=64`, `Mesh(data:8, model:8)` | `devices=64`, DP8xTP8 | rejected — identical |
| wider scheduler window | `max_num_batched_tokens=256` | `256` | rejected — identical |
| different model | `num_total_layers=36` | 36 | rejected — identical |

### 2.2 The one quantity that does differ

From vLLM's own throughput logger (same emitter, same fields, both runs),
restricted to genuinely decoding windows (`gen > 50 tok/s` and `running > 0`):

| metric (p50) | `74x7dx50` fast | bd07 now | ratio |
|---|---:|---:|---:|
| concurrent requests | 47 | 73 | — |
| generation throughput (tok/s) | 1326.9 | 479.2 | 2.8x |
| prefill throughput (tok/s) | 56361.3 | 9599.3 | 5.9x |
| **per-sequence decode rate (tok/s)** | **28.91** | **7.03** | **4.1x** |
| **implied engine step (ms)** | **34.6** | **142.3** | **4.1x** |

`56361 tok/s x 34.6 ms ~= 1950 tok`, i.e. the fast run was retiring an essentially
full **2048-token padded program every 34.6 ms**. bd07 needs 142.3 ms for the same
shape.

**The workload is not different; the same program executes ~4x slower.**

bd07's own `[ENGINE_STEP]` instrumentation agrees (n=6836):

```
   exec_ms: p50=  8.66     sample_ms: p50= 110.99
    gap_ms: p50=  8.46       sync_ms: p50=   1.05
  TOTAL_ms: p50=129.66
```

---

## 3. Readings that are NOT supported (do not repeat them)

`canon-zero-tim/debug_logs/r11_perf3_tile_v2_20260913/ANALYSIS.md` sections 4.2-4.3
already retracted the intuitive reading of these same fields. Repeating it here so
the retraction travels with the data:

| tempting reading | status |
|---|---|
| "the forward pass costs 6-9 ms" | **RETRACTED** — `exec_ms` times Python dispatch; JAX is async |
| "the ~111 ms is fixed host sampling overhead" | **RETRACTED** — the first blocking sync is inside `sample_tokens`, so `sample_ms` absorbs device execution time |
| "`sample_ms` is flat across load, therefore host overhead" | **RETRACTED** — the program is padded to a constant 2048 tokens, so constant device time is expected |
| "the `cd` path is much cheaper" | **RETRACTED** — different sync point, not a different cost |

What survives: a 2048-token program measured at 142 ms against a ~0.56 ms roofline
(64 v5p chips, bf16) gives **MFU ~0.4%**; the fast run's 34.6 ms gives **~1.6%**.
Both are poor; ours is 4x poorer. These receipts **cannot** say where inside the
step the time goes — that requires a profile.

---

## 4. Where the rollout time goes at the phase level

`74x7dx50` predates the `deepswe/train/timing/*` instrumentation, so this lane
exists only for `ocl0zt4v`. Its slowest step (step 21, 3728 s):

```
batch_elapsed              2591 s      (70% of the step)
  model_generation  p50    1440 s / max 2468 s   <- 94% of trajectory time
  sandbox_start     p50     110 s
  environment_step  p50       3 s
  environment_reset p50       3 s
  final_reward      p50       2 s
  cleanup           p50       2 s
```

Sandbox lifecycle, environment execution and reward scoring are **rounding errors**.
The cost is model generation. bd07's `[P58.BATCH_METRICS_JSON]` agrees: per-group
`completion_seconds` 871-950 s with 119-120 of 128 trajectories complete.

---

## 5. Contents

| path | description |
|---|---|
| `raw/engine_step_lines.log` | 6836 `[ENGINE_STEP]` receipts, timestamped |
| `raw/perf_stage_lines.log` | 5497 `[PERF] stage=` lines |
| `raw/p78_actor_logps_receipts.log` | 7 `[P78.ACTOR_LOGPS]` receipts |
| `raw/batch_metrics_json.log` | 2 `[P58.BATCH_METRICS_JSON]` blobs |
| `raw/vllm_throughput_lines.log` | 1095 vLLM throughput-logger lines |
| `raw/strictness_receipts.log` | 115 `CANON_ALIGN` / `P34_DP8` / `P33.DP8` / `REDUCE_ONCE` / `P28.G6` / `P41.OPTIMIZER` lines |
| `derived/engine_steps.csv` | `[ENGINE_STEP]` parsed to columns |
| `derived/perf_stages.csv` | stage timings, timestamped |
| `derived/vllm_throughput.csv` | prompt/gen throughput, concurrency, KV usage, prefix-cache hit rate |
| `derived/step_timeline.csv` | per-training-step boundaries with per-stage sums |
| `derived/history_74x7dx50_fast_635s.csv` | wandb per-step history of the fast run (curated column subset) |
| `derived/history_ocl0zt4v_slow_3000s.csv` | wandb per-step history of the slow run (curated column subset) |
| `baseline_runs/LAUNCH_CONFIG_DIFF.md` | **full argv diff of the two historical runs** — the flags are identical, which is what narrows the gap to a source-code range |
| `baseline_runs/74x7dx50_.../` | complete wandb mirror of the fast "native" run: all 139 history columns (`history.csv`, `history.jsonl`), `summary.json`, `config.yaml`, `wandb-metadata.json`, `requirements.txt`, and the full 54.2 MiB `output.log` |
| `baseline_runs/ocl0zt4v_.../` | same mirror for the slow run, all 149 history columns. **No `output.log`** — wandb has no console log server-side for this run |
| `SHA256SUMS` | checksums for everything above |

> [!NOTE]
> Console logs are treated asymmetrically here, deliberately.
>
> **bd07's head-pod console log is not included.** Only the receipt lanes above
> are extracted from it. Rationale matches
> `debug_logs/full_train_perf_20260908/README.md`: bd07 is still running, its log
> is still growing, and it can be re-pulled from the pod at any time.
>
> **`74x7dx50`'s `output.log` is included in full** (54.2 MiB, 751 242 lines).
> That run is `failed` and finished on 2026-08-23; its pod is long gone, so wandb
> is the only remaining copy and there is nothing to re-pull it from later. It
> compresses to ~4.2 MiB, which is what git actually stores, and it is the sole
> source of the vLLM throughput series that establishes the 34.6 ms engine step.
> It is below GitHub's 100 MiB hard limit but above the 50 MiB soft-warning
> threshold, so pushing it produces a warning. The repository already carries a
> 72.6 MiB console log under `wandb_exports/`, so this is within existing
> practice.
>
> `ocl0zt4v` has **no** console log server-side — wandb only holds
> `wandb_manifest.json`, `requirements.txt` and `wandb-metadata.json` for it. Its
> console output is unrecoverable.

### Provenance caveats

* Capture is a snapshot taken at 2026-09-16T00:14Z while bd07 was still running in
  training step 3. Steps 1-2 are complete; step 3 is not represented.
* `[ENGINE_STEP]` emission is throttled: every step for the first 5000, then every
  20th (`CANON_ENGINE_STEP_LOG_FULL` / `CANON_ENGINE_STEP_LOG_EVERY`). Aggregates
  over `derived/engine_steps.csv` are therefore **sampled, not exhaustive**, and
  must not be summed as if they were a complete census.
* wandb `config` is empty for every DeepSWE run — the trainer never populates it,
  and it also ignores `CANON_WANDB_PROJECT` / `RUN_NAME` / `GROUP`. Run id and
  `created_at` are the only stable identifiers.
* The 4.1x engine-step gap is **established**; its *cause* is **not**, but the
  search space is now much smaller. `baseline_runs/LAUNCH_CONFIG_DIFF.md` shows
  the fast and slow runs were launched with **effectively identical flags** —
  same `rollout_mesh_dp/tp=8/8`, same `rollout_split_fraction=0.5`, same
  `max_num_batched_tokens=256`, same `rollout_vllm_max_num_seqs=16`, same model
  and dataset revision. The only substantive differences are the source commit
  (`24b1bbcf` vs `c25f6f2d`, 343 commits apart) and the host node.
* The head-pod CPU-starvation hypothesis was raised by that host-node difference
  and then **rejected by direct measurement** on bd07: cgroup average 0.60 cores
  of a 24-core ceiling, `nr_throttled` 9 of 82 425 periods (0.011%), and a
  busiest single thread of **0.136 cores**. Details and caveats in
  `baseline_runs/LAUNCH_CONFIG_DIFF.md` section 3.
* Still unexamined: whether the fixed 2048/128 padding was already in force on
  2026-08-23; `tpu_inference` / `vllm-tpu` differences between the two images
  (both `requirements.txt` files are included under `baseline_runs/` and have
  not been diffed); the `pathways-proxy` dispatch path.


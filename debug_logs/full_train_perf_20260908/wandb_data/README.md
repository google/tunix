# FrozenLake full-training WandB exports

Per-step metric histories for the P57/M15 FrozenLake runs. Everything here is
indexed by `_step`, so each file plots directly as a step-wise curve.

## Runs

| Run ID | Experiment | Steps | Arm |
|---|---|---|---|
| `3osny0pb` | `canon-p57-fl-zero-m15-r10-06a0fdb9` | 0–233 | M15 Zero-TIM baseline, 15-turn |
| `e45soyn7` | `canon-p57-fl-zero-m15-r11-3f95bdc7` | 0–183 | M15 Zero-TIM tile-v2 / perf3, 15-turn |
| `tybj4xr0` | `canon-p57-fl-zero-r10a-06a0fdb9` | 0–300 | P45 Zero-TIM, 5-turn |

`ablation_summary.{csv,md}` is recomputed from the `history.csv` files in this
directory and therefore covers every run present.

> [!WARNING]
> Until 2026-09-15 the committed `3osny0pb` history held only **17 steps** of a
> run that actually reached **234**, and `e45soyn7` was absent entirely. Anything
> plotted from the old revisions of this directory is truncated. The files here
> now carry the complete exports; overlapping rows were verified byte-identical
> against the earlier copies before replacement, and the 57-column header is
> unchanged.

## Files per run

| File | Purpose |
|---|---|
| `history.csv` | 57 columns keyed by `_step`. **Use this for curves.** |
| `history.jsonl` | Same series, preserving the nested structure the CSV flattens. |
| `summary.json` | Terminal values. |
| `config.yaml` | Hyper-parameters. |

## What is deliberately not here: `console.log`

The raw console log is **intentionally excluded from version control**, not lost:

- r11's alone is **368,020,804 B (351 MiB)** — above GitHub's hard 100 MB
  per-file limit, so it cannot be pushed without LFS.
- It contributes nothing to per-step curves; every plotted series lives in
  `history.csv`.
- Git would keep the blob forever, taxing every future clone, and it could not
  be removed without rewriting history.
- WandB retains it server-side regardless of whether the JobSet still exists, so
  it can be re-pulled on demand with the `wandb-export` skill.

The local archive that this directory was built from, console log included, is
at `debug_logs/m15_final_20260915/` on the workstation (outside the repo).
Note that r10's console log was never successfully fetched — the export hit a
GraphQL timeout — so even the local archive is not a complete console set.

## Plotting

```python
import pandas as pd
df = pd.read_csv("3osny0pb_canon-p57-fl-zero-m15-r10-06a0fdb9/history.csv")
df.plot(x="_step", y="rewards/train/solve_ratio")
```

Commonly useful columns:

| Column | Meaning |
|---|---|
| `rewards/train/solve_ratio` | Batch solve rate |
| `rewards/train/solve_all` / `solve_none` / `solve_partial` | Outcome breakdown |
| `rewards/train/advantage/{mean,std,min,max}` | RLOO advantages |
| `trajectory_rewards/train/{mean,min,max,sum}` | Raw trajectory reward |
| `perf/train/global_step_time` | Wall-clock seconds per step |
| `perf/train/opt_optimizer_transaction_seconds` | Optimizer commit cost |
| `perf/train/weight_sync_seconds` | Rollout weight sync cost |
| `canonical/train/commit_gradient_norm` | Gradient norm at commit |
| `canonical/train/segmented_loss` | Training loss |
| `canonical/train/max_abs_parameter_delta` | Largest parameter change |
| `generation/train/completions/{mean,max,min}_length` | Completion lengths |
| `sampler_trainer/train/{logp_diff_mean,probs_pearson_corr}` | Sampler/trainer agreement |
| `trajectory/train/env_time/step_latency/mean` | Environment step latency |

# Qwen3 GSM8K VTC Math-Agent Quality Tuning Summary

Status: draft

Date documented: 2026-08-25

Scope: Tunix core agentic RL quality tuning on GSM8K VTC before applying the
same lessons to larger agentic workloads.

## Executive Summary

The GSM8K VTC math-agent recipe reached approximately 81% accuracy, matching
the GPU baseline. The current strongest recipe uses the NeMo-style script with
`batch_size=4`, `mini_batch_size=2`, and trainer-side old-logp recomputation.

These runs are not only GSM8K demo experiments. They are quality tuning and
ablation studies for Tunix agentic RL, covering prompt/parser behavior, reward
source, rollout runtime, model loading and dtype choices, old-logp source, and
KL mode.

Terminology note: the raw notes use "fullbatch=4/2" as experiment shorthand for
the script-level batch setting. In Tunix GRPO, the number of trajectories in one
full RL batch is `batch_size * num_generations`.

## Key Outcome

| Area | Result |
|---|---|
| GSM8K quality tuning | Reached approximately 81% accuracy, matching GPU baseline |
| Best current recipe | NeMo-style recipe with `batch_size=4`, `mini_batch_size=2` |
| Ablation status | Runs collected to identify key changes for single-turn math-agent stability |
| Related DeepSWE work | DeepSWE v7x was being debugged for vLLM compilation issues; marked fixed on Jun 7 in raw notes |

## Fix and Stabilization Summary

The quality recovery came from turning the GSM8K VTC script into a more stable
agentic RL recipe rather than changing only one isolated flag. The main fixes
are summarized below.

| Area | Problem addressed | Fix or stabilization change | Why it matters |
|---|---|---|---|
| Recipe structure | Earlier runs mixed prompt, reward, runtime, model-loading, and KL changes, making quality regressions hard to attribute | Added ablation presets and grouped runs into screening plus root-cause drilldowns | Makes each follow-up run isolate one likely source of convergence change |
| Batch shape | Some early experiments used very small or differently aligned full-batch/minibatch settings | Settled on the NeMo-style recipe with `batch_size=4`, `num_generations=8`, `mini_batch_size=2`, `train_micro_batch_size=1` | Keeps the train microbatch small while still using GRPO groups for reward normalization |
| Prompt/parser path | Chat-template and thinking behavior can change the exact text being optimized | Moved the target recipe to `raw_vtc`; kept `qwen_chat` as a rollback ablation | Reduces parser/template ambiguity and makes the action/answer protocol easier to compare across runs |
| Reward source | Posthoc reward and environment reward can disagree or log under different metric paths | Uses environment reward in the target recipe; keeps posthoc reward as an ablation | Ensures the reward used for optimization matches the math-agent environment outcome |
| Old-policy logps | Rollout-returned logps can differ from trainer-side scoring | Uses actor-side recomputed old logps with `use_rollout_logps=False` | Anchors PPO/GRPO ratios to the same scoring path used by the trainer |
| KL estimator | Direct KL and NeMo k2-style MSE KL may have different scale and stability | Uses `mse_kl`, with direct `kl` kept as a rollback ablation | Matches the intended NeMo-style recipe and separates KL-scale effects from other changes |
| Model dtype/loading | Model construction, actor dtype, reference dtype, and attention kernel changes can affect numerics | Uses split dtype loading with fp32 actor, bf16 reference, and flash attention enabled | Separates train precision from reference-model memory cost while keeping the target path efficient |
| Rollout runtime | High concurrency, prefix caching, and multiple inflight train computations can affect reproducibility and policy freshness | Target recipe disables rollout async scheduling and prefix caching, uses one inflight train computation | Makes the reference curve easier to reason about before reintroducing throughput optimizations |
| Metrics validity | Several runs had no reward metrics, making them unsafe for quality comparison | Marked no-reward-metric runs as skipped | Prevents drawing conclusions from runs where the main solve/reward signal is missing |

The current interpretation is that the stable recipe is the product of these
combined fixes. The ablation matrix is designed to determine which individual
fixes are essential and which are incidental.

## Main Run History

| Label | Script | Batch setting | W&B |
|---|---|---|---|
| Version May 22 | [`qwen3_grpo_gsm8k_vtc_demo.py`](https://github.com/google/tunix/blob/haoyu-quality/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py) | Not specified | [p5gpdpst](https://wandb.ai/linchai-google/tunix/runs/p5gpdpst?nw=nwuserhaoyugao) |
| Version May 25 | [`qwen3_grpo_gsm8k_vtc_demo.py`](https://github.com/google/tunix/blob/haoyu-quality/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py) | `batch_size=4`, `mini_batch_size=4` | [09e4nw67](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/09e4nw67?nw=nwuserhaoyugao) |
| FrozenLake-style + NeMo recipe | [`qwen3_grpo_gsm8k_vtc_nemo_recipe.py`](https://github.com/google/tunix/blob/haoyu-quality/examples/agentic/qwen3_grpo_gsm8k_vtc_nemo_recipe.py) | `batch_size=2`, `mini_batch_size=2` | [1mgfppbo](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/1mgfppbo?nw=nwuserhaoyugao) |
| Current reasonable recipe | [`qwen3_grpo_gsm8k_vtc_nemo_recipe.py`](https://github.com/google/tunix/blob/haoyu-quality/examples/agentic/qwen3_grpo_gsm8k_vtc_nemo_recipe.py) | `batch_size=4`, `mini_batch_size=2` | [bc7spz6r](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/bc7spz6r?nw=nwuserhaoyugao) |

## Current Target Recipe

`--ablation_preset final` is the target recipe used as the main control curve.

| Area | Final setting |
|---|---|
| Model | `Qwen/Qwen3-1.7B` |
| Batch | `batch_size=4`, `num_generations=8`, `mini_batch_size=2`, `train_micro_batch_size=1` |
| Parser | `raw_vtc` |
| Reward | Environment reward |
| Thinking | `qwen_enable_thinking=True`, but unused by `raw_vtc` parser |
| Model bundle | `final_split_dtype` |
| Reference dtype | `bf16` |
| Actor dtype | `fp32` |
| Flash attention | Enabled |
| Remat | Disabled |
| Rollout async scheduling | Disabled |
| Prefix caching | Disabled |
| Max concurrency | `batch_size * num_generations`, normally 32 |
| Max inflight train computations | 1 |
| Old logps source | Recomputed actor logps, because `use_rollout_logps=False` |
| KL mode | `mse_kl`, matching NeMo k2 style `0.5 * (logp - ref_logp)^2` |

## Ablation Commit Inventory

| Commit | W&B | Notes |
|---|---|---|
| [`2732c646e4494ccd29d4e9b243a9f2caec3a92ae`](https://github.com/google/tunix/commit/2732c646e4494ccd29d4e9b243a9f2caec3a92ae) | [ger4i2eg](https://wandb.ai/linchai-google/tunix/runs/ger4i2eg?nw=nwuserhaoyugao) | Included in ablation set |
| [`e248bf6fafc7cd23e1ffe2e581cf2b78a20208e4`](https://github.com/google/tunix/commit/e248bf6fafc7cd23e1ffe2e581cf2b78a20208e4) | [u5rolf8s](https://wandb.ai/linchai-google/tunix/runs/u5rolf8s?nw=nwuserhaoyugao) | No reward metrics; skip for quality comparison |
| [`2c347e05e017fac47284e23a576bc6f0d8d769fe`](https://github.com/google/tunix/commit/2c347e05e017fac47284e23a576bc6f0d8d769fe) | [s7uqqtly](https://wandb.ai/linchai-google/tunix/runs/s7uqqtly?nw=nwuserhaoyugao) | No reward metrics; skip for quality comparison |
| [`ae7707ea9bc1448baf0a9fb5fd69ddc619aeecc9`](https://github.com/google/tunix/commit/ae7707ea9bc1448baf0a9fb5fd69ddc619aeecc9) | [t51ws3x5](https://wandb.ai/linchai-google/tunix/runs/t51ws3x5?nw=nwuserhaoyugao) | No reward metrics; skip for quality comparison |
| [`ac1f49842b7f9e25c29ac4625c684584522cc318`](https://github.com/google/tunix/commit/ac1f49842b7f9e25c29ac4625c684584522cc318) | [8z7eam60](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/8z7eam60?nw=nwuserhaoyugao) | Included in ablation set |

## Ablation Design

The ablation presets are implemented in `apply_ablation_preset()`. The runner
grouping is implemented in
`examples/agentic/run_qwen3_grpo_gsm8k_vtc_ablations.py`.

The common launch flag `--pathways_enforce_subset_devices_form_subslice=false`
is appended by the script before JAX/absl parsing. It is shown in commands only
for explicitness.

### Screening Runs

| Run | Purpose | Main change from final | W&B |
|---|---|---|---|
| `screening_final` | Control run for the target recipe | None | [xyu2t5b6](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/xyu2t5b6?nw=nwuserhaoyugao) |
| `screening_oldish_full` | Full old-ish recipe rollback | `qwen_chat`, posthoc reward, legacy model copy, no flash attention, async rollout, prefix caching, high concurrency, direct `kl` | [dc4qxes3](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/dc4qxes3?nw=nwuserhaoyugao) |
| `screening_revert_rollout_runtime` | Test rollout/training runtime settings | Async rollout, prefix caching, `max_concurrency=1024`, `max_inflight_train_computations=2` | [jyyt54rk](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/jyyt54rk/overview?nw=nwuserhaoyugao) |
| `screening_revert_model_bundle` | Test model construction and attention | `legacy_copy`, flash attention disabled | [f69pi1m8](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/f69pi1m8?nw=nwuserhaoyugao) |
| `screening_revert_old_logps_to_rollout` | Test old-logp source | `use_rollout_logps=True` | [u6m5i9ss](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/u6m5i9ss?nw=nwuserhaoyugao) |
| `screening_revert_kl` | Test KL estimator/mode | `kl` instead of `mse_kl` | [gscbusev](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/gscbusev?nw=nwuserhaoyugao) |
| `screening_revert_prompt_reward` | Test combined prompt/parser/reward rollback | `qwen_chat`, posthoc reward, thinking enabled | [nunf1lmx](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/nunf1lmx?nw=nwuserhaoyugao) |

### Root-cause Drilldown Runs

| Run | Purpose | Main change from final | W&B |
|---|---|---|---|
| `rootcause_drilldown_revert_parser_only` | Isolate parser/chat-template/thinking from reward | `qwen_chat`, thinking enabled, reward remains env | [q4t9b7wf](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/q4t9b7wf?nw=nwuserhaoyugao) |
| `rootcause_drilldown_revert_reward_only` | Isolate reward computation mode from parser | Posthoc reward, parser remains `raw_vtc` | [utwluek8](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/utwluek8?nw=nwuserhaoyugao) |
| `rootcause_drilldown_revert_actor_dtype_only` | Isolate actor dtype from broader model bundle | Actor dtype `bf16`, model variant remains `final_split_dtype`, flash remains enabled | [lghtpxkx](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/lghtpxkx?nw=nwuserhaoyugao) |
| `rootcause_drilldown_revert_flash_only` | Isolate flash attention from broader model bundle | Flash attention disabled, model variant remains `final_split_dtype`, actor dtype remains `fp32` | [0u157y1i](https://wandb.ai/linchai-google/tunix-gsm8k-vtc/runs/0u157y1i?nw=nwuserhaoyugao) |

## Command Reference

Use `screening_final` as the control. Because it runs for 200 steps while most
ablations run for 125 steps, compare ablations against the first 125 steps of
the final curve. Use the full 200-step final curve as a stability reference.

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py \
  --ablation_preset final \
  --max_steps 200 \
  --experiment_tag screening_final \
  --pathways_enforce_subset_devices_form_subslice=false
```

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py \
  --ablation_preset oldish_full \
  --max_steps 125 \
  --experiment_tag screening_oldish_full \
  --pathways_enforce_subset_devices_form_subslice=false
```

For the remaining ablations, replace `--ablation_preset` and
`--experiment_tag` with the run names in the tables above and keep
`--max_steps 125`.

## Comparison Guide

Primary curves to compare:

| Curve | Why it matters |
|---|---|
| `rewards/solve_ratio` | Main solve-rate signal for GSM8K VTC |
| `rewards/reward_mean` | Dense reward trend, including partial credit |
| `rewards/solve_all`, `rewards/solve_none`, `rewards/solve_partial` | Group-level outcome distribution |
| Training loss, policy loss, KL, entropy | Distinguish optimization instability from reward/parser effects |
| Logp-diff diagnostics, if present | Check trainer-vs-rollout logp agreement |
| Generation length metrics | Detect verbosity, collapse, or token-budget shifts |

Suggested interpretation flow:

1. Compare each ablation to `screening_final` over the first 125 steps.
2. If `oldish_full` improves, use drilldowns to split parser/reward, model
   bundle, runtime, old-logp source, and KL effects.
3. If `revert_prompt_reward` improves, compare `revert_parser_only` and
   `revert_reward_only`.
4. If `revert_model_bundle` improves, compare `revert_actor_dtype_only` and
   `revert_flash_only`.
5. If `revert_old_logps_to_rollout` differs materially, inspect sampler-trainer
   logp-diff diagnostics before changing the default old-logp source.

## Current Takeaways

* The NeMo-style recipe is the current reference for GSM8K VTC quality.
* The strongest known configuration uses `batch_size=4`, `mini_batch_size=2`,
  `raw_vtc`, environment rewards, actor-side old-logp recompute, fp32 actor
  dtype, bf16 reference dtype, flash attention, and `mse_kl`.
* Runs without reward metrics should be excluded from quality comparisons.
* The ablation structure is designed to identify which single-turn recipe
  choices matter before transferring lessons to DeepSWE.

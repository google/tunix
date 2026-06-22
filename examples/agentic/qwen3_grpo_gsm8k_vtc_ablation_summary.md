# Qwen3 GSM8K VTC Math-Agent Ablation Summary

This note summarizes the GSM8K VTC ablation runs driven by
`examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py`.

This is a **math-agent tuning experiment** for Tunix core agentic RL, not just a
standalone GSM8K demo. The agent is trained to solve math problems with an
explicit reasoning-and-answer protocol, and these ablations test which Tunix
recipe choices are necessary for stable convergence.

The ablation presets are implemented in `apply_ablation_preset()`. The runner
grouping is implemented in `examples/agentic/run_qwen3_grpo_gsm8k_vtc_ablations.py`.

## Baseline Recipe

`--ablation_preset final` is the current target recipe. Important defaults:

| Area | `final` setting |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B` |
| Batch | `batch_size=4`, `num_generations=8`, `mini_batch_size=2`, `train_micro_batch_size=1` |
| Parser | `parser_mode=raw_vtc` |
| Reward | `reward_mode=env` |
| Thinking | `qwen_enable_thinking=True`, but unused by `raw_vtc` parser |
| Model bundle | `model_variant=final_split_dtype` |
| Reference dtype | `bf16` |
| Actor dtype | `fp32` |
| Flash attention | enabled |
| Remat | disabled |
| Rollout async scheduling | disabled |
| Prefix caching | disabled |
| Max concurrency | `batch_size * num_generations`, normally `32` |
| Max inflight train computations | `1` |
| Old logps source | recomputed actor logps, because `use_rollout_logps=False` |
| KL mode | `mse_kl`, matching NeMo k2 style `0.5 * (logp - ref_logp)^2` |

The primary goal of this experiment set is to tune and validate the Tunix math
agent recipe: prompt/parser behavior, reward source, rollout runtime, model
loading/dtype, old-logp source, and KL mode are all treated as core system
variables that can affect convergence.

The common launch flag
`--pathways_enforce_subset_devices_form_subslice=false` is already appended by
the script before JAX/absl parsing, but it is included in the commands below for
explicitness.

## Screening Runs

### 1. `screening_final`

Purpose: control run for the current target recipe.

What it tests: this is not an ablation; it is the reference curve for all other
comparisons.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset final --max_steps 200 --experiment_tag screening_final --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add final control curves here.
```

### 2. `screening_oldish_full`

Purpose: full old-ish recipe rollback.

What it changes relative to `final`:

| Area | `oldish_full` setting |
| --- | --- |
| Parser | `qwen_chat` |
| Reward | `posthoc` |
| Thinking | enabled |
| Model bundle | `legacy_copy` |
| Flash attention | disabled |
| Rollout async scheduling | enabled |
| Prefix caching | enabled |
| Max concurrency | `1024` |
| Max inflight train computations | `2` |
| KL mode | `kl` |

Interpretation: if this run recovers behavior relative to `final`, the root
cause is somewhere in the combined old bundle. Use the drilldown runs below to
separate prompt/reward, rollout runtime, model construction, logp source, and KL.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset oldish_full --max_steps 125 --experiment_tag screening_oldish_full --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add oldish_full curves here.
```

### 3. `screening_revert_rollout_runtime`

Purpose: test whether rollout/training runtime settings explain the behavior.

What it changes relative to `final`:

| Area | `revert_rollout_runtime` setting |
| --- | --- |
| Rollout async scheduling | enabled |
| Prefix caching | enabled |
| Max concurrency | `1024` |
| Max inflight train computations | `2` |

Interpretation: if this improves over `final`, the cause is likely in runtime
scheduling/cache/concurrency/inflight behavior rather than parser, reward, model
weights, KL, or old-logp source.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_rollout_runtime --max_steps 125 --experiment_tag screening_revert_rollout_runtime --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add rollout runtime curves here.
```

### 4. `screening_revert_model_bundle`

Purpose: test whether model construction and attention implementation explain
the behavior.

What it changes relative to `final`:

| Area | `revert_model_bundle` setting |
| --- | --- |
| Model bundle | `legacy_copy` |
| Flash attention | disabled |

Important code detail: in the `legacy_copy` path, the actor is copied from the
reference model when LoRA is disabled. Because the default reference dtype is
`bf16`, this preset also couples actor construction to the reference-loaded
model path. The `revert_actor_dtype_only` and `revert_flash_only` runs below
split two obvious parts of this bundle.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_model_bundle --max_steps 125 --experiment_tag screening_revert_model_bundle --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add model bundle curves here.
```

### 5. `screening_revert_old_logps_to_rollout`

Purpose: test whether old-policy logps should come from rollout/vLLM rather
than trainer-side recomputation.

What it changes relative to `final`:

| Area | `revert_old_logps_to_rollout` setting |
| --- | --- |
| `use_rollout_logps` | `True` |

Interpretation: `final` recomputes old logps on the actor/trainer path.
This run uses rollout-returned logps as the old-policy denominator. If this
changes convergence or logp-diff curves, the issue may be sampler/trainer logp
alignment or recompute semantics.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_old_logps_to_rollout --max_steps 125 --experiment_tag screening_revert_old_logps_to_rollout --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add old-logps-source curves here.
```

### 6. `screening_revert_kl`

Purpose: test whether the KL estimator/mode explains the behavior.

What it changes relative to `final`:

| Area | `revert_kl` setting |
| --- | --- |
| KL mode | `kl` |

Interpretation: `final` uses `mse_kl`, which is intended to match NeMo k2.
This run reverts to direct `kl`. If this changes learning, compare KL magnitude,
policy loss, entropy, and reward curves.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_kl --max_steps 125 --experiment_tag screening_revert_kl --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add KL curves here.
```

### 7. `screening_revert_prompt_reward`

Purpose: test the combined prompt/parser/reward rollback.

What it changes relative to `final`:

| Area | `revert_prompt_reward` setting |
| --- | --- |
| Parser | `qwen_chat` |
| Reward | `posthoc` |
| Thinking | enabled |

Interpretation: this combines the old parser/chat-template path and the old
posthoc reward path. If this is strong, use `revert_parser_only` and
`revert_reward_only` to split the factor.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_prompt_reward --max_steps 125 --experiment_tag screening_revert_prompt_reward --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add prompt+reward curves here.
```

## Root-Cause Drilldown Runs

### 8. `rootcause_drilldown_revert_parser_only`

Purpose: isolate parser/chat-template/thinking from reward.

What it changes relative to `final`:

| Area | `revert_parser_only` setting |
| --- | --- |
| Parser | `qwen_chat` |
| Thinking | enabled |
| Reward | remains `env` |

Interpretation: compare this directly to `screening_revert_prompt_reward` and
`rootcause_drilldown_revert_reward_only`. If parser-only reproduces the effect,
the root cause is likely prompt formatting/chat template/thinking behavior.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_parser_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_parser_only --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add parser-only curves here.
```

### 9. `rootcause_drilldown_revert_reward_only`

Purpose: isolate reward computation mode from parser.

What it changes relative to `final`:

| Area | `revert_reward_only` setting |
| --- | --- |
| Reward | `posthoc` |
| Parser | remains `raw_vtc` |

Interpretation: if reward-only reproduces the effect, the root cause is likely
the difference between environment reward and posthoc reward, not chat template.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_reward_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_reward_only --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add reward-only curves here.
```

### 10. `rootcause_drilldown_revert_actor_dtype_only`

Purpose: isolate actor model dtype from the broader model bundle.

What it changes relative to `final`:

| Area | `revert_actor_dtype_only` setting |
| --- | --- |
| Actor dtype | `bf16` |
| Model variant | remains `final_split_dtype` |
| Flash attention | remains enabled |

Interpretation: compare against `screening_revert_model_bundle`. If actor dtype
alone matches the model-bundle effect, dtype is a likely root cause. If not,
model loading/copy or flash attention remains suspect.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_actor_dtype_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_actor_dtype_only --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add actor-dtype-only curves here.
```

### 11. `rootcause_drilldown_revert_flash_only`

Purpose: isolate flash attention from the broader model bundle.

What it changes relative to `final`:

| Area | `revert_flash_only` setting |
| --- | --- |
| Flash attention | disabled |
| Model variant | remains `final_split_dtype` |
| Actor dtype | remains `fp32` |

Interpretation: compare against `screening_revert_model_bundle`. If flash-only
matches the model-bundle effect, attention kernel behavior is a likely root
cause. If actor dtype and flash-only both fail to match, the remaining suspect
is the legacy model construction/copy path itself.

Command:

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_flash_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_flash_only --pathways_enforce_subset_devices_form_subslice=false
```

Attach curves:

```text
TODO: add flash-only curves here.
```

## Comparison Guide

Use `screening_final` as the control. Because `screening_final` runs for 200
steps while most ablations run for 125, compare ablations against the first 125
steps of the final curve, then use the full 200-step final curve as a stability
reference.

Suggested primary curves:

| Curve | Why it matters |
| --- | --- |
| `rewards/solve_ratio` | Main solve-rate signal for GSM8K VTC |
| `rewards/reward_mean` | Dense reward trend, including partial credit |
| `rewards/solve_all`, `rewards/solve_none`, `rewards/solve_partial` | Group-level outcome distribution |
| `train/loss`, policy loss, KL, entropy | Distinguish optimization instability from reward/parser effects |
| Logp-diff diagnostics, if present | Check trainer-vs-rollout logp agreement |
| Generation length metrics | Detect verbosity/collapse/token budget shifts |

Interpretation shortcuts:

| Observation | Likely conclusion |
| --- | --- |
| `oldish_full` improves, but none of the split runs improve | Interaction effect across multiple old settings |
| `revert_prompt_reward` improves | Prompt/parser/reward bundle is important |
| `revert_parser_only` improves more than `revert_reward_only` | Parser/chat template/thinking is the likely driver |
| `revert_reward_only` improves more than `revert_parser_only` | Reward timing/source is the likely driver |
| `revert_rollout_runtime` improves | Runtime scheduling/cache/concurrency/inflight settings matter |
| `revert_model_bundle` improves | Model construction, effective dtype, or flash attention matters |
| `revert_actor_dtype_only` improves | Actor dtype is likely important |
| `revert_flash_only` improves | Flash attention/kernel behavior is likely important |
| `revert_old_logps_to_rollout` improves | Old-logp source or sampler/trainer logp mismatch is likely important |
| `revert_kl` improves | KL estimator/magnitude is likely important |

## Raw Command List

```bash
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset final --max_steps 200 --experiment_tag screening_final --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset oldish_full --max_steps 125 --experiment_tag screening_oldish_full --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_rollout_runtime --max_steps 125 --experiment_tag screening_revert_rollout_runtime --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_model_bundle --max_steps 125 --experiment_tag screening_revert_model_bundle --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_old_logps_to_rollout --max_steps 125 --experiment_tag screening_revert_old_logps_to_rollout --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_kl --max_steps 125 --experiment_tag screening_revert_kl --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_prompt_reward --max_steps 125 --experiment_tag screening_revert_prompt_reward --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_parser_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_parser_only --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_reward_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_reward_only --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_actor_dtype_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_actor_dtype_only --pathways_enforce_subset_devices_form_subslice=false
/home/haoyugao_google_com/tunix/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py --ablation_preset revert_flash_only --max_steps 125 --experiment_tag rootcause_drilldown_revert_flash_only --pathways_enforce_subset_devices_form_subslice=false
```

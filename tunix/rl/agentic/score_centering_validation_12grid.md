# Score Centering: 12x12 FrozenLake and Qwen3 A/B report

Follow-up to [`score_centering_validation.md`](score_centering_validation.md).
That report established that the Score Centering (SC) implementation is
correct once the rollout transport is fixed. This one asks two further
questions:

1. Do the SC diagnostics stay healthy on a harder dataset with longer
   multi-turn trajectories?
2. Does SC measurably change training compared with an identical SC-off run?

**Run:** 2026-09-23, single 8-chip TPU v6e host, vLLM 0.25.0 (tpu-inference
0.25.0) in server mode, JAX 0.10.2.

**Dataset:** FrozenLake regenerated with grid sizes 2x2 to 12x12 (previously
2x2 to 9x9), 10,000 train and 100 test prompts in `data/frozenlake/`.
`gs://tunix/data/Frozenlake` still holds the old 2x2 to 9x9 set, so these runs
read the local copy through `DATA_DIR`.

---

## 1. Verdict

- **The SC diagnostics are healthy on both models and do not drift.** Over 30
  steps on Gemma4-E2B and 15 steps on Qwen3-1.7B, `abs_coeff_sum_mean`,
  `head_mass_*` and `tail_ratio_rho` stay inside a narrow band and show no
  upward trend. The larger grids change none of the SC readings compared with
  the 9x9 validation run.
- **SC produced no measurable difference in training outcome.** On
  Qwen3-1.7B, paired SC-on and SC-off runs saw the same prompts at every step.
  Mean reward was 0.203 vs 0.197, and grad norm, entropy and every
  sampler/trainer agreement metric matched within noise.
- **The one consistent difference is in the PPO ratio.** With SC on,
  `pg_clipfrac` is 0 by construction. With SC off, it averages 4.3% because the
  recipe's gspo-token clip band (0.003 / 0.005) is about as wide as the
  measured |log ratio| (~0.003).
- **This setup is close to the least favourable one for showing an SC
  effect.** The runs are strictly on-policy (`off_policy_steps=0`,
  `num_iterations=1`) with LR 1e-6 and only 15 steps, so the only mismatch
  source is bf16 numerics. §4 lists the setups that should separate the arms.

---

## 2. Gemma4-E2B on the 12x12 dataset (SC on only)

`examples/frozenlake/train_frozenlake.py`: rollout mesh `(2, 4)`, trainer mesh
`(8, 1)`, `batch_size=8`, `num_generations=4`, `num_batches=10` (30 steps),
`score_centering_top_k=32`, `sampler_is="token"`, `exact_token_continuity=False`.

### 2.1 SC diagnostics

| metric | 12x12, 30 steps | 9x9, 6 steps (previous report) |
|---|---|---|
| `abs_coeff_sum_mean` | 2.9e-4 to 5.7e-4, mean ~3.9e-4 | 3.3e-4 to 5.0e-4 |
| `head_mass_q_mean` | 0.99995 to 0.99998 | ~0.99996 |
| `head_mass_p_mean` | equal to q to 5 decimals | same |
| `tail_ratio_rho_mean` | 0.987 to 0.992 | 0.988 to 0.990 |
| `ppo_kl` | exactly 0 | exactly 0 |
| `sampler_is/weight_mean` | 0.9996 to 1.0005 | ~1.0000 |
| `sampler_is` fraction clipped | <= 0.016% | — |

Longer multi-turn trajectories on bigger grids, several of which hit
`MAX_CONTEXT_LIMIT_REACHED` at the 2048-token response budget, did not widen
the sampler/trainer gap. k=32 remains ample.

### 2.2 Single-token `mult_err` outliers are invisible to SC

Most steps have `mult_err` mean ~1.01 with max 2 to 20. Three do not:

| rollout step | `mult_err` mean | `mult_err` max |
|---|---|---|
| 9 | 1.03 | 502 |
| 25 | **1.40** | **14,781** (\|Δlogp\| ≈ 9.6) |
| 27 | 1.13 | 1,722 |

`abs_coeff_sum_mean` and the TIS clip fraction do not move on these steps.
That is expected: SC measures distribution-level disagreement over the top-k
head, and an extreme ratio on a low-probability sampled token contributes
almost nothing in probability space. However, a batch mean of 1.40 means many
tokens disagreed badly, which is more than bf16 noise.

**Unverified hypothesis:** with `exact_token_continuity=False` the trainer
re-tokenizes the multi-turn conversation, and tokens at turn boundaries no
longer match what the sampler emitted. Checking whether the outlier positions
cluster at turn boundaries would confirm or rule this out.

### 2.3 Empty steps

12 of 30 steps had `grad_norm=0`. With `num_generations=4` and a strongly
bimodal dataset (small grids are always solved, large grids never are), every
group in a batch can have identical rewards, which makes every RLOO advantage
zero. The SC diagnostics are still computed on these steps, but SC has no
effect on the update. §3 uses `num_generations=8` for this reason.

---

## 3. Qwen3-1.7B A/B (SC on vs SC off)

`examples/frozenlake/train_frozenlake_qwen3.py --model_version Qwen/Qwen3-1.7B`:

- Trainer mesh `(1, 8)`, rollout mesh `(2, 4)` (see §5.1).
- `batch_size=16`, `mini_batch_size=16`, `num_generations=8`, giving 128
  trajectories per step.
- `num_batches=10`, stopped after 15 steps (step 0 to 14).
- `seed=42`, `score_centering_top_k=32`, `sampler_is="token"`,
  `loss_algo="gspo-token"` (epsilon 0.003 / 0.005).
- `exact_token_continuity=False`, env `max_steps=8`.

The only difference between the arms is `--score_centering`. Both runs use the
same seed and `num_batches`, so step *s* sees the same prompts in both arms.
Runs were stopped at 15 steps rather than using `--num_batches 5`, which would
have changed the prompts seen at steps 5 to 14. LR is constant, so an early
stop is equivalent to a shorter schedule.

### 3.1 Per-step comparison

Reward is the rollout at step *s*. The actor metrics are those of the update
that consumed that rollout (TensorBoard actor step *s+1*). "—" means the value
was lost when the run was stopped (§5.4).

| step | reward on | reward off | `pg_clipfrac` off | `log_ratio` off | `abs_coeff_sum` on | `rho` on |
|---|---|---|---|---|---|---|
| 0 | 0.242 | 0.273 | 0.012 | 0.0010 | 4.79e-03 | 0.976 |
| 1 | 0.289 | 0.211 | 0.073 | 0.0027 | 5.55e-03 | 0.977 |
| 2 | 0.219 | 0.188 | 0.090 | 0.0036 | 4.91e-03 | 0.976 |
| 3 | 0.125 | 0.172 | 0.021 | 0.0027 | 4.53e-03 | 0.978 |
| 4 | 0.141 | 0.109 | 0.052 | 0.0042 | 4.04e-03 | 0.978 |
| 5 | 0.234 | 0.234 | 0.034 | 0.0030 | 3.94e-03 | 0.983 |
| 6 | 0.109 | 0.133 | 0.026 | 0.0031 | 3.71e-03 | 0.983 |
| 7 | 0.062 | 0.062 | 0.014 | 0.0027 | 4.28e-03 | 0.980 |
| 8 | 0.297 | 0.258 | 0.036 | 0.0027 | 5.27e-03 | 0.978 |
| 9 | 0.211 | 0.250 | 0.031 | 0.0035 | 3.95e-03 | 0.976 |
| 10 | 0.297 | 0.258 | 0.097 | 0.0034 | 5.18e-03 | 0.971 |
| 11 | 0.219 | 0.305 | 0.061 | 0.0033 | 3.99e-03 | 0.980 |
| 12 | 0.242 | 0.180 | 0.015 | 0.0028 | 4.84e-03 | 0.977 |
| 13 | 0.156 | 0.125 | — | — | 5.03e-03 | 0.975 |
| 14 | 0.109 | 0.141\* | — | — | — | — |

\* From the stdout step log. The TensorBoard point was lost at shutdown.

With SC on, `pg_clipfrac` and `log_ratio/abs_mean` are exactly 0 at every step,
and `is_ratio/max` is exactly 1.

### 3.2 Aggregates

| metric | SC on | SC off |
|---|---|---|
| reward, steps 0 to 13 | 0.203 | 0.197 |
| `grad_norm` | 0.373 | 0.379 |
| `entropy` | 0.0756 | 0.0753 |
| `pg_clipfrac` | **0** | **0.043** (0.012 to 0.097) |
| `log_ratio/abs_mean` | **0** | **0.0030** |
| `is_ratio/max` | 1.000 | ~1.017 |
| `sampler_trainer/mult_prob_error_mean` | 1.012 | 1.012 |
| `sampler_trainer/mult_prob_error_max` (mean over steps) | 9.7 | 9.1 |
| `sampler_trainer/probs_pearson_corr` | 0.985 | 0.985 |
| `sampler_is/weight_mean` | 0.9997 | 0.9999 |
| `sampler_is` fraction clipped | 0.026% | 0.032% |
| TensorBoard scalar tags | 72 | 68 |

The reward difference of 0.006 is far below the step-to-step spread of 0.06 to
0.31. The two per-step reward curves track each other closely, which confirms
that reward variance here is driven by grid difficulty, not by SC. The 4 extra
tags are the `score_centering/*` diagnostics.

### 3.3 Interpretation

- **SC is doing real correction work on Qwen, but not enough to change the
  outcome.** `abs_coeff_sum_mean ≈ 4.6e-3` is about 10x Gemma's value and falls
  in the "substantive centering" band (`1e-3` to `5e-2`) described in
  `score_centering_validation.md` §3.4. It is flat across the run (3.7e-3 to
  5.6e-3). At LR 1e-6 over 15 on-policy steps, that correction does not show
  up in reward, entropy or grad norm.
- **`head_mass_q = head_mass_p = 1.0000` on Qwen.** The policy is extremely
  peaked (entropy ~0.075), so the top-32 head holds all the mass and k could be
  reduced further to shrink the rollout payload. `rho ≈ 0.978` is therefore a
  ratio of two tail masses near the `eps` floor, and should not be read as a
  distribution split.
- **With SC off, the PPO clip is binding on noise.** Each step here makes
  exactly one optimizer update and `num_iterations=1`, so the policy does not
  change between the start-of-step logp recompute and the loss forward.
  |log ratio| ≈ 0.003 is therefore most likely numerical disagreement between
  those two trainer forwards, not policy movement. Against a gspo-token clip
  band of 0.003 / 0.005, that noise is enough to clip about 4% of tokens. SC
  with `num_iterations=1` pins the ratio at 1 and avoids this entirely. This
  is a mechanism difference, not an outcome difference, and the explanation is
  not yet verified.
- **Qwen's gap is larger than Gemma's for an unconfirmed reason.** Possible
  causes are the rollout/trainer sharding mismatch (vLLM DP=2 x TP=4 against
  trainer TP=8, both bf16) and model-specific numerics. Neither was isolated.

---

## 5. Reproduction

Recipe changes (uncommitted at the time of writing):

- `train_frozenlake.py`: `DATA_DIR` and `TB_LOG_DIR` are now overridable
  through environment variables.
- `train_frozenlake_qwen3.py`:
  - adds `--model_version` (`Qwen/Qwen3-8B`, `Qwen/Qwen3-1.7B`);
  - adds `--score_centering*`, `--no_exact_token_continuity` and
    `--disable_eval`;
  - sets `num_logprobs` before `RLEngine` is built;
  - adds a `(2, n/2)` rollout mesh;
  - sets `compute_logps_chunk_size=2048`;
  - makes `DATA_DIR`, `TB_LOG_DIR` and `WANDB_RUN_NAME` overridable through
    environment variables.

```bash
# Gemma4-E2B, 12x12, SC on, 30 steps.
DATA_DIR=$PWD/data/frozenlake PYTHONPATH=$PWD \
  python examples/frozenlake/train_frozenlake.py \
    --batch_size 8 --mini_batch_size 8 --num_generations 4 --num_batches 10 \
    --disable_eval --no_exact_token_continuity \
    --score_centering --score_centering_top_k 32

# Qwen3-1.7B A/B. Run each arm in its own process group
# (setsid) and stop it after "[step 14]" is logged.
COMMON="--model_version Qwen/Qwen3-1.7B --batch_size 16 --mini_batch_size 16 \
  --num_generations 8 --num_batches 10 --seed 42 --disable_eval \
  --no_exact_token_continuity --score_centering_top_k 32"
DATA_DIR=$PWD/data/frozenlake PYTHONPATH=$PWD setsid \
  python examples/frozenlake/train_frozenlake_qwen3.py $COMMON --score_centering
DATA_DIR=$PWD/data/frozenlake PYTHONPATH=$PWD setsid \
  python examples/frozenlake/train_frozenlake_qwen3.py $COMMON
```

TensorBoard:

- Gemma: `gs://linchai-bucket-dev/tensorboard/grpo/sc_12grid_20260923_192138`
- Qwen SC on: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_on_20260923_201935`
- Qwen SC off: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_off_20260923_201935`

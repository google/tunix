# Score Centering: 12x12 FrozenLake and Qwen3 A/B report

Follow-up to [`score_centering_validation.md`](score_centering_validation.md).
That report established that the Score Centering (SC) implementation is
correct once the rollout transport is fixed. This one asks two further
questions:

1. Do the SC diagnostics stay healthy on a harder dataset with longer
   multi-turn trajectories?
2. Does SC measurably change training compared with an identical SC-off run?

**Run:** 2026-09-23 and 2026-09-24, single 8-chip TPU v6e host, vLLM 0.25.0
(tpu-inference 0.25.0) in server mode, JAX 0.10.2.

**Dataset:** FrozenLake regenerated with grid sizes 2x2 to 12x12 (previously
2x2 to 9x9), 10,000 train and 100 test prompts in `data/frozenlake/`.
`gs://tunix/data/Frozenlake` still holds the old 2x2 to 9x9 set, so these runs
read the local copy through `DATA_DIR`.

---

## 1. Verdict

- **The SC diagnostics are healthy on both models and do not drift.** Over 30
  steps on Gemma4-E2B and up to 120 steps on Qwen3-1.7B,
  `abs_coeff_sum_mean`, `head_mass_*` and `tail_ratio_rho` stay inside a
  narrow band and show no upward trend. The larger grids change none of the
  SC readings compared with the 9x9 validation run.
- **Over 15 steps, SC produced no measurable difference in training
  outcome.** On Qwen3-1.7B, paired SC-on and SC-off runs saw the same prompts
  at every step. Mean reward was 0.203 vs 0.197, and grad norm, entropy and
  every sampler/trainer agreement metric matched within noise.
- **Over 120 steps and three seeds, SC does not measurably change reward.**
  - The per-seed paired reward differences (SC on minus SC off) are +0.025,
    +0.029 and −0.016. The mean is +0.012 and the across-seed t-test gives
    p=0.48.
  - Each seed's two arms do diverge significantly, but the sign follows the
    run, not SC. The arm whose policy sharpens (entropy falls, completions
    shorten) is the arm that wins, and that was SC-on for seeds 42 and 43 and
    SC-off for seed 44.
  - The single-seed result first reported in §4.1 to §4.3 (SC-on +0.046 over
    steps 60 to 119) was a single-seed artifact. See §4.5.
- **Two differences are consistent across every run and attributable to SC.**
  - *The PPO ratio.* With SC on, `pg_clipfrac` is 0 by construction. With SC
    off it is 2% to 6%, because the recipe's gspo-token clip band (0.003 /
    0.005) is about as wide as the measured |log ratio| (~0.003).
  - *Early grad norm.* Over the first 30 steps, before the arms' trajectories
    diverge, grad norm is about 25% higher with SC on in all three seeds. The
    cause has not been identified.
- **This setup is close to the least favourable one for showing an SC
  effect.** The runs are strictly on-policy (`off_policy_steps=0`,
  `num_iterations=1`) with LR 1e-6, so the only mismatch source is bf16
  numerics. Any SC effect here therefore has to come from how SC reshapes the
  on-policy update, not from correcting policy lag.

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

- Trainer mesh `(1, 8)`, rollout mesh `(2, 4)`.
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
was lost when the run was stopped.

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

## 4. Qwen3-1.7B 120-step A/B (SC on vs SC off)

§3 ran too few steps for any outcome difference to show. This run extends each
arm to 120 steps.

> §4.1 to §4.4 describe the first pair (seed 42) as it was originally
> analysed. §4.5 adds seeds 43 and 44, which **overturn the seed-42
> conclusion on reward**. Read §4.5 before citing any number from
> §4.1 to §4.3.

Setup: same as §3, except for the following.

- `--num_batches 120 --num_epochs 1`: 120 distinct batches, no prompt is
  repeated. The recipe gained `--num_epochs`; the default is still 3.
- Env `max_steps=15` instead of 8, matching the Gemma recipe. Many 12x12
  grids cannot be solved in 8 moves.
- `DISABLE_TRAJECTORY_LOG=1`. The per-trajectory CSV logger on a `gs://` log
  dir rewrites the whole multi-GB file on every flush. The writes time out and
  abandoned upload threads pile up. An earlier attempt at this run accumulated
  ~780 GB of leaked `.tmp` objects and 2,200+ threads, and had ~5 min vLLM
  idle gaps per step by step 80. It was discarded.
- `seed=42` for both arms, so step *s* sees the same prompts in both arms.

Both arms ran to completion (rc=0) with all 120 rollout and actor points in
TensorBoard. Each arm took about 1.6 h, at 35 to 60 s per step.

### 4.1 Reward

Paired per-step difference (SC on minus SC off):

| steps | SC on | SC off | Δ | paired t / p | Wilcoxon p | on > off |
|---|---|---|---|---|---|---|
| 0 to 59 | 0.185 | 0.181 | +0.004 | 0.72 / 0.48 | 0.90 | 25 / 30 |
| 60 to 119 | 0.282 | 0.237 | **+0.046** | 4.22 / 9e-5 | 9e-5 | 43 / 12 |
| 90 to 119 | 0.324 | 0.248 | **+0.077** | 4.51 / 1e-4 | 2e-4 | 25 / 4 |
| all | 0.234 | 0.209 | +0.025 | 3.91 / 2e-4 | 6e-4 | 68 / 42 |

- The bootstrap 95% CI over all steps is [+0.013, +0.038].
- The per-step differences have lag-1 autocorrelation 0.23. Correcting for
  it (effective n ≈ 0.63·n) leaves p ≈ 0.002 to 0.003 for both the full run
  and the second half.
- Both arms improve, but SC-on improves about twice as fast: the linear reward
  slope is +0.16 vs +0.08 per 100 steps. The difference itself grows with step
  (p=1.6e-6).

### 4.2 Training dynamics (10-step block means)

| block start | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reward on | .165 | .164 | .184 | .155 | .216 | .223 | .284 | .198 | .238 | .309 | .328 | .337 |
| reward off | .166 | .174 | .166 | .159 | .219 | .200 | .257 | .203 | .218 | .269 | .277 | .197 |
| entropy on | .075 | .067 | .067 | .064 | .059 | .052 | .052 | .052 | .047 | .036 | .027 | .022 |
| entropy off | .074 | .070 | .074 | .070 | .072 | .066 | .059 | .062 | .069 | .079 | .105 | .109 |
| compl. len on | 267 | 289 | 281 | 268 | 221 | 211 | 191 | 197 | 179 | 167 | 158 | 156 |
| compl. len off | 266 | 266 | 265 | 274 | 239 | 219 | 209 | 218 | 197 | 274 | 369 | 406 |
| grad norm on | .39 | .40 | .47 | .30 | .38 | .31 | .43 | .47 | .34 | .44 | .51 | .38 |
| grad norm off | .29 | .31 | .29 | .26 | .38 | .29 | .38 | .22 | .26 | .23 | .19 | .16 |

- **SC on:** the policy sharpens monotonically. Entropy drops 3.4x,
  completions shorten by 40%, and reward rises alongside.
- **SC off:** the run tracks SC on until about step 90, then drifts. Entropy
  and completion length both climb (length about doubles), reward falls back
  to 0.197 in the last block, and grad norm decays.
- **Grad norm is about 30% higher with SC on from the first block**, before
  the runs diverge. The SC term enters the loss as `adv * logp_correction`
  with a coefficient mass of about 4e-3, which is too small to account for
  this on its own. Pinning the PPO ratio at 1 (no clipping, §3.3) is the other
  mechanism difference. SC-off clips 2.4% to 3.9% of tokens, which also seems
  too small to explain a 30% gap. **The cause has not been identified.**

### 4.3 SC diagnostics and sampler/trainer agreement

| metric | SC on | SC off |
|---|---|---|
| `abs_coeff_sum_mean` | 3.9e-3 to 5.1e-3 (block means), flat | — |
| `tail_ratio_rho_mean` / `head_mass_q_mean` | 0.980 / 1.0000 | — |
| `pg_clipfrac` | 0 | 2.4% → 3.9% (rising) |
| `log_ratio/abs_mean` | 0 | ~0.0033 |
| `mult_prob_error_mean`, median over steps | 1.012 | 1.014 |
| steps with a single-token `mult_err` > 1e3 / > 1e6 | 6 / 1 | 7 / 3 |
| `probs_pearson_corr` | 0.985 → 0.955 | 0.984 → 0.991 |
| `sampler_is/weight_mean` | 0.9997 | 0.9997 |

- `abs_coeff_sum_mean` does not track the entropy collapse (r = −0.01). The
  distribution-level sampler/trainer gap SC measures is unchanged even as the
  policy sharpens.
- The falling Pearson correlation in SC-on most likely reflects the sharper
  distribution: most sampled-token probabilities sit near 1, so a few
  low-probability tokens dominate the correlation. It is not evidence of worse
  alignment, but this has not been verified.
- Extreme single-token `mult_err` outliers (§2.2) occur at the same rate in
  both arms. They are unrelated to SC.

### 4.4 Caveats

- **One seed per arm.** The paired design controls which prompts each step
  sees, but the sampled rollouts, and therefore the policy trajectories,
  differ between arms. The tests above show that these two runs differ, not
  that SC caused the difference. Independent RL runs under identical settings
  can diverge in their second half, which is exactly where this difference
  appears. §4.5 confirms that this caveat was decisive.
- **Entropy collapse.** Entropy reached 0.022 at step 119 and was still
  falling. §4.5 shows this belongs to whichever arm sharpens, not to SC.
- `seq_logprob_error_threshold` and `off_policy_steps` were unused. The
  result says nothing about SC under real policy lag, which is the regime it
  was designed for.

### 4.5 Seed replicates (seeds 43, 44)

Both arms were rerun with `--seed 43` and `--seed 44`, with everything else as
in §4. The seed also sets the data shuffle, so prompts are paired within a
seed but differ across seeds. All six runs completed 120 steps with full
TensorBoard data.

**Reward, paired difference (SC on minus SC off):**

| seed | steps 0 to 59 | steps 60 to 119 | steps 90 to 119 | all | within-seed p (all) |
|---|---|---|---|---|---|
| 42 | +0.004 | +0.046 | +0.077 | **+0.025** | 2e-4 |
| 43 | +0.021 | +0.036 | +0.043 | **+0.029** | 7e-7 |
| 44 | −0.009 | −0.023 | −0.031 | **−0.016** | 0.011 (SC off wins) |
| mean over seeds | +0.005 | +0.020 | +0.030 | **+0.012** | across-seed t=0.86, **p=0.48** |

Each seed produces a significant within-seed difference, but its sign is not
stable across seeds. The within-seed tests measure run-to-run divergence, not
an SC effect. **With three seeds there is no evidence that SC changes reward in
this setup.**

**The winning arm is the one that sharpens.**

| seed | arm that sharpens | entropy, last 30 steps (sharpening vs other) | completions, last 30 steps | reward winner |
|---|---|---|---|---|
| 42 | SC on | 0.028 vs 0.098 | 160 vs 350 tokens | SC on |
| 43 | SC on | 0.028 vs 0.082 | 180 vs 344 tokens | SC on |
| 44 | **SC off** | 0.049 vs 0.084 | 138 vs 271 tokens | **SC off** |

In every seed, one arm's entropy falls and its completions shorten while the
other arm's entropy stays flat or rises. The sharpening arm always earns more
reward. Which arm sharpens differs by seed and does not depend on SC.

**The sampler/trainer agreement shifts follow sharpening, not SC.** In §4.3
the falling `probs_pearson_corr` and the lower `logp_diff_mean` under SC-on
were conjectured to come from sharpening. Seed 44 confirms this. There the
SC-*off* arm sharpens, and its Pearson drops (to 0.973) and its
`logp_diff_mean` falls (0.008 vs 0.010). Meanwhile the SC-on arm keeps
Pearson at 0.989.

**Effects that are attributable to SC (consistent in all three seeds):**

- `pg_clipfrac` = 0 with SC on, 2% to 6% with SC off (§3.3).
- **Early grad norm is about 25% higher with SC on.** Over steps 0 to 29, before
  the arms diverge, it was 0.42 vs 0.30 (seed 42), 0.41 vs 0.28 (seed 43)
  and 0.41 vs 0.32 (seed 44). The cause is still unidentified (§4.2).
- `abs_coeff_sum_mean` stays in 2.5e-3 to 6e-3. In seed 44 it drifts upward
  over the run (about 2.5e-3 to 6e-3), and the SC-on arm's
  `logp_diff_max` spikes to 10 to 20 between steps 80 and 110. Neither was
  seen in seed 42. Observed, not explained.

**Plots.** `plot_sc_ab.py` renders a 2x3 panel from TensorBoard: reward,
entropy, `abs_coeff_sum_mean`, `logp_diff` mean and max, and Pearson. It was
run for seed 42, the largest second-half gap, and for seed 44, where the gap
reverses.

**Revised conclusion.** In a strictly on-policy run (LR 1e-6, 120 steps),
SC's only reliable effects are mechanical: no PPO clipping, and larger early
gradients. Neither translates into a reward difference that survives seed
replication. Testing SC where it is designed to help needs real policy lag
(`off_policy_steps > 0` or `num_iterations > 1`) and at least 3 seeds per arm.

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
    environment variables;
  - adds `--num_epochs` (default 3);
  - raises env `max_steps` from 8 to 15;
  - adds the `DISABLE_TRAJECTORY_LOG=1` opt-out.

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

# Qwen3-1.7B 120-step A/B (§4). Runs to completion on its own; kill the
# process group after "[step 119]" if the interpreter hangs at exit.
COMMON120="--model_version Qwen/Qwen3-1.7B --batch_size 16 --mini_batch_size 16 \
  --num_generations 8 --num_batches 120 --num_epochs 1 --seed 42 \
  --disable_eval --no_exact_token_continuity --score_centering_top_k 32"
DISABLE_TRAJECTORY_LOG=1 DATA_DIR=$PWD/data/frozenlake PYTHONPATH=$PWD setsid \
  python examples/frozenlake/train_frozenlake_qwen3.py $COMMON120 --score_centering
DISABLE_TRAJECTORY_LOG=1 DATA_DIR=$PWD/data/frozenlake PYTHONPATH=$PWD setsid \
  python examples/frozenlake/train_frozenlake_qwen3.py $COMMON120
```

TensorBoard:

- Gemma: `gs://linchai-bucket-dev/tensorboard/grpo/sc_12grid_20260923_192138`
- Qwen SC on: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_on_20260923_201935`
- Qwen SC off: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_off_20260923_201935`
- Qwen 120-step SC on: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_on_120_20260923_232244`
- Qwen 120-step SC off: `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_off_120_20260923_232244`
- Qwen 120-step seed replicates (§4.5): `gs://linchai-bucket-dev/tensorboard/grpo/qwen1p7b_sc_{on,off}_120_s{43,44}_20260924_033023`
  (same command as above with `--seed 43` / `--seed 44`)

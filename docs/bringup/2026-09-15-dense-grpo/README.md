# Dense GRPO bring-up on Trellis — 2026-09-15

First end-to-end bring-up of the GRPO recipe gates (`seq-mask-TIS`, `grpo-loo`,
overlong loss masking) on the Trellis distributed orchestrator, on a single v5p-8.

Three results:

1. **It trains.** `grad_norm` is nonzero and the reward signal is live.
2. **Two silent wiring bugs**, both of which made the importance-sampling gate
   metrics unreadable. Both fixed on this branch.
3. **The trainer's own log-probs depend on its forward-pass micro-batch shape**
   — 6.2× more divergence at micro-batch 4 than at micro-batch 1, on identical
   data and identical weights.

Everything below is recomputed from the logs in [`logs/`](logs/) by
[`verify_report_numbers.py`](verify_report_numbers.py). Run it yourself:

```bash
python3 docs/bringup/2026-09-15-dense-grpo/verify_report_numbers.py
```

Claims are tagged **[M]** measured, **[D]** derived from measurements,
**[H]** hypothesis not yet tested.

---

## Setup

| | |
|---|---|
| hardware | one v5p-8, 4 chips: trainer on 0–1, rollout on 2–3, both `tp=2`, `fsdp=1` |
| model | `Qwen/Qwen3-1.7B`, **dense** |
| task | GSM8K, `REWARD_MODE=exact` |
| rollout engine | **tunix native (`vanilla`) sampler** — *not* vLLM |
| recipe | `seq-mask-TIS` band `[0.999, 1.002]`, `grpo-loo`, `epsilon_high=0.28`, `token-mean`, `beta=0`, `seq_logprob_error_threshold=2.0` |
| lengths | `max_prompt_length=512`, `max_response_length=768` |
| batch | 2 prompt groups × 8 generations = 16 trajectories per step |

The rollout node logs itself as `vllm-rollout-0` and prints "Serving vLLM
rollout worker" — **these strings are naming artifacts.** With `SAMPLER=vanilla`
the node prints `Creating native sampler on the rollout mesh` and runs tunix's
own JAX sampler. **vLLM is not in this loop.** Both sides of every log-ratio
below are the same JAX model under the same framework. [M]

Nothing outside this repo was modified: MaxText on the test machine is stock
upstream (`79e5978de`) with a clean working tree, and there is no vLLM checkout
involved. The branch touches 12 source files, all under `tunix/` and `tests/`.

### Reproducing

```bash
export TRELLIS_ROOT=/mnt/disks/maxdiffusion-data/trellis   # holds tunix/ and trellis_env/
cd $TRELLIS_ROOT/tunix
git checkout jfacevedo/trellis-mlperf

D=docs/bringup/2026-09-15-dense-grpo

# the training-signal run
RUN_ID=tis4 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4 bash $D/run_bringup.sh

# the batch-shape sweep (one optimizer step, so weights never move)
for T in 1 2 4; do
  RUN_ID=mb$T MAX_STEPS=1 TRAIN_MICRO_BATCH_SIZE=$T bash $D/run_bringup.sh
done

python3 $D/verify_report_numbers.py --logs $TRELLIS_ROOT   # or against logs/
```

Runs are deterministic: `mb1` and `tok1` are independent executions of the same
config and agree to the last printed digit (`absmean` 0.011337, signed
−0.002188), as do `mb4`/`tok4` (0.070474, +0.000299). [M]

> One operational trap: the orchestrator silently resumes from
> `CHECKPOINT_ROOT_DIRECTORY` and runs zero steps if that directory already has
> a checkpoint. The only evidence is a `Resuming from checkpoint: step=N` line.
> `run_bringup.sh` clears it per run.

---

## 1. The dense path trains

From [`logs/tis4/orchestrator.log`](logs/tis4/orchestrator.log), 6 steps: [M]

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `grad_norm` | N/A | 0 | **0.5863** | **1.154** | 0 | 0 |
| `loss` | N/A | 0.0000 | −0.0568 | 0.0406 | 0.0000 | 0.0000 |
| `reward_mean` | 0.5000 | 0.4688 | 0.2188 | 0.5000 | 0.5000 | 0.3438 |
| `advantage_mean` | 0.0000 | 0.1341 | 0.1341 | 0.0000 | 0.0000 | 0.0239 |

**ELI5:** three things have to be true before you can say a policy is learning —
the scorer gives different scores to different answers, the algorithm can tell
which were better, and the weights actually move. `grad_norm` is "how far the
weights moved". It is nonzero on 2 of the 5 steps that report it.

Do not be surprised by the three zeros. `grpo-loo` centres advantages inside a
prompt group, so when all 8 generations for a prompt earn the same reward the
advantage is identically zero and there is no gradient — and `reward_mean` is
exactly 0.5000 on two of those steps, i.e. a degenerate group on a 2-prompt
batch. I have **not** proven that is the cause of each zero [H]; the correlation
is imperfect (step 3 has `advantage_mean` 0.0000 yet `grad_norm` 1.154, because
`advantage_mean` is near zero by construction and it is the *spread* that drives
the gradient). What is established is that the gradient path is live end to end.

Gate metrics over the same run's 24 loss calls: [M]

- `sample_mask/kept_frac` = 1.0 on every call — the second gate
  (`mult_prob_error` vs `seq_logprob_error_threshold`) passes everything
- `tis/is_oob_ratio` mean **0.8125**, taking values {0.5, 0.75, 1.0}
- `sampler_is/token_logdiff_mean` = **−0.000466** nats/token

---

## 2. Two silent wiring bugs

### A. The trainer node never received `--num_generations`

`run_trainer_node.py:453`:

```
gradient_accumulation_steps = mini_batch_size * num_generations / train_micro_batch_size
```

`--num_generations` defaults to **1**. `launcher.sh` passed it to the
orchestrator but not to the trainer node, so with the launcher's own defaults
this computed **2 instead of 16**.

The orchestrator still streams all `mini_batch_size × num_generations`
trajectories; only the optimizer cadence is wrong. **The optimizer fired 8× per
intended step: effective batch 1/8 of configured, LR schedule 8× too fast.**
Nothing raised. It only surfaced because setting `train_micro_batch_size > 1`
tripped the divisibility check.

Fixed in `2a938f50`.

> `run_trainer_node.py:405` (`_create_maxtext_trainer_factory`) computes the same
> quantity as `ceil(mini_batch_size / train_micro_batch_size)`, omitting
> `num_generations` entirely. That the code differs is **[M]**; that it is the
> same defect is **[H]**. Untouched — the MaxText backend is not exercised here.

### B. `--train_micro_batch_size` was parsed, validated, logged, then dropped

`run_gsm8k_dist_grpo.py` range-checked the flag and printed it in the startup
banner but never passed it to `GRPOAdapter`. `rl_program.py:188` reads it back
off the adapter to size the batch assembler's payloads, so it silently stayed at
the adapter default: **one sequence per loss call.**

This is what makes the gate look broken. With a single sequence per call:

- `tis/is_oob_ratio` **cannot** be anything but 0.0 or 1.0
- `sample_mask/kept_frac` likewise
- `sampler_is/seq_geomean_min == mean == max` by construction

Measured, before and after: [M]

| | `is_oob_ratio` values | `seq_geomean_min == max` on every call |
|---|---|---|
| [`logs/mb1`](logs/mb1/) (1 seq/call) | {0.0, 1.0} only | **yes** |
| [`logs/tis4`](logs/tis4/) (4 seq/call) | {0.5, 0.75, 1.0} | no |

**The generalisable lesson:** if you see `is_oob_ratio` pinned at 1.0 anywhere,
check `seq_geomean_min` against `seq_geomean_max` first. If they are equal, the
metric is reporting *one* sequence and "1.0" means "this sequence was out of
band" — not "every sequence is out of band".

Fixed in `61e9a588`. `grad_norm` was also only ever sent to the metrics logger,
so it was invisible without wandb attached; `aa5ee665` puts it on the console
line.

---

## 3. A high `is_oob_ratio` is not by itself evidence of a defect

`seq_geomean = exp(mean(log p_trainer − log q_sampler))` per sequence.
`is_oob_ratio` is the fraction of sequences outside `[0.999, 1.002]` — a band
only −0.001/+0.002 wide.

Across `tis4`'s 24 min/max pairs the across-sequence spread is **σ = 0.00519**
[D] (range-of-4 estimator, E[range] = 2.059σ). Zero-mean Gaussian noise at that
σ predicts **77.36% out of band** [D]. Measured: **81.25%** [M].

So most — not quite all — of an 81% OOB rate is noise against a band that is too
tight for short sequences. σ_seq shrinks roughly as 1/√T, and these completions
are **114–196 tokens** [M].

**Therefore `is_oob_ratio` is not comparable across sequence lengths.** The
length-invariant quantity is the signed per-token mean, `token_logdiff_mean`,
which here is −0.000466 and two-sided: 16/24 calls have a sequence above 1.002
and 22/24 have one below 0.999. [M] That is symmetric noise around zero, not a
systematic bias.

---

## 4. Trainer log-probs depend on the forward-pass batch shape

Sweep at `MAX_STEPS=1`, so the weights are the freshly loaded checkpoint and no
optimizer step has happened. Only `train_micro_batch_size` varies. [M]

| `train_micro_batch_size` | `token_logdiff_absmean` | signed mean |
|---:|---:|---:|
| 1 | **0.011337** | −0.002188 |
| 2 | **0.066967** | −0.003559 |
| 4 | **0.070474** | +0.000299 |

A **6.22×** step change from 1 → 2, then flat. Not a smooth scaling, so I ruled
out the alternatives:

- **Different rollouts?** No. Per-sequence scored-token counts at micro-batch 1
  were `196,114,115,166 | 124,117,158,123 | 173,151,146,183 | 176,158,158,182`,
  and the micro-batch-4 per-call means came back `147.75 | 130.5 | 163.25 |
  168.5` — **exactly** those four group means. Identical sequences, identical
  order. [M]
- **Padding leaking into the mask?** No. 114–196 scored tokens per sequence
  against a 1280-token padded buffer; the mask is tight at both sizes. [M]
- **Weights drifted?** No. Step 0, fresh checkpoint, no optimizer step. [M]
- **Sampler changed?** No. `q_sampler` is the rollout's own reported log-probs,
  fixed once the rollouts are fixed. [M]

**Conclusion [D]:** with `q_sampler` fixed and the sequences identical, the
change has to be in `log p_trainer`. The trainer's own log-probs shift by
~0.07 nats/token purely as a function of its micro-batch shape. Converting
mean-absolute to σ (E|x| = 0.7979σ for zero-mean Gaussian) and subtracting in
quadrature gives **σ ≈ 0.087 nats/token of batch-shape-induced noise**, against
0.014 at batch 1.

**What it does and does not mean.** The induced noise is *symmetric* — the
signed mean is +0.000299 at micro-batch 4. Symmetric noise does not bias the
policy gradient. But `is_oob_ratio` keys on **spread, not sign**, so this
inflates the gate's rejection rate without any real train/sample mismatch: at
micro-batch 4 with one optimizer step, `is_oob_ratio` is 1.0 on all four calls.
Anyone tuning the band, or reading the gate as a health signal, is partly
reading this.

**Mechanism [H], untested.** `selective_log_softmax` already casts to fp32 for
the logsumexp (`tunix/rl/common.py:220`), so the normalizer is not the exposure.
The final logits matmul runs in bf16, and XLA picks different tiling and
accumulation orders for batch 1 versus batch ≥ 2, which would produce this
signature. The test that settles it is to force that matmul to fp32 and see
whether the 6× gap collapses. The tunix model path has no such knob today.

---

## 5. Caveats

- One model (Qwen3-1.7B **dense**), one task (GSM8K), 4 chips, ≤ 6 steps.
  Nothing here is confirmed on MoE, at scale, or with vLLM as the rollout engine.
- Completions are 114–196 tokens. Short sequences exaggerate `is_oob_ratio`
  (§3) and may exaggerate or mask the batch-shape effect (§4).
- The σ figures in §3 and §4 use a range-of-n estimator and a Gaussian
  assumption. They are order-of-magnitude arguments, not precise variances.
- §4 deliberately quotes no noise prediction for `mb2`: the range-of-n constant
  differs for n=2 and the n=4 value does not apply.
- The MaxText-backend instance of bug A (§2) is untested.
- The three zero `grad_norm` steps in §1 are explained by a hypothesis, not a
  measurement.

---

## 6. Branch

`jfacevedo/trellis-mlperf`, 15 commits on `origin/e2e-head` @ `7d995511`.
**12 source files** touched, all under `tunix/` and `tests/`, plus this
directory. No MaxText, no vLLM, nothing outside this repo.
71 tests pass (`tests/experimental/orchestrator/mlperf_recipe_wiring_test.py`
and `tests/rl/algo_core_test.py`), including a regression test for each of the
two wiring bugs.

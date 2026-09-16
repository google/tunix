# Dense GRPO bring-up on Trellis — 2026-09-15

First end-to-end bring-up of the GRPO recipe gates (`seq-mask-TIS`, `grpo-loo`,
overlong loss masking) on the Trellis distributed orchestrator, on a single v5p-8.

Five results:

1. **It trains.** `grad_norm` is nonzero and the reward signal is live. The
   plumbing — rollout log-probs reaching the trainer, both recipe gates
   computing, metrics emitting — is validated end to end.
2. **Two silent wiring bugs**, both of which made the importance-sampling gate
   metrics unreadable. Both fixed on this branch.
3. **The trainer's own log-probs depend on its forward-pass micro-batch shape**
   — 6.2× more divergence at micro-batch 4 than at micro-batch 1, on identical
   data and identical weights.
4. **Root cause found: bf16 matmul rounding on the MXU**, whose tiling and
   accumulation order are keyed on matmul shape. Established by elimination —
   true fp32 arithmetic removes the batch-dependence entirely (~20,000×). §5.
5. **It is fixable, and the sequence dropping is entirely numerical.** With
   fp32 activations + 3-pass matmuls, acceptance goes 18.75% → 42.71%
   (trainer-side only) → 100% (both sides), on the real GRPO loop. §6 explains
   why the 100% is an ideal-case result that will not transfer intact to a vLLM
   sampler.

Every number here is traceable to committed evidence, in one of two ways.

**GRPO-loop numbers** (§1, §3, §4, §6) come from the run logs in
[`logs/`](logs/) and are recomputed by
[`verify_report_numbers.py`](verify_report_numbers.py) — stdlib only, no TPU:

```bash
python3 docs/bringup/2026-09-15-dense-grpo/verify_report_numbers.py
```

**Probe numbers** (§5: the precision/cost table and the scope ablation) come
from standalone `probe_batch_shape.py` runs, whose raw transcripts are committed
in [`probe_out/`](probe_out/) — one file per configuration, each headed by the
exact command that produced it. They are **not** derivable from `logs/`, because
the probe does not go through the orchestrator.

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

The precision A/B/C in §6. `HP_TRAINER_*` is read inside the trainer process and
re-exported under the name `models.py` looks for, so it does **not** reach the
rollout node; setting `TRAINER_ACT_DTYPE` / `JAX_DEFAULT_MATMUL_PRECISION`
directly hits both:

```bash
RUN_ID=hp0 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4 bash $D/run_bringup.sh

HP_TRAINER_ACT_DTYPE=float32 HP_TRAINER_PRECISION=high \
  RUN_ID=hp1 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4 bash $D/run_bringup.sh

TRAINER_ACT_DTYPE=float32 JAX_DEFAULT_MATMUL_PRECISION=high \
  RUN_ID=hp2 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4 bash $D/run_bringup.sh
```

The standalone probe behind §5 needs no orchestrator and runs in seconds:

```bash
python3 $D/probe_batch_shape.py --model_dir $MODEL_DIR --tp 2 \
  --test logps,same,jit,accuracy,bench [--model_dtype float32] \
  [--weights bf16|fp32] [--precision high|highest] [--hp_scope head|nohead|all]
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

The three zeros are accounted for. `grad_norm` reported at step N+1 tracks
`advantage_mean` at step N — nonzero exactly when the advantage was nonzero, on
**5 of 5** observed pairs [M]:

| `advantage_mean` @ N | 0.0000 | 0.1341 | 0.1341 | 0.0000 | 0.0000 |
|---|---|---|---|---|---|
| `reward_mean` @ N | 0.5000 | 0.4688 | 0.2188 | 0.5000 | 0.5000 |
| `grad_norm` @ N+1 | 0 | 0.5863 | 1.154 | 0 | 0 |

Every zero sits under a step whose `reward_mean` is exactly 0.5000 with
`advantage_mean` exactly 0.0000 — a degenerate group. `grpo-loo` centres
advantages inside a prompt group, so when all 8 generations for a prompt earn
the same reward the advantage is identically zero and there is no gradient to
take. On a 2-prompt batch that happens often. **The zeros are a property of the
reward signal on this tiny batch, not of the TIS gate** [D] — the gate's
rejection rate does not predict them (steps 3 and 4 have `is_oob_ratio` of
0.875 and 0.6875 respectively yet both give zero gradient, while step 2 at
0.8125 gives 1.154). Five points is few, but the correlation is exact.

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

So the OOB rate is fully accounted for by the observed spread of `seq_geomean`.
The tempting next step is to call that spread short-sequence noise and assume it
washes out at production lengths, because σ_seq should shrink as 1/√T. **That
assumption is wrong here, and it is worth being precise about why.**

At `train_micro_batch_size=1` there is exactly one sequence per loss call, so
`seq_geomean` is observed directly rather than estimated from a range. Over the
16 sequences of [`logs/tok1`](logs/tok1/): [M]

| | |
|---|---|
| observed σ_seq | **0.003628** |
| σ_seq predicted if per-token errors were independent | **0.001140** |
| ratio | **3.18×** |
| implied effective independent tokens | **≈ 15**, against 152.5 actual |

(The prediction is σ_token/√T with σ_token = `absmean`/0.7979 = 0.01421 and
T = 152.5 mean scored tokens.) Predicted OOB at the observed σ is 0.6822 against
0.6875 measured — the spread explains the rejection rate exactly. [D]

**The per-token errors are strongly correlated within a sequence.** Averaging
over a 152-token completion buys the noise reduction of about 15 independent
samples, not 152. So each sequence carries its own near-constant offset, and
longer completions will not dilute it the way the band's design assumes. Whether
the effective count grows with T is untested [H] — but the 1/√T argument cannot
be leaned on without measuring it.

There is also a small systematic component: mean `seq_geomean` over those 16
sequences is **0.997819**, not 1.0, i.e. the trainer assigns slightly *lower*
log-probs than the sampler on average (`token_logdiff_mean` = −0.002188). [M]

**`is_oob_ratio` is therefore not comparable across sequence lengths or batch
shapes**, and a low value should not be read as agreement. The length-invariant
quantity is the signed per-token mean, `token_logdiff_mean`: −0.000466 over
`tis4`, two-sided (16/24 calls have a sequence above 1.002, 22/24 have one below
0.999). [M]

### Calibration: what the reference actually accepts

A high rejection rate is normal for this gate. NVIDIA published per-step
acceptance curves for seq-mask-TIS at the same `[0.999, 1.002]` bounds
(mlcommons/training#905, "Standard-TIS controls: accepted samples", accepted
among `global_valid_seqs`):

| reference run | accepted | rejected |
|---|---:|---:|
| GBS256 · job 429453 | 55.8% | 44.2% |
| GBS512 · job 429454 | 28.1% | 71.9% |
| GBS1024 · job 429455 | 13.5% | 86.5% |

Two things follow. First, acceptance **degrades sharply with global batch size**
and declines over training steps within each run, fastest at the largest GBS.
Second, our numbers should be read against that range, not against 100%:

| | accepted | verdict |
|---|---:|---|
| reference span | 13.5 – 55.8% | — |
| Trellis micro-batch 1 | **31.25%** | inside the reference range |
| Trellis micro-batch 4 | **0%** | below the reference's worst |

So the gate rejecting most of the batch is not by itself a defect, and §3's
spread analysis explains our rate at micro-batch 1. What is not normal is
micro-batch ≥ 2 — see §4.

The reference's GBS trend and our micro-batch trend point the same way but are
**not the same mechanism** [D]: theirs is consistent with policy drift (a bigger
batch takes a more effective step, so the policy moves further from the sampler
that generated the rollouts), whereas ours is measured at step 0 with the
weights never updated, so drift cannot explain it. Whether the reference's
trend also carries a numerics component is untested [H].

[`reference_baseline.py`](reference_baseline.py) recomputes band statistics from
a NeMo-RL trace's `train_data_step*.jsonl` dumps. It is a tool, not the source
of the table above — its denominator does not necessarily match the
`global_valid_seqs` used in the published curves, and the trace available here
is a different job from the ones plotted.

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

Because out-of-band sequences are zeroed but stay in the loss denominator
(`truncated_importance_weights` leaves the denominator alone by design,
`algo_core.py`), a rejection rate driven by numerics is a direct loss of
gradient signal for no modelling reason.

**Do not read micro-batch 1 as the good configuration.** Its low
sampler-vs-trainer number is *correlated error*, not accuracy: it rounds the way
the sampler does. Against a genuine fp32 ground truth the bf16 path is 0.0247 at
bs=1, 0.0228 at bs=2 and 0.0220 at bs=4 — **bs=1 is marginally the worst.**
Keep `train_micro_batch_size: 1` for recipe fidelity with the reference, not for
numerical accuracy. §5 resolves the mechanism.

---

## 5. Root cause: bf16 matmul rounding, established by elimination

`probe_batch_shape.py` (committed here) reproduces the effect with no
orchestrator, rollout or GRPO — just the model, fixed tokens, and different
forward batch sizes through the real jitted `compute_per_token_logps`. [M]

Ruled out, each by measurement:

| hypothesis | test | result |
|---|---|---|
| tensor parallelism | rerun at `tp=1` | survives, 0.0276 |
| padding / ragged lengths / cross-row leakage | batch of N **identical** copies | still shifts; all rows agree with each other, none with bs=1 |
| log-prob chunking | `chunk_size` 128/256/768/2048 | no change at any size |
| `default_matmul_precision` alone | `high`, `highest` on the bf16 model | **bit-identical no-op** |

The last one is the clue. `jax.lax.Precision` only affects **f32** operands
(`jax/_src/lax/lax.py:2135`), and this path already has bf16 operands
everywhere, so the flag could never do anything. Widening the operands is what
matters:

| weights | acts | precision | fwd bs=1 | fwd bs=4 | \|Δ\| bs=4 vs bs=1 | `seq_geomean` shift |
|---|---|---|---|---|---|---|
| bf16 | bf16 | DEFAULT *(production)* | 13.7 ms | 44.6 ms | 0.03134 | up to **0.0107** |
| bf16 | fp32 | DEFAULT | 17.2 ms | 58.6 ms | 0.01132 | up to 0.0021 |
| **bf16** | **fp32** | **HIGH** | **24.7 ms** | **87.7 ms** | **0.0000299** | **≤ 6e-6** |
| bf16 | fp32 | HIGHEST | 30.9 ms | 111.1 ms | 0.0000014 | ≤ 1e-6 |

For completeness, **full fp32 (weights *and* activations) + HIGHEST** gives
`|Δ|` = 0.0000016 with the `seq_geomean` shift at exactly 1.0 — a ~20,000×
reduction ([`probe_out/13_full_fp32_highest.txt`](probe_out/13_full_fp32_highest.txt)).
That is the configuration that establishes the mechanism; it is not a
recommendation.

Note that **HIGHEST is genuinely more accurate than HIGH here** — 0.0000014 vs
0.0000299, about 21× — because the activations are truly fp32 and the third
bf16 component of the operand split is doing real work. HIGH is recommended not
because HIGHEST is useless but because HIGH already puts the shift four orders
of magnitude inside the band, at ~25% less cost.

Raw transcripts for every row are in [`probe_out/`](probe_out/), one file per
configuration. Timings are wall-clock and vary ~1% run to run.

**Mechanism: bf16 matmul rounding on the MXU**, whose tiling and accumulation
order are keyed on matmul shape — and batch size is part of that shape. True
fp32 arithmetic removes it entirely (~20,000-22,000× depending on weight dtype).

Two scoping results that matter for cost:

- **An fp32 output head alone does not work.** It buys 8% (0.03134 → 0.02870)
  and leaves the hidden-state divergence untouched at 1.50 max. The complement
  confirms where the variance lives: upgrading *everything except* the head
  drops it to 0.00986 and collapses the hidden-state divergence from 1.50 to
  0.0000439. The variance is born in the transformer body; the head only adds
  to it. [M, `probe_out/02_scope_head.txt`, `probe_out/03_scope_nohead.txt`]
- **fp32 *weights* are not needed** — bf16 weights with fp32 activations measured
  marginally *better* than fp32 weights (0.0000299 vs 0.0000344), so weight
  memory is unchanged. Only activations widen. [M]

On cost: for **f32 × bf16** operands the TPU pass counts are **1 / 2 / 3**, not
1 / 3 / 6, and `HIGHEST` buys zero accuracy over `HIGH` in that mixed case
(verified against `lax.py:2129-2175` and measured on this v5p). `HIGH` is the
right setting.

---

## 6. Does it fix `is_oob`? Yes — and it shows the dropping is entirely numerics

Real GRPO loop, identical config, 24 loss calls each. Precision is the only
variable. [M]

| run | configuration | `is_oob_ratio` | **accepted** | `token_logdiff_absmean` |
|---|---|---|---|---|
| [hp0](logs/hp0/) | bf16 both sides *(production)* | 0.8125 | **18.75%** | 0.067160 |
| [hp1](logs/hp1/) | fp32 acts + HIGH, **trainer only** | 0.5729 | **42.71%** | 0.008290 |
| [hp2](logs/hp2/) | fp32 acts + HIGH, **both sides** | **0.0000** | **100.00%** | **0.000008** |

All three still train (`grad_norm` peaks 0.5863 / 0.8952 / 1.431). Step time is
~10 s in every case — the ~2× trainer forward is hidden because rollout
generation dominates.

### Why hp2 reaches zero, and why that will not transfer intact

hp2 is the *ideal* case and should not be read as a production forecast:

- **Both sides are the same tunix JAX model.** hp2 makes two copies of one
  implementation agree. Production pairs **vLLM** against MaxText/tunix —
  different kernels, layouts and fused ops. Agreement across engines is a
  strictly harder problem that precision alone may not close.
- **`max_staleness=0`** [M, from the orchestrator config]. Sampler and trainer
  hold identical weights, so the run is exactly on-policy. The reference recipe
  is **async**, where rollouts can be scored against a newer policy.
- The agreement is 1.0e-5 – 1.6e-5, not literally 0, confirming the two sides
  are genuinely independent computations rather than the same array.

**The tell that separates the two effects: our rejection rate is flat across
steps (1.00, 0.69, 0.81, 0.88, 0.69, 0.81 — no trend), while the reference's
*rises* (GBS1024: 49% → 86.5% over 7 steps).** Flat means numerics; rising means
policy drift. Precision fixes the floor, not the drift. The reference's ~49%
step-1 rejection is plausibly its numerics floor — that is the part this finding
speaks to.

**hp1 is the portable result**: trainer-side precision alone, which is the half
we control when the sampler is vLLM.

---

## 7. Caveats

- One model (Qwen3-1.7B **dense**), one task (GSM8K), 4 chips, ≤ 6 steps.
  Nothing here is confirmed on MoE, at scale, or with vLLM as the rollout engine.
- Completions are 114–196 tokens. Short sequences exaggerate `is_oob_ratio`
  (§3) and may exaggerate or mask the batch-shape effect (§4).
- The σ figures in §3 and §4 use a range-of-n estimator and a Gaussian
  assumption. They are order-of-magnitude arguments, not precise variances.
- §4 deliberately quotes no noise prediction for `mb2`: the range-of-n constant
  differs for n=2 and the n=4 value does not apply.
- The MaxText-backend instance of bug A (§2) is untested.
- The `grad_norm`/`advantage_mean` correspondence in §1 is exact on all 5
  observed pairs, but 5 pairs is a small sample.
- Whether the effective independent-token count in §3 grows with sequence length
  is untested; it was measured at one length (152.5 mean tokens).
- The reference acceptance figures in §3 are NVIDIA's published curves for a
  different model, task and rollout engine at far longer sequences. They
  calibrate what rejection rate is normal for this gate; they are not a
  like-for-like comparison with a 1.7B dense model on GSM8K.
- **§6's hp2 (100% acceptance) is an ideal case**: one model implementation on
  both sides, `max_staleness=0`, 6 steps, one seed. It demonstrates the *cause*
  is numerics. It does not predict production, where the sampler is a different
  engine and the recipe is async.
- Completions here are 152 scored tokens against the reference's 65536.
  `seq_geomean` averages over T, so sequence length changes how per-token noise
  projects onto the gate.
- The ~2x trainer forward cost in §5 is measured on this model at these shapes,
  forward only. Backward was not measured, so no step-time delta is claimed.
- Precision was applied globally within each process. Scoping it to only the
  log-prob pass is possible (`precision=` is a per-op kwarg) but untested.

---

## 8. Branch

`jfacevedo/trellis-mlperf`, 15 commits on `origin/e2e-head` @ `7d995511`.
**12 source files** touched, all under `tunix/` and `tests/`, plus this
directory. No MaxText, no vLLM, nothing outside this repo.
71 tests pass (`tests/experimental/orchestrator/mlperf_recipe_wiring_test.py`
and `tests/rl/algo_core_test.py`), including a regression test for each of the
two wiring bugs.

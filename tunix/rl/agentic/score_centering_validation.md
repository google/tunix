# Score Centering: validation report and usage guidance

Companion to [`score_centering_walkthrough.md`](score_centering_walkthrough.md),
which describes the design. This document records an end-to-end validation of
that implementation on the agentic FrozenLake recipe, the defects it surfaced,
and how to configure Score Centering (SC) in practice.

**Validated:** 2026-09-22, `examples/frozenlake/train_frozenlake.py`,
Gemma4-E2B-it on a single 8-chip TPU host (rollout mesh `(2, 4)`, trainer mesh
`(8, 1)`), vLLM 0.25.0 in server mode, JAX 0.10.2.

---

## 1. Verdict

The math and the loss-side implementation are correct: all 9 tests in
`ScoreCenteringTest` pass, and once the rollout plumbing is repaired the
`score_centering/*` diagnostics behave exactly as the walkthrough predicts.

The **transport** between rollout and learner was broken, so SC never engaged on
the agentic path it was written for. Three defects, in dependency order.

### 1.1 `RLEngine.generate()` drops the top-k fields (blocking)

`tunix/rl/rl_cluster.py` — `RLEngine.generate()` rebuilds a fresh
`RolloutOutput` from the per-micro-batch outputs and copies only `text`,
`logits`, `tokens`, `left_padded_prompt_tokens`, `logprobs` and
`prompt_lengths`. `topk_token_ids` and `topk_logprobs` are silently dropped.

`VllmRollout.generate()` populates them correctly, and
`TrajectoryCollectEngine._one_step` reads them back off `rollout_output`
(`trajectory_collect_engine.py:917`) — but by then they are already `None`.
Downstream this means `has_topk_data=False` in `_process_results`,
`old_topk_token_ids=None` on the `TrainExample`, and therefore
`use_score_centering=False` at `algo_core.py:426`. SC is a no-op.

Note `routed_experts` is dropped by the same merge, so this is a general lossy
chokepoint rather than an SC-specific oversight. **Fix:** concatenate and
forward both fields the way `logprobs` is already handled.

### 1.2 Metric registration and metric emission disagree (crash)

`agentic_grpo_learner.py:357` registers `score_centering/*` in
`rl_metrics_to_log` based on `algo_config.score_centering` alone, but
`grpo_loss_fn` only writes those aux keys when `use_score_centering` is true —
which additionally requires the top-k data to be present. When the two
disagree, `RLTrainer._post_process_train_step` indexes `aux[metric_name]`
unconditionally (`tunix/rl/trainer.py:80`) and the run dies mid-training with:

```
KeyError: 'score_centering/head_mass_q_mean'
```

The crash site is far from the cause, which made 1.1 expensive to find. Either
gate the registration on the same condition the loss uses, or have the loss
emit zeros for the diagnostics when SC is configured but inert. Failing loudly
is right; failing *here* is not.

### 1.3 The learner's `num_logprobs` auto-config arrives too late (silent)

`agentic_grpo_learner.py:306` raises `train_rollout_config.num_logprobs` to
`score_centering_top_k`. But `RLEngine.__init__` calls `_init_engine()`
(`rl_cluster.py:153`), which constructs `VllmRollout` and snapshots
`num_logprobs` into both `VllmConfig` and vLLM's `max_logprobs` engine
argument. Because `GRPOLearner` is constructed *after* `RLEngine`, that
auto-config never reaches the engine — it is effectively dead code.

Consequence: `sampling_params.logprobs` stays at 1, `num_logprobs > 1` is
false, no top-k is extracted, and SC silently degrades to plain GRPO. **Callers
must set `num_logprobs` on the `RolloutConfig` before building `RLEngine`.**
Either make the learner fail loudly when it detects an already-built engine, or
move the auto-config into `ClusterConfig` validation.

---

## 2. Measurements

Six train steps, `batch_size=8`, `num_generations=4`, `top_k=32`,
`sampler_is="token"`, `sampler_is_threshold=2.0`, `num_iterations=1`,
`exact_token_continuity=False` (see §4.1).

| step | `abs_coeff_sum_mean` | `head_mass_q_mean` | `head_mass_p_mean` | `tail_ratio_rho_mean` |
|---|---|---|---|---|
| 1 | 4.916e-4 | 0.99996114 | 0.99996078 | 0.98824 |
| 2 | 4.951e-4 | 0.99995118 | 0.99995095 | 0.98900 |
| 3 | 3.302e-4 | 0.99996042 | 0.99996007 | 0.98876 |
| 4 | 4.688e-4 | 0.99995524 | 0.99995506 | 0.98799 |
| 5 | 3.806e-4 | 0.99995673 | 0.99995625 | 0.98963 |
| 6 | 3.948e-4 | 0.99995458 | 0.99995434 | 0.98814 |

Against an SC-off control run (identical flags, `--score_centering` removed):

| metric | SC on | SC off |
|---|---|---|
| `sampler_is/weight_mean` | ~1.0000 | ~1.0000 |
| `pg_clipfrac` | 0.0 | 0.0 |
| `entropy` | 0.142, 0.152, 0.131, 0.145 | 0.145, 0.150, 0.139, 0.138 |
| `ppo_kl` | exactly 0 | ~1e-5 |
| TB scalar tags | 75 | 71 |

Corresponding rollout-side drift on the SC run, per step, as
`(mean, max)`:

| step | `logp_diff` | `mult_err` | `pearson` |
|---|---|---|---|
| 1 | (0.0163, 0.926) | (1.018, 2.52) | 0.9976 |
| 2 | (0.0078, 2.989) | (1.011, 19.87) | 0.9952 |
| 3 | (0.0080, 6.137) | (1.047, **462.58**) | 0.9967 |
| 4 | (0.0077, 1.052) | (1.008, 2.86) | 0.9981 |
| 5 | (0.0108, 1.054) | (1.012, 2.87) | 0.9975 |
| 6 | (0.0108, 0.860) | (1.012, 2.36) | 0.9970 |

### Interpretation

- **`abs_coeff_sum_mean ~4e-4`** sits in the "bf16 inference is well aligned
  with the trainer, no policy lag" regime. SC's correction is real but light —
  consistent with `num_iterations=1` and `off_policy_steps=0`, where the only
  mismatch source is bf16 kernel divergence between vLLM and the trainer
  forward.
- **`head_mass ≈ 0.99996` at k=32.** The top-32 head already captures 99.996%
  of the sampler's probability mass, so the scaled-tail approximation is
  carrying almost nothing. Raising k to the default 128 would quadruple the
  rollout payload and the `[B, L, k]` trainer gather to recover ~4e-5 of extra
  mass. See §3.2.
- **`tail_ratio_rho ≈ 0.988`,** i.e. very close to 1, confirming the head
  masses under `q` and `p` agree; `rho` far from 1 is what would signal a real
  distribution split.
- **`ppo_kl` is exactly 0 with SC on.** This is the
  `use_score_centering and num_iterations == 1` branch at `algo_core.py:458`
  forcing `old_per_token_logps = stop_gradient(per_token_logps)`, pinning the
  PPO ratio at 1 so SC carries the whole off-policy correction. Expected — but
  it means `ppo_kl` stops being a usable drift signal in SC runs.
  `abs_coeff_sum_mean` is its replacement.
- **The step-3 `mult_err` max of 462** is a single-token importance ratio
  outlier three orders of magnitude past the clip threshold. Mean `mult_err`
  stays at 1.01 throughout, so these are rare, but they are precisely what the
  composed TIS + SC formulation exists to absorb.

---

## 3. Usage guidance

### 3.1 Minimum viable configuration

SC needs three things set consistently, and two of them are not automatic
(§1.3):

```python
rollout_config = base_rollout.RolloutConfig(
    ...,
    return_logprobs=True,   # required
    num_logprobs=32,        # must equal score_centering_top_k, set BEFORE RLEngine
)

rl_engine = rl_engine_lib.RLEngine(..., cluster_config=cluster_config)

grpo_config = GRPOConfig(
    ...,
    score_centering=True,
    score_centering_top_k=32,   # keep in sync with num_logprobs
    sampler_is="token",         # composed TIS + SC, recommended
    sampler_is_threshold=2.0,
)
```

`ROLLOUT_ENGINE` must be `"vllm"`. The vanilla and SGLang-JAX rollouts never
populate `topk_logprobs`, so SC degrades to plain GRPO without a warning — fail
loudly on this in your recipe rather than discovering it from a flat
`abs_coeff_sum_mean`.

### 3.2 Choosing `score_centering_top_k`

Let the measured `head_mass_q_mean` pick k, not the default:

- `head_mass_q_mean > 0.999` — k is already saturating. Lower it; you are
  paying linear cost in the rollout payload, the trajectory buffers, the packed
  `[L, k]` per-token fields and the trainer's `[B, L, k]` gather for nothing.
- `head_mass_q_mean` in `0.99`–`0.999` — k is reasonable.
- `head_mass_q_mean < 0.99` — the tail is doing real work and `rho` is carrying
  a large approximation. Raise k.

Head mass shrinks as sampling temperature rises and as the vocabulary flattens,
so re-check after any temperature change. On this recipe
(Gemma4-E2B, `temperature=0.7`) **k=32 is ample and the default of 128 is
roughly 4x more than needed.** Consider lowering the default, or at minimum
documenting that k should be tuned against observed head mass.

Memory cost is worth stating explicitly: the per-sequence top-k buffers are
`[max_response_length, k]` in both `int32` and `float32`, i.e. `2 * 4 * L * k`
bytes. At `L=2048, k=128` that is 2 MB per trajectory, so a
`batch_size=64 x num_generations=8` batch carries ~1 GB of top-k arrays before
anything reaches the device. At k=32 it is ~256 MB.

### 3.3 Turn off the rollout-side sampling filters

Keep `top_p=1.0` and `top_k=0` at rollout time. This already matters for
`sampler_is`, and it matters more for SC: the sampler's `processed_logprobs`
are `log_softmax` over the *filtered* logit set, so with filters active the
recorded `q_v` are normalized over a truncated support while the trainer's
`p_v` are normalized over the full vocabulary. The head masses then disagree
systematically, `rho` is biased, and `abs_coeff_sum_mean` reports drift that is
an artifact of the filter rather than a real distribution gap. Control
exploration with temperature instead.

### 3.4 Reading `abs_coeff_sum_mean`

This is the primary diagnostic: the mean over tokens of
`sum_{v in H} |q_v * w_v - alpha * p_v|`. It is **exactly 0.0 when p == q**,
for both the standalone and the composed TIS formulation (with
`f(r) = min(r, C)` and `C >= 1`, `w_v` and `alpha` both collapse to 1, leaving
`q_v - p_v = 0`). So any nonzero reading is genuine sampler-trainer mismatch,
not numerical floor.

- **`< 1e-4`** — bf16 inference is tightly aligned and there is no policy lag.
  SC is correcting very little. It is cheap insurance, but do not expect it to
  move your training curves; if you are paying for it, verify the cost is
  acceptable.
- **`1e-3` to `5e-2`** — bf16 operator divergence or policy staleness is
  producing a meaningful distribution shift, and SC is doing substantive
  gradient centering. This is the regime it was designed for.
- **`> 5e-2`** — investigate rather than trusting the correction. Likely causes
  in order: rollout sampling filters still active (§3.3), `off_policy_steps`
  set high, `num_iterations > 1` with many inner epochs, or a genuine
  weight-sync bug between trainer and rollout.

Watch it alongside `sampler_is/weight_mean` (should hover at 1.0) and
`head_mass_q_mean` (validates your k). If `abs_coeff_sum_mean` climbs over
training while head mass stays flat, the policy is drifting away from the
sampler — expected with off-policy steps, alarming without them.

### 3.5 Standalone vs composed

Prefer the composed form (`sampler_is="token"`, paper Eq. 11/14) for multi-turn
agentic rollouts, which is what this recipe uses. Standalone SC
(`sampler_is=None`, Eq. 8) leaves the raw importance ratio unclipped, and the
step-3 `mult_err` max of 462 in §2 shows what a single unclipped outlier can
look like on this workload.

### 3.6 Interaction with `num_iterations`

With `num_iterations=1` SC forces the PPO ratio to exactly 1 (§2). With
`num_iterations > 1` the learner instead routes `old_per_token_logps` through
the trainer's start-of-step recompute, so the ratio becomes meaningful again
and `ppo_kl` regains its usual interpretation. Be aware the diagnostic you rely
on changes between these two modes.

---

## 4. Pre-existing issues encountered

Neither is caused by Score Centering; both block the FrozenLake recipe and are
worth separate fixes.

### 4.1 Exact-token budget overshoot

```
ValueError: Exact trajectory exceeds training padding budget:
    prompt=1235/2048, completion=2049/2048
```

Raised at `agentic_grpo_learner.py:672` under `exact_token_continuity`. The
overshoot tracks whatever limit is configured and is always small — observed
2049/2048, 3076/3072, 2051/2048 — so this is a structural off-by-a-few in the
collect engine's clipping, not a budget that is merely too tight. Raising
`max_response_length` does not help; it only moves the boundary. Hit in the
eval path first, then in training.

Worked around here with `exact_token_continuity=False`, which is why the §2
numbers carry that caveat.

### 4.2 Shutdown hang

At interpreter exit the process wedges in `_python_exit` joining a thread while
rollout requests are still in flight, spraying:

```
RuntimeError: cannot schedule new futures after shutdown
```

from `VllmSampler._postprocess_as_completed`. The TPU chips stay held until the
process is `kill -9`'d and `/dev/vfio` is released, so back-to-back runs need a
wait loop on device availability.

### 4.3 Sizing note

`--max_response_length 3072` OOMs the train step (17.67 GB requested, 16.33 GB
free). The trainer mesh is `(fsdp=8, tp=1)`, so the vocab dimension of the
logits tensor is unsharded. Raising the response length requires dropping
`train_micro_batch_size` to 1.

---

## 5. Reproduction

```bash
git fetch origin pull/2386/head:pr-2386
git checkout -b sc-val pr-2386

# Loss-side correctness, ~23s, no TPU contention.
python -m pytest tests/rl/agentic/agentic_grpo_learner_test.py -k ScoreCentering -q

# End-to-end. Requires the RLEngine.generate() fix from §1.1 and the recipe
# flags below, none of which are part of PR #2386 as submitted.
PYTHONPATH=$PWD python examples/frozenlake/train_frozenlake.py \
    --batch_size 8 --mini_batch_size 8 --num_generations 4 --num_batches 2 \
    --disable_eval --no_exact_token_continuity \
    --score_centering --score_centering_top_k 32
```

`--score_centering`, `--score_centering_top_k`, `--score_centering_eps`,
`--disable_eval` and `--no_exact_token_continuity` were added to
`train_frozenlake.py` for this validation; the recipe ships with none of them.
`--score_centering` also sets `RolloutConfig.num_logprobs` up front, which is
what works around §1.3.

The control run is the same command without `--score_centering`. Metrics land
in TensorBoard under `actor/train/score_centering/*`.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algorithm core implementations for RL and Agentic RL learners."""

from collections.abc import Sequence
import functools
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry
from tunix.sft import utils as sft_utils

registry = function_registry.default_registry

# ==============================================================================
# Utils
# ==============================================================================


@registry.register("advantage_estimator", "gae")
@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    completion_mask: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
  """Compute advantages using Generalized Advantage Estimation (GAE).

  Computing GAE is a two-step process:

  First, compute the temporal difference (TF), `δ_t`, for each timestep `t`:

  ```
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
  ```

  Then, compute the GAE advantage, `A_t`, by summing the discounted TD
  residuals. It is calculated recursively, starting from the last timestep:

  ```
  A_t = δ_t + (γ * λ) * A_{t+1}
  ```

  where:

  - `A_t` is the GAE advantage at timestep `t`.
  - `δ_t` is the temporal difference at timestep `t`.
  - `γ` is the discount factor.
  - `λ` is the GAE lambda parameter.
  - `V(s_t)` is the value function at timestep `t`.
  - `r_t` is the reward at timestep `t`.

  Args:
    rewards: A 2D array of rewards for each step in the rollout.
    values: A 2D array of value estimates from the critic for each step.
    completion_mask: A 2D mask, which is 0 for padding tokens.
    gamma: The discount factor, `γ`.
    gae_lambda: The GAE lambda parameter, `λ`.

  Returns:
    A tuple of two 2D arrays - advantages and returns for each step.
  """
  batch_size = values.shape[0]

  def gae_step(state_t_plus_1, xs):
    # Unpack state and inputs.
    gae_t_plus_1, next_values = state_t_plus_1
    rewards_t, values_t, mask_t = xs

    # Compute Temporal Difference (TD).
    delta = rewards_t + gamma * next_values - values_t
    # Compute GAE for this time step.
    gae_t = delta + gamma * gae_lambda * gae_t_plus_1

    # Skip values on non-completion tokens.
    next_values = values_t * mask_t + (1 - mask_t) * next_values
    gae_t = gae_t * mask_t + (1 - mask_t) * gae_t_plus_1

    # New state to carry over comprises `gae_t` and `next_values`. Output for
    # this step is `gae_t`.
    return (gae_t, next_values), gae_t

  _, advantages_transposed = jax.lax.scan(
      gae_step,
      init=(jnp.zeros((batch_size,)), jnp.zeros((batch_size,))),
      xs=(
          jnp.transpose(jnp.array(rewards)),
          jnp.transpose(jnp.array(values)),
          jnp.transpose(jnp.array(completion_mask)),
      ),
      reverse=True,
  )
  advantages = jnp.transpose(advantages_transposed)
  returns = advantages + values

  # Normalise advantages.
  advantages = masked_whiten(advantages, completion_mask)
  return advantages, returns


@jax.jit
def masked_whiten(
    x: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
  """Normalize the input array."""
  x_mean = masked_mean(x, completion_mask)
  x_var = masked_var(
      x,
      completion_mask,
      x_mean,
  )
  x = (x - x_mean) * jax.lax.rsqrt(x_var + 1e-8)
  return x


@functools.partial(jax.jit, static_argnames=("axis",))
def masked_mean(
    x: jax.Array, mask: jax.Array, axis: int | None = None
) -> jax.Array:
  """Compute the mean of a masked array."""
  cast_mask = mask.astype(x.dtype)
  return jnp.sum(x * cast_mask, axis=axis) / (
      jnp.sum(cast_mask, axis=axis) + 1e-8
  )


@jax.jit
def masked_var(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array | None = None,
) -> jax.Array:
  """Compute the variance of a masked array."""
  cast_mask = mask.astype(x.dtype)
  if mean is None:
    mean = masked_mean(x, cast_mask)

  variance = masked_mean(jnp.square(x - mean), cast_mask)

  mask_sum = cast_mask.sum()
  bessel_corr = mask_sum / (mask_sum - 1)
  return variance * bessel_corr


# ==============================================================================
# Sampler-vs-trainer divergence
#
# The rollout engine that generates tokens and the trainer that scores them run
# the same weights but different kernels, so their log-probabilities differ
# slightly. The helpers below measure that divergence, drop sequences where it
# is too large to train on, and correct for what remains. Every one of them is
# inert unless explicitly configured.
# ==============================================================================


def sequence_geomean_ratio(
    log_is: jax.Array,
    completion_mask: jax.Array,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Per-sequence geometric mean of the sampler-to-trainer importance ratio.

  One number per sequence summarising how far the two engines disagree over it:
  1.0 is agreement, 1.01 means the trainer assigned the sequence about 1% more
  probability than the sampler did. Geometric rather than arithmetic because
  the per-token quantities are ratios that compose multiplicatively, which is
  the same as averaging in log space.

  Args:
    log_is: Per-token trainer-minus-sampler log ratio, `[B, T]`.
    completion_mask: Per-token mask over scored tokens, `[B, T]`.
    segment_ids: Packing segment ids, or None when each row is one sequence.
    num_segments: Static segment-bucket count; required with `segment_ids`.

  Returns:
    `(geomean, valid)` -- the per-sequence ratio and a 1.0/0.0 flag marking
    entries backed by at least one scored token.
  """
  if segment_ids is None:
    token_count = completion_mask.sum(axis=-1)
    log_mean = (log_is * completion_mask).sum(axis=-1) / (token_count + 1e-8)
  else:
    seg_sum = common.segmented_sum(
        log_is * completion_mask, segment_ids, num_segments
    )
    token_count = common.segmented_count(
        segment_ids, num_segments, mask=completion_mask
    )
    log_mean = seg_sum / (token_count + 1e-8)
  return jnp.exp(log_mean), (token_count > 0).astype(jnp.float32)


def sequence_mult_prob_error(
    log_is_raw: jax.Array,
    mask: jax.Array,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
) -> jax.Array:
  """Per-sequence multiplicative probability error, `mean_t exp(|log_is_t|)`.

  A scale-free measure of how far the sampler and trainer disagree: 1.0 is
  perfect agreement, 2.0 means the typical token's probability differs by a
  factor of two. Intended to catch plumbing or truncation faults rather than
  the ordinary numerical drift between two differently optimised engines.

  Takes the absolute value per token, so disagreements cannot cancel within a
  sequence. That is what separates it from an importance ratio and is the whole
  point: a sequence off by `+d` on half its tokens and `-d` on the other half
  has a geometric-mean ratio of exactly 1 and looks perfectly on-policy, while
  its multiplicative error is `exp(d)`.

  Takes the ratio **before** any `nan_to_num`. A scored token the sampler gave
  zero probability makes the error non-finite, which fails any threshold and
  drops the sequence. Sanitising first would turn that token into a log ratio of
  0, i.e. `exp(0) = 1`, and record the worst possible disagreement as perfect
  agreement.

  Args:
    log_is_raw: Per-token trainer-minus-sampler log ratio, `[B, T]`, before
      sanitising.
    mask: Tokens to include, `[B, T]`.
    segment_ids: Optional `[B, T]` packing segment IDs.
    num_segments: Optional static segment count; required when `segment_ids` is
      provided.

  Returns:
    `[B]` when `segment_ids` is None, or `[B, num_segments]` when `segment_ids`
    is provided, and 0.0 for fully masked sequences so they can neither drag a
    minimum statistic down nor trip a threshold spuriously.
  """
  # `where` rather than multiplying by the mask: an unmasked position may hold
  # an infinity, and `inf * 0` is NaN, which would spread a single padded slot
  # across the whole sequence's error.
  abs_log_is = jnp.where(mask > 0, jnp.abs(log_is_raw), 0.0)
  if segment_ids is None:
    denom = mask.sum(axis=-1)
    num = (jnp.exp(abs_log_is) * mask).sum(axis=-1)
  else:
    denom = common.segmented_sum(mask, segment_ids, num_segments)
    num = common.segmented_sum(
        jnp.exp(abs_log_is) * mask, segment_ids, num_segments
    )
  return jnp.where(denom > 0, num / jnp.clip(denom, 1.0, None), 0.0)


class SequenceMask(NamedTuple):
  """Result of `sequence_loss_mask`.

  Attributes:
    sample_mask: Per-sequence loss multiplier `[B]` (or `[B, num_segments]` when
      `segment_ids` is provided); 1.0 to keep.
    loss_mask: `completion_mask` restricted by `sample_mask`, `[B, T]`. What the
      loss and its denominator should aggregate over.
    mult_prob_error: Per-sequence multiplicative probability error `[B]` (or
      `[B, num_segments]` when `segment_ids` is provided), or None when no
      threshold was supplied. Reported for every sequence the gate saw, kept or
      dropped, so the distribution is visible before anyone tightens the
      threshold.
  """

  sample_mask: jax.Array
  loss_mask: jax.Array
  mult_prob_error: jax.Array | None


def sequence_loss_mask(
    completion_mask: jax.Array,
    overlong: jax.Array | None = None,
    mask_overlong: bool = False,
    log_is_raw: jax.Array | None = None,
    mult_prob_error_threshold: float | None = None,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
) -> SequenceMask:
  """Per-sequence loss multiplier, and the token mask it induces.

  Two reasons a sequence may not belong in the update at all, applied in order:

    truncated rollouts
      A prefix of a trajectory rather than a trajectory, so the reward attached
      to it is not the reward for the behaviour that produced it.
    log-probability disagreement
      Sampler and trainer disagree so much that treating them as the same
      policy is unsound. See `sequence_mult_prob_error`.

  Dropping a sequence is not the same as down-weighting it. A dropped sequence
  must leave the loss *denominator* too, so the survivors keep their full
  gradient magnitude and only the effective batch size shrinks. Returning a
  token mask rather than a weight is what achieves that: callers aggregate over
  `loss_mask`, so dropped tokens vanish from numerator and denominator alike.
  Contrast `truncated_importance_weights`, which multiplies per-token weights
  and deliberately leaves the denominator alone.

  Args:
    completion_mask: Per-token mask over scored tokens, `[B, T]`.
    overlong: 1.0 for sequences the rollout engine truncated, `[B]` (or `[B, T]`
      under sequence packing), or None when the engine reports no truncation
      verdict.
    mask_overlong: Whether to drop truncated sequences from the update.
    log_is_raw: Per-token trainer-minus-sampler log ratio `[B, T]` before
      sanitising, or None when the rollout engine returned no
      log-probabilities. The gate needs the unsanitised ratio; see
      `sequence_mult_prob_error`.
    mult_prob_error_threshold: Drop sequences whose multiplicative probability
      error exceeds this. None disables the gate.
    segment_ids: Optional `[B, T]` packing segment IDs.
    num_segments: Optional static segment count; required when `segment_ids` is
      provided.

  Returns:
    A `SequenceMask`. Every field takes its unrestricted value when no source
    is active, so callers can use them unconditionally.
  """
  batch = completion_mask.shape[0]
  if segment_ids is None:
    sample_mask = jnp.ones((batch,), dtype=jnp.float32)
    if mask_overlong and overlong is not None:
      sample_mask = sample_mask * (1.0 - jnp.astype(overlong, jnp.float32))

    mult_prob_error = None
    if mult_prob_error_threshold is not None and log_is_raw is not None:
      # Measured over the already-truncation-restricted mask. A sequence dropped
      # above contributes no tokens here, scores 0.0 and passes the threshold,
      # and stays dropped either way -- so ordering does not change the outcome,
      # but it keeps the reported error free of tokens nobody is training on.
      mult_prob_error = sequence_mult_prob_error(
          log_is_raw, completion_mask * sample_mask[:, None]
      )
      sample_mask = sample_mask * jnp.astype(
          mult_prob_error <= mult_prob_error_threshold, jnp.float32
      )

    return SequenceMask(
        sample_mask=sample_mask,
        loss_mask=completion_mask * sample_mask[:, None],
        mult_prob_error=mult_prob_error,
    )
  else:
    sample_mask = jnp.ones((batch, num_segments), dtype=jnp.float32)
    if mask_overlong and overlong is not None:
      overlong_arr = jnp.asarray(overlong, dtype=jnp.float32)
      if overlong_arr.shape == completion_mask.shape:
        seg_overlong = (
            common.segmented_sum(
                overlong_arr * (segment_ids > 0),
                segment_ids,
                num_segments,
            )
            > 0
        ).astype(jnp.float32)
      elif overlong_arr.ndim == 2 and overlong_arr.shape == (batch, num_segments):
        seg_overlong = overlong_arr
      else:
        seg_overlong = jnp.broadcast_to(
            overlong_arr.reshape((batch, 1)), (batch, num_segments)
        )
      sample_mask = sample_mask * (1.0 - seg_overlong)

    token_sample_mask = jnp.take_along_axis(
        sample_mask, segment_ids.astype(jnp.int32), axis=1
    )

    mult_prob_error = None
    if mult_prob_error_threshold is not None and log_is_raw is not None:
      mult_prob_error = sequence_mult_prob_error(
          log_is_raw,
          completion_mask * token_sample_mask,
          segment_ids=segment_ids,
          num_segments=num_segments,
      )
      sample_mask = sample_mask * jnp.astype(
          mult_prob_error <= mult_prob_error_threshold, jnp.float32
      )
      token_sample_mask = jnp.take_along_axis(
          sample_mask, segment_ids.astype(jnp.int32), axis=1
      )

    return SequenceMask(
        sample_mask=sample_mask,
        loss_mask=completion_mask * token_sample_mask,
        mult_prob_error=mult_prob_error,
    )


def token_outlier_stats(
    log_is: jax.Array,
    completion_mask: jax.Array,
    n_seq: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Extreme-value statistics of the sampler-trainer per-token disagreement.

  A mean cannot tell "every token disagrees slightly" apart from "a few tokens
  disagree categorically", and the two have different causes: ordinary
  numerical drift against an alignment or tokenisation fault. A handful of
  tokens off by tens of nats can dominate a sequence's geometric mean while
  leaving `token_logdiff_absmean` looking unremarkable.

  Args:
    log_is: Per-token trainer-minus-sampler log ratio, `[B, T]`.
    completion_mask: Per-token mask over scored tokens, `[B, T]`.
    n_seq: Sequence count to divide the per-sequence count by; already floored
      away from zero by the caller.

  Returns:
    `(absmax, outlier_frac, outliers_per_seq)` -- the largest single
    disagreement in nats, the share of scored tokens beyond
    `TOKEN_LOGDIFF_OUTLIER_NATS`, and the mean number of such tokens per
    sequence. The last is the useful one when a defect fires once per
    conversation turn, since the count is then meaningful on its own.
  """
  abs_log_is = jnp.abs(log_is)
  scored = completion_mask > 0
  absmax = jnp.where(
      scored.any(), jnp.max(jnp.where(scored, abs_log_is, 0.0)), 0.0
  )
  outliers = (abs_log_is > TOKEN_LOGDIFF_OUTLIER_NATS) * completion_mask
  outlier_frac = outliers.sum() / jnp.maximum(completion_mask.sum(), 1.0)
  return absmax, outlier_frac, outliers.sum() / n_seq


# Magnitude bins for the per-token log-ratio. Chosen so that the reported
# per-bin *mass* sums to `token_logdiff_mean` exactly, which is what turns
# "the centre is -0.16 nats" into "and here is which tokens paid for it".
LOG_IS_BIN_EDGES = (
    -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.05,
    0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0,
)
# Relative position within each sequence's own scored tokens, not absolute
# position: completions are right-padded to `max_response_length` and their real
# lengths differ by an order of magnitude, so absolute buckets would put
# everything in bucket 0.
N_LOG_IS_POS_BUCKETS = 8
# Sampler log-probability thresholds, ascending. Separates "the trainer
# disagrees about tokens the sampler was certain of" from "about tokens it was
# guessing at" -- different causes, and the two are indistinguishable in a mean.
LOG_IS_CONF_EDGES = (-1.0, -0.1, -0.01)


def _log_is_edge_label(value: float) -> str:
  text = f"{abs(value):g}".replace(".", "p")
  return f"m{text}" if value < 0 else text


def log_is_bin_labels() -> list[str]:
  """Metric-name-safe labels for `LOG_IS_BIN_EDGES`, low to high."""
  edges = LOG_IS_BIN_EDGES
  labels = [f"lt_{_log_is_edge_label(edges[0])}"]
  labels += [
      f"{_log_is_edge_label(lo)}_{_log_is_edge_label(hi)}"
      for lo, hi in zip(edges, edges[1:])
  ]
  labels.append(f"ge_{_log_is_edge_label(edges[-1])}")
  return labels


def log_is_attribution_metric_names() -> list[str]:
  """Every key `log_is_attribution` emits.

  The learner registers metric aggregators up front and raises on a registered
  key that is missing from `aux`, so the two lists have to be generated from
  the same place.
  """
  names = [
      "sampler_is/token_logdiff_mean",
      "sampler_is/scored_tokens_per_seq",
  ]
  for label in log_is_bin_labels():
    names.append(f"sampler_is/hist_frac/{label}")
    names.append(f"sampler_is/hist_mass/{label}")
  for i in range(N_LOG_IS_POS_BUCKETS):
    names.append(f"sampler_is/pos_signed/{i}")
    names.append(f"sampler_is/pos_absmean/{i}")
  for i in range(len(LOG_IS_CONF_EDGES) + 1):
    names.append(f"sampler_is/conf_signed/{i}")
    names.append(f"sampler_is/conf_frac/{i}")
  return names


def log_is_attribution(
    log_is: jax.Array | None,
    completion_mask: jax.Array,
    rollout_logps: jax.Array | None,
) -> dict[str, jax.Array]:
  """Attribute the per-token sampler-trainer disagreement to its sources.

  `seq_geomean` says the two engines disagree and `token_logdiff_absmean` says
  by how much on average; neither says which tokens account for it. Three
  decompositions do, all of them masked reductions over arrays the loss already
  holds:

  magnitude
    Per-bin share of scored tokens (`hist_frac/*`) and per-bin contribution to
    the mean (`hist_mass/*`). The masses sum to `token_logdiff_mean`, so a
    dashboard can attribute that mean to the bins that produced it. A sparse
    catastrophic tail and a diffuse bias are identical in the mean and obvious
    here.

  position
    Mean signed and mean absolute ratio in eight buckets of *relative* position
    within each sequence's scored tokens. Flat means a per-token cause; rising
    means something that accumulates along the sequence -- a recurrent state, a
    KV-cache effect, a drifting position index.

  sampler confidence
    The same, bucketed by the sampler's own log-probability for the token. A
    defect in the shared forward math shows up everywhere; one that only bites
    where the distribution is sharp shows up in the high-confidence buckets.

  `token_logdiff_mean` is signed, unlike the absolute-valued `*_absmean`
  metrics: the keep-band gates on the direction of the disagreement, not only
  its size.

  Args:
    log_is: Per-token trainer-minus-sampler log ratio, `[B, T]`, or None when
      the rollout engine returned no log-probabilities.
    completion_mask: Per-token mask over scored tokens, `[B, T]`.
    rollout_logps: The sampler's own per-token log-probabilities, `[B, T]`, or
      None. Only the confidence buckets need it; they report zeros without it.

  Returns:
    A dict keyed exactly by `log_is_attribution_metric_names()`.
  """
  names = log_is_attribution_metric_names()
  if log_is is None:
    return {name: jnp.float32(0.0) for name in names}

  mask = completion_mask.astype(jnp.float32)
  denom = jnp.maximum(mask.sum(), 1.0)
  out = {"sampler_is/token_logdiff_mean": (log_is * mask).sum() / denom}
  # The denominator every other metric here divides by. A mask that is too wide
  # or too narrow rescales all of them identically and is otherwise invisible,
  # and the count also shows whether two runs scored the same rollouts.
  out["sampler_is/scored_tokens_per_seq"] = mask.sum() / jnp.maximum(
      (mask.sum(axis=-1) > 0).sum(), 1.0
  )

  edges = jnp.asarray(LOG_IS_BIN_EDGES, dtype=jnp.float32)
  bin_idx = jnp.searchsorted(edges, log_is, side="right")
  for i, label in enumerate(log_is_bin_labels()):
    sel = mask * (bin_idx == i)
    out[f"sampler_is/hist_frac/{label}"] = sel.sum() / denom
    out[f"sampler_is/hist_mass/{label}"] = (log_is * sel).sum() / denom

  # (cumsum - 1) / total puts the first scored token of every sequence at 0.0
  # and the last just under 1.0, independently of how long the sequence is.
  cum = jnp.cumsum(mask, axis=-1)
  total = jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
  rel = (cum - 1.0) / total
  pos_idx = jnp.clip(
      (rel * N_LOG_IS_POS_BUCKETS).astype(jnp.int32),
      0,
      N_LOG_IS_POS_BUCKETS - 1,
  )
  for i in range(N_LOG_IS_POS_BUCKETS):
    sel = mask * (pos_idx == i)
    sel_denom = jnp.maximum(sel.sum(), 1.0)
    out[f"sampler_is/pos_signed/{i}"] = (log_is * sel).sum() / sel_denom
    out[f"sampler_is/pos_absmean/{i}"] = (jnp.abs(log_is) * sel).sum() / sel_denom

  n_conf = len(LOG_IS_CONF_EDGES) + 1
  if rollout_logps is None:
    for i in range(n_conf):
      out[f"sampler_is/conf_signed/{i}"] = jnp.float32(0.0)
      out[f"sampler_is/conf_frac/{i}"] = jnp.float32(0.0)
  else:
    conf_edges = jnp.asarray(LOG_IS_CONF_EDGES, dtype=jnp.float32)
    conf_idx = jnp.searchsorted(
        conf_edges, jnp.astype(rollout_logps, jnp.float32), side="right"
    )
    for i in range(n_conf):
      sel = mask * (conf_idx == i)
      sel_denom = jnp.maximum(sel.sum(), 1.0)
      out[f"sampler_is/conf_signed/{i}"] = (log_is * sel).sum() / sel_denom
      out[f"sampler_is/conf_frac/{i}"] = sel.sum() / denom
  return out


def truncated_importance_weights(
    log_is_raw: jax.Array,
    seq_geomean: jax.Array,
    seq_valid: jax.Array,
    sample_mask: jax.Array,
    band_min: float,
    band_max: float,
    segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Sequence-masked truncated importance sampling (`seq-mask-tis`) weights.

  Corrects for the residual mismatch between the rollout sampler -- the
  behaviour policy that actually produced the tokens -- and the trainer, the
  target policy being updated. Each token is weighted by its raw importance
  ratio `exp(log pi_trainer - log pi_sampler)`, and any sequence whose
  geometric-mean ratio falls outside `[band_min, band_max]` has its weights
  zeroed entirely. Sequences inside the band keep raw, unclipped weights.

  Two details carry the semantics:

  Non-finite log ratios (`±inf`, `NaN`) are gated to a weight of `0.0` via
  `jnp.isfinite(log_is_raw)` while finite log ratios are clamped to
  `[-20.0, 20.0]` before `jnp.exp()` and detached with `stop_gradient`,
  discarding zero-probability sampler tokens and keeping squared gradients
  within `float32` range.

  The keep-mask multiplies the weights and never the loss mask, so a rejected
  sequence stays in the loss denominator and the loss scales down with the
  rejection rate. This is the opposite of `sequence_loss_mask`, which removes a
  sequence from the denominator too: a high out-of-band ratio here is an
  effective learning-rate change rather than a smaller batch.

  Args:
    log_is_raw: Per-token trainer-minus-sampler log ratio **before** any
      `nan_to_num`, `[B, T]`.
    seq_geomean: Per-sequence geometric-mean ratio `[B]` (or `[B,
      num_segments]`), from `sequence_geomean_ratio`.
    seq_valid: 1.0 for entries backed by at least one scored token, `[B]` (or
      `[B, num_segments]`).
    sample_mask: Per-sequence validity `[B]` (or `[B, num_segments]`), used
      only to normalise the reported out-of-band ratio over sequences that are
      actually training.
    band_min: Lower end of the keep-band, on the geometric-mean ratio.
    band_max: Upper end of the keep-band.
    segment_ids: Optional `[B, T]` packing segment IDs.

  Returns:
    `(weights, oob_ratio)` -- per-token weights `[B, T]` to multiply into the
    per-token loss before aggregation, and the fraction of valid sequences the
    band rejected.
  """
  keep = jnp.astype(
      (seq_geomean >= band_min) & (seq_geomean <= band_max), jnp.float32
  ) * (seq_valid > 0).astype(jnp.float32)
  counted = sample_mask * seq_valid
  # 0.0 rather than 1.0 when nothing is counted: the band rejected nothing, and
  # a 1.0 here would pool across micro-batches as if it had rejected everything.
  oob_ratio = jnp.where(
      counted.sum() > 0,
      1.0 - (keep * counted).sum() / jnp.maximum(counted.sum(), 1.0),
      0.0,
  )
  if segment_ids is None:
    token_keep = keep[:, None]
  else:
    token_keep = jnp.take_along_axis(
        keep, segment_ids.astype(jnp.int32), axis=1
    )
  safe_log_is = jnp.where(jnp.isfinite(log_is_raw), log_is_raw, 0.0)
  weights = jnp.where(
      jnp.isfinite(log_is_raw),
      jnp.exp(jnp.clip(safe_log_is, -20.0, 20.0)),
      0.0,
  )
  return jax.lax.stop_gradient(weights * token_keep), oob_ratio


# |log p_trainer - log q_sampler| above which a token counts as an outlier in
# the reported diagnostic. Set well clear of ordinary numerical disagreement, so
# a non-zero count means tokens the two engines disagree about categorically
# rather than imprecisely. Reporting only; nothing keys off this value.
TOKEN_LOGDIFF_OUTLIER_NATS = 10.0

# Metric suffixes emitted per (length bucket, completion status). Raw sums, so
# that pooling across micro-batches, shards and steps is exact -- and so that
# the two statuses add back to the bucket total, losing nothing to the split.
SAMPLER_IS_LENGTH_BUCKET_METRICS = (
    "count",
    "len_sum",
    "logmean_sum",
    "logmean_sq_sum",
    "iid_var_sum",
)

# Buckets are split by truncation status because truncated sequences all pile
# up at the response-length cap; without the split the longest bucket is really
# "the truncated ones" and no length trend can be read from it.
SAMPLER_IS_COMPLETION_STATUSES = ("complete", "truncated")


def sampler_is_length_bucket_names(
    bucket_edges: Sequence[int] | None,
) -> list[str]:
  """Bucket names for the length-scaling diagnostic.

  Args:
    bucket_edges: Inclusive upper bounds on completion length, in tokens,
      strictly increasing; one open-ended bucket is appended. None or empty
      disables the diagnostic.

  Returns:
    One name per bucket, e.g. (256, 1024) -> ["le256", "le1024", "gt1024"].
  """
  if not bucket_edges:
    return []
  return [f"le{e}" for e in bucket_edges] + [f"gt{bucket_edges[-1]}"]


def sampler_is_length_bucket_metric_names(
    bucket_edges: Sequence[int] | None,
) -> list[str]:
  """Every metric-name suffix the length-scaling diagnostic emits.

  Args:
    bucket_edges: As `sampler_is_length_bucket_names`.

  Returns:
    Suffixes of the form `<bucket>/<status>/<metric>`, for the learner to
    register and the loss to prefix.
  """
  return [
      f"{bucket}/{status}/{metric}"
      for bucket in sampler_is_length_bucket_names(bucket_edges)
      for status in SAMPLER_IS_COMPLETION_STATUSES
      for metric in SAMPLER_IS_LENGTH_BUCKET_METRICS
  ]


def sampler_is_length_bucket_sums(
    log_is: jax.Array,
    seq_log_mean: jax.Array,
    completion_mask: jax.Array,
    seq_valid: jax.Array,
    bucket_edges: Sequence[int],
    overlong: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Length-bucketed sums for the offset's scaling behaviour.

  Distinguishes two explanations for a per-sequence sampler-vs-trainer offset
  that have opposite consequences at production sequence lengths:

    iid token noise
      The per-sequence mean is an average of T independent terms, so its spread
      falls as 1/sqrt(T). Longer sequences fix the offset on their own.
    systematic within-sequence bias
      Every token in a sequence leans the same way, nothing cancels, and the
      spread is flat in T. Longer sequences do not help, and a fixed keep-band
      stays unreachable at any scale.

  Both are measurable inside a single batch, by bucketing the sequences already
  present by their own length. Raw sums are returned rather than ratios so that
  pooling across micro-batches and shards is exact -- a plain sum, not an
  average of averages. Per bucket, offline:

      mean         = logmean_sum / count
      observed_var = logmean_sq_sum / count - mean**2  # spread actually seen
      iid_var      = iid_var_sum / count               # spread iid noise allows
      excess       = sqrt(observed_var / iid_var)

  `observed_var` is centred on the bucket's own mean, so a constant offset
  shared by every sequence does not read as length-dependent spread.

  `excess ~= 1` in every bucket means iid, and the offset shrinks like
  1/sqrt(T). `excess` rising with bucket length means systematic, and it does
  not.

  `iid_var_sum` is built from the *within-sequence* scatter of the per-token
  log ratio about that sequence's own mean, which is the correct null model: it
  is what the per-sequence spread would be if those same tokens were
  independent. Using it rather than a batch-wide noise estimate keeps the
  comparison valid even when per-token scatter itself varies with length.

  Args:
    log_is: Per-token trainer-minus-sampler log ratio, `[B, T]`.
    seq_log_mean: Its per-sequence masked mean, `[B]`.
    completion_mask: Per-token mask over scored tokens, `[B, T]`.
    seq_valid: 1.0 for rows holding a scored sequence, `[B]`.
    bucket_edges: Inclusive upper bounds on completion length, in tokens.
    overlong: 1.0 for truncated sequences `[B]`. When None every sequence is
      reported as `complete` and the `truncated` series is empty, which is how
      a missing verdict announces itself.

  Returns:
    Metric name (`<bucket>/<status>/<metric>`) to scalar sum.
  """
  seq_len = completion_mask.sum(axis=-1)
  seq_len_safe = jnp.maximum(seq_len, 1.0)
  centered = (log_is - seq_log_mean[:, None]) * completion_mask
  within_var = (centered**2).sum(axis=-1) / seq_len_safe

  truncated = (
      jnp.zeros_like(seq_valid)
      if overlong is None
      else jnp.astype(overlong, jnp.float32)
  )
  by_status = {"complete": 1.0 - truncated, "truncated": truncated}

  sums = {}
  lowers = (0,) + tuple(bucket_edges)
  uppers = tuple(bucket_edges) + (None,)
  for name, lower, upper in zip(
      sampler_is_length_bucket_names(bucket_edges), lowers, uppers
  ):
    in_bucket = seq_len > lower
    if upper is not None:
      in_bucket = in_bucket & (seq_len <= upper)
    in_bucket = in_bucket.astype(jnp.float32) * seq_valid
    for status, status_mask in by_status.items():
      m = in_bucket * status_mask
      prefix = f"{name}/{status}"
      sums[f"{prefix}/count"] = m.sum()
      sums[f"{prefix}/len_sum"] = (m * seq_len).sum()
      sums[f"{prefix}/logmean_sum"] = (m * seq_log_mean).sum()
      sums[f"{prefix}/logmean_sq_sum"] = (m * seq_log_mean**2).sum()
      sums[f"{prefix}/iid_var_sum"] = (m * within_var / seq_len_safe).sum()
  return sums


# ==============================================================================
# PPO Core
# ==============================================================================


@function_registry.register_policy_loss_fn("ppo")
def ppo_policy_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
) -> sft_utils.LossOutput:
  """PPO policy loss function."""
  epsilon_low = algo_config.epsilon_low
  epsilon_high = algo_config.epsilon_high
  entropy_coef = algo_config.entropy_coef

  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask

  return_entropy = entropy_coef is not None and entropy_coef != 0.0
  graphdef, state = nnx.split(model)
  outputs = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_entropy=return_entropy,
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
  )
  if return_entropy:
    per_token_logps, token_entropy = outputs
  else:
    per_token_logps = outputs


  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = jnp.exp(
      jnp.where(completion_mask > 0, per_token_logps - old_per_token_logps, 0.0)
  )

  # Compute pg_clipfrac
  pg_losses_1 = -seq_importance_ratio * advantages
  pg_losses_2 = (
      -jnp.clip(seq_importance_ratio, 1 - epsilon_low, 1 + epsilon_high)
      * advantages
  )

  per_token_loss = jnp.maximum(pg_losses_1, pg_losses_2)

  # add dual clip logic
  epsilon_c = getattr(algo_config, "epsilon_c", None)
  if epsilon_c is not None:
    pg_loss_3 = -epsilon_c * advantages
  else:
    pg_loss_3 = per_token_loss
  unreduced_pg_clipfrac_lower = jnp.sum(
      ((per_token_loss > pg_loss_3) & (advantages < 0.0)).astype(jnp.float32)
      * completion_mask
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  pg_losses = jnp.where(advantages < 0.0, pg_loss_clipped_dual, per_token_loss)
  pg_losses = jnp.where(completion_mask > 0, pg_losses, 0.0)

  denominator = jnp.sum(completion_mask)
  unreduced_pg_clipfrac = jnp.sum(
      jnp.greater(pg_losses_2, pg_losses_1).astype(jnp.float32)
      * completion_mask
  )
  unreduced_policy_loss = jnp.sum(pg_losses)

  aux = {
      "pg_clipfrac": sft_utils.WeightedMetric(
          unreduced_pg_clipfrac, denominator, min_denom=1.0
      ),
      "pg_clipfrac_lower": sft_utils.WeightedMetric(
          unreduced_pg_clipfrac_lower, denominator, min_denom=1.0
      ),
  }

  if return_entropy:
    unreduced_entropy = jnp.sum(token_entropy * completion_mask)  # pyrefly: ignore[unbound-name]
    unreduced_policy_loss = (
        unreduced_policy_loss - entropy_coef * unreduced_entropy
    )
    aux["loss/entropy"] = sft_utils.WeightedMetric(
        unreduced_entropy, denominator, min_denom=1.0
    )

  # kl penalty term logic as before
  kl_coef = getattr(algo_config, "kl_coef", 0.0)
  if kl_coef > 0.0 and train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        "kl",
        clamp_value=getattr(algo_config, "kl_clamp_value", None),
    )
    unreduced_kl = jnp.sum(kl * completion_mask)
    unreduced_policy_loss = unreduced_policy_loss + kl_coef * unreduced_kl
    aux["kl"] = sft_utils.WeightedMetric(
        unreduced_kl, denominator, min_denom=1.0
    )

  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(
          unreduced_policy_loss, denominator, min_denom=1.0
      ),
      aux_metrics=aux,
  )


@function_registry.register_value_loss_fn("ppo")
def ppo_value_loss_fn(
    model: nnx.Module,
    train_example,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
) -> sft_utils.LossOutput:
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns

  segment_ids = getattr(train_example, "segment_ids", None)
  if segment_ids is not None:
    # For packed sequences, prompt_ids is empty and completion_ids holds the
    # full sequence.
    # We predict values for token t using the model's output at t-1.
    logits_to_keep = completion_ids.shape[1] - 1
  else:
    logits_to_keep = completion_ids.shape[1]

  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      segment_ids=segment_ids,
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]

  if segment_ids is not None:
    # Pad the first token's value with 0.0, since it has no preceding token to predict it.
    vpreds = jnp.pad(vpreds, ((0, 0), (1, 0)), constant_values=0.0)
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)

  denominator = jnp.sum(completion_mask)
  unreduced_vf_loss = 0.5 * jnp.sum(clipped_vf_losses * completion_mask)
  unreduced_vpred_mean = jnp.sum(vpreds * completion_mask)
  unreduced_vf_clipfrac = jnp.sum(
      jnp.greater(vf_losses2, vf_losses1).astype(jnp.float32) * completion_mask
  )
  unreduced_return_mean = jnp.sum(returns * completion_mask)

  primary_loss = sft_utils.WeightedMetric(
      unreduced_vf_loss, denominator, min_denom=1.0
  )
  aux = {
      "vf_loss": primary_loss,
      "vpred_mean": sft_utils.WeightedMetric(
          unreduced_vpred_mean, denominator, min_denom=1.0
      ),
      "vf_clipfrac": sft_utils.WeightedMetric(
          unreduced_vf_clipfrac, denominator, min_denom=1.0
      ),
      "return_mean": sft_utils.WeightedMetric(
          unreduced_return_mean, denominator, min_denom=1.0
      ),
  }

  return sft_utils.LossOutput(primary_loss=primary_loss, aux_metrics=aux)


@function_registry.register_policy_loss_fn("grpo")
def grpo_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
) -> sft_utils.LossOutput:
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    algo_config: The algorithm config.
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from.

  Returns:
    A LossOutput containing the loss and an aux dictionary.
  """
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  loss_algo = algo_config.loss_algo
  epsilon_high = (
      algo_config.epsilon_high
      if hasattr(algo_config, "epsilon_high")
      else epsilon
  )
  epsilon_c = getattr(algo_config, "epsilon_c", None)
  loss_aggregation_mode = algo_config.loss_agg_mode

  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # Packing metadata: `segment_ids` labels each token's sequence within a packed
  # row; `num_segments` is the static segment-bucket count. Both are None when
  # not packing, in which case every aggregate_loss/reduced_loss_agg below (and
  # the gspo-token pooling) takes its per-row branch unchanged.
  segment_ids = getattr(train_example, "segment_ids", None)
  num_segments = getattr(train_example, "num_segments", None)

  # Sequence-level options, read up front so an unsupported combination fails
  # before any compute. All of them act per sequence and are applied below,
  # once this step's `per_token_logps` exist to compare the sampler against.
  mask_overlong = getattr(algo_config, "overlong_loss_masking", False)
  mult_prob_error_threshold = getattr(
      algo_config, "seq_logprob_error_threshold", None
  )
  tis_type = getattr(algo_config, "truncated_importance_sampling_type", None)
  tis_band_min = getattr(
      algo_config, "truncated_importance_sampling_ratio_min", None
  )
  tis_band_max = getattr(
      algo_config, "truncated_importance_sampling_ratio", None
  )
  # The checks below are on the batch, which `GRPOConfig` cannot see when it
  # validates the options themselves. Both refuse rather than let an option
  # silently disable itself.
  # Both gates compare the sampler against the trainer, so both need the
  # rollout engine's log-probabilities.
  if (
      mult_prob_error_threshold is not None or tis_type is not None
  ) and getattr(train_example, "rollout_per_token_logps", None) is None:
    raise ValueError(
        "seq_logprob_error_threshold and"
        " truncated_importance_sampling_type require the rollout engine's"
        " per-token log-probabilities, which this batch does not carry."
        " Enable them on the rollout config, or unset these options."
    )
  # Dropping a sequence has to take it out of the loss denominator as well as
  # the numerator. Every aggregation mode derives its denominator from the mask
  # it is given except `sequence-mean-token-sum-norm`, which divides by a fixed
  # factor, so there a drop would scale the loss down instead -- the semantics
  # of an importance weight, not of a mask.
  if loss_aggregation_mode == "sequence-mean-token-sum-norm" and (
      mask_overlong or mult_prob_error_threshold is not None
  ):
    raise ValueError(
        "overlong_loss_masking and seq_logprob_error_threshold drop sequences"
        " from the loss denominator, which loss_agg_mode"
        " 'sequence-mean-token-sum-norm' cannot express: it normalises by a"
        " fixed factor rather than by the mask. Use another aggregation mode,"
        " or truncated_importance_sampling_type, which scales the loss down by"
        " design."
    )
  # Overlong masking needs the rollout's truncation verdict. Not every rollout
  # source reports one, and without it the option would keep every sequence.
  if mask_overlong and getattr(train_example, "overlong", None) is None:
    raise ValueError(
        "overlong_loss_masking requires a per-sequence truncation verdict,"
        " which this batch does not carry. Use a rollout source that reports a"
        " trajectory status, or unset the option."
    )

  # TODO(tsbao): split can be avoided with updated peft_trainer model handling.
  graphdef, state = nnx.split(model)
  completion_attention_mask = getattr(
      train_example, "completion_attention_mask", None
  )
  token_mask = None
  if isinstance(completion_attention_mask, (jax.Array, np.ndarray)):
    token_mask = jnp.concatenate(
        [train_example.prompt_mask, completion_attention_mask], axis=1
    )
  per_token_logps, token_entropy = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_entropy=True,
      segment_ids=segment_ids,
      segment_positions=getattr(train_example, "segment_positions", None),
      temperature=algo_config.temperature,
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
      routed_experts=getattr(train_example, "routed_experts", None),
      token_mask=token_mask,
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)

  # Per-token trainer-minus-sampler log ratio, computed once and shared by the
  # gate, the correction and the diagnostics, so none of them can disagree
  # about what the disagreement is. `log_is_raw` keeps the infinities, because
  # the importance weights need exp() to collapse them to 0 and discard the
  # token; `log_is` is the sanitised version used everywhere an infinity would
  # otherwise poison a whole sequence's average.
  rollout_logps = getattr(train_example, "rollout_per_token_logps", None)
  log_is_raw = None
  log_is = None
  if rollout_logps is not None:
    log_is_raw = jax.lax.stop_gradient(per_token_logps) - jnp.astype(
        rollout_logps, jnp.float32
    )
    log_is = jnp.nan_to_num(log_is_raw, nan=0.0, posinf=0.0, neginf=0.0)

  # `loss_mask` is `completion_mask` with dropped sequences removed and is what
  # the loss and its denominator aggregate over. `completion_mask` remains the
  # full scored-token set and is what the diagnostics measure, so a drop cannot
  # quietly change what they report. With no source enabled the two are
  # identical and every aggregation below is unchanged.
  sample_mask, loss_mask, seq_mult_prob_error = sequence_loss_mask(
      completion_mask,
      overlong=getattr(train_example, "overlong", None),
      mask_overlong=mask_overlong,
      log_is_raw=log_is_raw,
      mult_prob_error_threshold=mult_prob_error_threshold,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )

  # TODO(tsbao): We should handle token level advantages.
  advantages = jnp.astype(train_example.advantages, jnp.float32)

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = jnp.astype(
        train_example.old_per_token_logps, jnp.float32
    )

  # Advantages must be broadcast against seq_length.
  # When sequence packing is used, advantages are already 2D [B, seq_length].
  # When unpacked, they are 1D [B].
  adv = advantages if advantages.ndim == 2 else jnp.expand_dims(advantages, 1)

  valid_loss_mask = (
      (loss_mask > 0)
      & jnp.isfinite(per_token_logps)
      & jnp.isfinite(old_per_token_logps)
      & jnp.isfinite(adv)
  )
  loss_mask = jnp.where(valid_loss_mask, loss_mask, 0.0)
  adv = jnp.where(valid_loss_mask, adv, 0.0)
  masked_logps = jnp.where(valid_loss_mask, per_token_logps, 0.0)
  masked_old_logps = jnp.where(valid_loss_mask, old_per_token_logps, 0.0)
  seq_importance_ratio = masked_logps - masked_old_logps
  # Record KL divergence before clipping.
  token_denom = jnp.sum(loss_mask)
  unreduced_ppo_kl = jnp.sum(-seq_importance_ratio)

  seq_importance_ratio = jnp.clip(seq_importance_ratio, max=20.0, min=-20.0)

  # TODO(sizhi): Refactor this to a separate function.
  if loss_algo == "gspo-token":
    if segment_ids is None:
      # Per-row mean log-ratio: each row is exactly one sequence.
      seq_mean_ratio = seq_importance_ratio.sum(axis=-1) / jnp.clip(
          loss_mask.sum(-1), min=1
      )
      seq_mean_ratio = jnp.expand_dims(seq_mean_ratio, axis=-1)
    else:
      # Per-SEGMENT mean log-ratio: a packed row holds K sequences, so pooling
      # per row would mix them into one biased ratio. Pool per segment, then
      # scatter each token its own segment's mean via take_along_axis. Padding
      # (segment 0, mask 0) yields 0 and is masked out downstream.
      per_seg_sum = common.segmented_sum(
          seq_importance_ratio,
          segment_ids,
          num_segments,  # pyrefly: ignore[bad-argument-type]
      )
      per_seg_count = common.segmented_count(
          segment_ids, num_segments, mask=loss_mask  # pyrefly: ignore[bad-argument-type]
      )
      per_seg_mean = per_seg_sum / jnp.clip(per_seg_count, min=1.0)
      seq_mean_ratio = jnp.take_along_axis(
          per_seg_mean, segment_ids.astype(jnp.int32), axis=1
      )
    # Sequence-level VALUE, per-token GRADIENT (stop-gradient trick): the
    # `x - stop_grad(x)` term is 0 in value but carries d/dtheta per token.
    seq_importance_ratio = (
        masked_logps
        - jax.lax.stop_gradient(masked_logps)
        + jax.lax.stop_gradient(seq_mean_ratio)
    )
    seq_importance_ratio = jnp.where(
        loss_mask > 0,
        jnp.clip(seq_importance_ratio, min=-20.0, max=10.0),
        0.0,
    )

  is_ratio = jnp.exp(seq_importance_ratio)

  pg_loss_1 = -adv * is_ratio
  pg_loss_2 = -adv * jnp.clip(is_ratio, 1 - epsilon, 1 + epsilon_high)

  per_token_loss = jnp.maximum(pg_loss_1, pg_loss_2).astype(jnp.float32)

  unreduced_clip_frac = jnp.sum(
      jnp.greater(pg_loss_2, pg_loss_1).astype(jnp.float32) * loss_mask
  )

  # dual-clip ppo loss
  if epsilon_c is not None:
    pg_loss_3 = -epsilon_c * adv
  else:
    pg_loss_3 = per_token_loss

  # pg_clipfrac_lower measures how often dual-clip ppo kicks in.
  # It kicks in when the standard clipped loss is larger than pg_loss_3
  # for instances with negative advantages.
  per_token_pg_clipfrac_lower = (
      (per_token_loss > pg_loss_3) & (adv < 0.0)
  ).astype(jnp.float32)
  pg_clipfrac_lower = common.aggregate_loss(
      per_token_pg_clipfrac_lower,
      loss_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  per_token_loss = jnp.where(adv < 0.0, pg_loss_clipped_dual, per_token_loss)

  # Optional truncated importance-sampling (TIS) correction for the residual
  # sampler-vs-trainer log-probability mismatch. The weights are precomputed
  # upstream (already detached and threshold-clipped) and applied per token
  # BEFORE loss aggregation so they affect the gradient through the loss
  # magnitude only, not as a stop-gradient bias on the ratio.
  # Per-sequence agreement, shared by the correction and the diagnostics.
  seq_geomean = None
  seq_valid = None
  if log_is is not None:
    seq_geomean, seq_valid = sequence_geomean_ratio(
        log_is, completion_mask, segment_ids, num_segments
    )

  # Use jnp.where (XLA Select) rather than multiplicative masking so masked/pad
  # tokens sever reverse-mode autodiff instead of evaluating 0.0 * Inf = NaN.
  sampler_is_weights = getattr(train_example, "sampler_is_weights", None)
  # Computed here rather than upstream: the trainer log-probabilities the
  # weights need are the ones this forward pass just produced, so there is no
  # second pass and no batching difference between the two sides.
  tis_oob_ratio = None
  if tis_type is not None and log_is_raw is not None:
    sampler_is_weights, tis_oob_ratio = truncated_importance_weights(
        log_is_raw,
        seq_geomean,
        seq_valid,
        sample_mask,
        band_min=tis_band_min,
        band_max=tis_band_max,
        segment_ids=segment_ids,
    )
  if sampler_is_weights is not None:
    per_token_loss = jnp.where(
        (loss_mask > 0) & (sampler_is_weights != 0),
        per_token_loss * sampler_is_weights.astype(jnp.float32),
        0.0,
    )
  else:
    per_token_loss = jnp.where(loss_mask > 0, per_token_loss, 0.0)

  # Two independent aggregations of the same policy loss (equal today):
  #   unreduced (sum/denom, deferred) — feeds the gradient
  #   reduced   (eager per-sequence mean, pre-CL form) — metric only
  unreduced_pg_loss = common.aggregate_loss(
      per_token_loss,
      loss_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )
  reduced_pg_loss = common.reduced_loss_agg(
      per_token_loss,
      loss_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )
  total_loss = unreduced_pg_loss  # KL added below when beta != 0; feeds gradient
  # Per-token diagnostics — log only over tokens the loss aggregates over.
  #
  # Sequence-level masking can leave a micro-batch with no such tokens at all,
  # most easily when `train_micro_batch_size == num_generations` and one
  # micro-batch is one prompt group. Each extremum keeps the +/-inf neutral
  # element of its own reduction for that case, because these are pooled across
  # a step's micro-batches with the matching max/min: an empty micro-batch then
  # contributes nothing, where any finite sentinel would win the reduction and
  # replace the step's real value.
  # 1.0 for rows (or segments under packing) that hold a sequence at all, as
  # opposed to the assembler's trailing padding, which carries no scored tokens.
  if segment_ids is None:
    valid_seq_mask = (completion_mask.sum(axis=-1) > 0).astype(jnp.float32)
  else:
    valid_seq_mask = (
        common.segmented_count(segment_ids, num_segments, mask=completion_mask)
        > 0
    ).astype(jnp.float32)
  # Per-token diagnostics — log only over assistant tokens (completion_mask).
  has_valid = jnp.any(completion_mask > 0)
  is_ratio_mean = masked_mean(is_ratio, completion_mask)
  is_ratio_max = jnp.max(jnp.where(completion_mask > 0, is_ratio, 0.0))
  is_ratio_min = jnp.where(
      has_valid,
      jnp.min(jnp.where(completion_mask > 0, is_ratio, jnp.inf)),
      0.0,
  )
  log_ratio_abs_mean = masked_mean(
      jnp.abs(seq_importance_ratio), loss_mask
  )
  pg_loss_1_mean = masked_mean(pg_loss_1, loss_mask)
  pg_loss_2_mean = masked_mean(pg_loss_2, loss_mask)
  adv_broadcast = jnp.broadcast_to(adv, completion_mask.shape)
  adv_abs_mean = masked_mean(jnp.abs(adv_broadcast), completion_mask)
  adv_max = jnp.where(
      has_valid,
      jnp.max(jnp.where(completion_mask > 0, adv_broadcast, -jnp.inf)),
      0.0,
  )
  adv_min = jnp.where(
      has_valid,
      jnp.min(jnp.where(completion_mask > 0, adv_broadcast, jnp.inf)),
      0.0,
  )
  nonzero_adv_frac = masked_mean(
      (jnp.abs(adv_broadcast) > 1e-8).astype(jnp.float32), loss_mask
  )
  aux = {
      "kl": sft_utils.WeightedMetric(jnp.array(0.0), jnp.array(1.0)),
      "kl_loss": sft_utils.WeightedMetric(jnp.array(0.0), jnp.array(1.0)),
      "reduced_pg_loss": reduced_pg_loss,
      # TODO(yuxzhang): equal to reduced_pg_loss today; diverges once sequence
      # packing lands (reduced -> segment-aware metric; unreduced -> global).
      # Also diverges when sequence-level masking leaves micro-batches with
      # differing denominators, since one pools means and the other pools sums.
      "unreduced_pg_loss": unreduced_pg_loss,
      "pg_clipfrac": sft_utils.WeightedMetric(
          unreduced_clip_frac, token_denom, min_denom=1.0
      ),
      "ppo_kl": sft_utils.WeightedMetric(
          unreduced_ppo_kl, token_denom, min_denom=1.0
      ),
      "pg_clipfrac_lower": pg_clipfrac_lower,
      "is_ratio/mean": is_ratio_mean,
      "is_ratio/max": is_ratio_max,
      "is_ratio/min": is_ratio_min,
      "log_ratio/abs_mean": log_ratio_abs_mean,
      "pg_loss/unclipped_mean": pg_loss_1_mean,
      "pg_loss/clipped_mean": pg_loss_2_mean,
      "advantage/abs_mean": adv_abs_mean,
      "advantage/max": adv_max,
      "advantage/min": adv_min,
      "advantage/nonzero_frac": nonzero_adv_frac,
      # Fraction of sequences still contributing, 1.0 when no sequence-level
      # masking is enabled. Sequences dropped here leave the loss denominator
      # too, so this measures lost effective batch size, where
      # `tis/is_oob_ratio` measures lost gradient magnitude. Counted over rows
      # that hold a sequence, so the assembler's trailing padding rows neither
      # inflate nor deflate it.
      "sample_mask/kept_frac": sft_utils.WeightedMetric(
          jnp.sum(sample_mask * valid_seq_mask),
          jnp.sum(valid_seq_mask),
          min_denom=1.0,
      ),
  }
  if seq_mult_prob_error is not None:
    # Reported over every sequence the gate saw, kept or dropped, so the
    # distribution is visible before the threshold is changed.
    #
    # Weighted rather than pre-averaged because `sequence_mult_prob_error`
    # scores a fully-masked row 0.0, while the metric is >= 1 on any real row.
    # A micro-batch with no rows left must therefore contribute no weight
    # rather than a 0.0 that would pull the pooled mean below its own floor.
    scored = (seq_mult_prob_error > 0) & jnp.isfinite(seq_mult_prob_error)
    scored_f = scored.astype(seq_mult_prob_error.dtype)
    safe_seq_err = jnp.where(scored, seq_mult_prob_error, 0.0)
    aux["sample_mask/mult_prob_error_mean"] = sft_utils.WeightedMetric(
        jnp.sum(safe_seq_err),
        jnp.sum(scored_f),
        min_denom=1.0,
    )
    aux["sample_mask/mult_prob_error_max"] = jnp.max(safe_seq_err)
  if tis_oob_ratio is not None:
    # Normalised over the sequences that are actually training, not over every
    # row, so a batch with rows already dropped upstream still reports the rate
    # among those the band could act on.
    aux["tis/is_oob_ratio"] = tis_oob_ratio

  # ---- Sampler-vs-trainer diagnostics (reported only) ---------------------
  # How far the behaviour policy that produced the tokens has drifted from the
  # target policy being updated. Nothing below affects the loss.
  #
  # Every key is emitted on every step, taking a neutral value when the rollout
  # engine returned no log-probabilities. Whether it does is a runtime property,
  # and a learner that registers metric aggregators up front raises on a
  # registered key that `aux` does not carry.
  bucket_edges = getattr(algo_config, "sampler_is_length_buckets", None)
  aux.update(log_is_attribution(log_is, completion_mask, rollout_logps))
  if log_is is None:
    aux["sampler_is/token_logdiff_absmean"] = jnp.float32(0.0)
    aux["sampler_is/token_logdiff_abs_max"] = jnp.float32(0.0)
    aux["sampler_is/token_outlier_frac"] = jnp.float32(0.0)
    aux["sampler_is/token_outliers_per_seq"] = jnp.float32(0.0)
    aux["sampler_is/token_weight_mean"] = jnp.float32(1.0)
    aux["sampler_is/token_weight_max"] = jnp.float32(1.0)
    aux["sampler_is/token_weight_min"] = jnp.float32(1.0)
    aux["sampler_is/seq_geomean_mean"] = jnp.float32(1.0)
    aux["sampler_is/seq_geomean_min"] = jnp.float32(1.0)
    aux["sampler_is/seq_geomean_max"] = jnp.float32(1.0)
    for suffix in sampler_is_length_bucket_metric_names(bucket_edges):
      aux[f"sampler_is/lenscale/{suffix}"] = jnp.float32(0.0)
  else:
    is_w = jnp.nan_to_num(jnp.exp(log_is), nan=0.0, posinf=0.0, neginf=0.0)
    n_seq = jnp.maximum(seq_valid.sum(), 1.0)
    has_seq = seq_valid.sum() > 0
    has_tok = completion_mask.sum() > 0
    aux["sampler_is/token_logdiff_absmean"] = masked_mean(
        jnp.abs(log_is), completion_mask
    )
    aux["sampler_is/token_weight_mean"] = masked_mean(is_w, completion_mask)
    aux["sampler_is/token_weight_max"] = jnp.max(
        jnp.where(completion_mask > 0, is_w, 0.0)
    )
    # The single token the trainer thought least likely relative to the
    # sampler. `token_weight_max` cannot see these, a disagreement in that
    # direction being a weight near zero rather than a large one.
    aux["sampler_is/token_weight_min"] = jnp.where(
        has_tok, jnp.min(jnp.where(completion_mask > 0, is_w, jnp.inf)), 1.0
    )
    absmax, outlier_frac, outliers_per_seq = token_outlier_stats(
        log_is, completion_mask, n_seq
    )
    aux["sampler_is/token_logdiff_abs_max"] = absmax
    aux["sampler_is/token_outlier_frac"] = outlier_frac
    aux["sampler_is/token_outliers_per_seq"] = outliers_per_seq
    aux["sampler_is/seq_geomean_mean"] = (seq_geomean * seq_valid).sum() / n_seq
    aux["sampler_is/seq_geomean_min"] = jnp.where(
        has_seq, jnp.min(jnp.where(seq_valid > 0, seq_geomean, jnp.inf)), 1.0
    )
    aux["sampler_is/seq_geomean_max"] = jnp.where(
        has_seq, jnp.max(jnp.where(seq_valid > 0, seq_geomean, -jnp.inf)), 1.0
    )

    # Whether the per-sequence offset shrinks as sequences get longer. See
    # `sampler_is_length_bucket_sums`.
    if bucket_edges and segment_ids is not None:
      # The buckets are per-sequence and packing has no scatter path for them.
      # This is a diagnostic, so emit neutral keys rather than refuse the step.
      for suffix in sampler_is_length_bucket_metric_names(bucket_edges):
        aux[f"sampler_is/lenscale/{suffix}"] = jnp.float32(0.0)
    elif bucket_edges:
      # Taken directly rather than as log(seq_geomean): the round trip through
      # exp can under- or overflow for a sequence with a large mean log ratio,
      # which is exactly the case this diagnostic is looking for.
      seq_log_mean = (log_is * completion_mask).sum(axis=-1) / jnp.maximum(
          completion_mask.sum(axis=-1), 1.0
      )
      for key, value in sampler_is_length_bucket_sums(
          log_is,
          seq_log_mean,
          completion_mask,
          seq_valid,
          bucket_edges,
          overlong=getattr(train_example, "overlong", None),
      ).items():
        aux[f"sampler_is/lenscale/{key}"] = value
  if sampler_is_weights is not None:
    sis = sampler_is_weights.astype(jnp.float32)
    aux["sampler_is/weight_mean"] = masked_mean(sis, completion_mask)
    aux["sampler_is/weight_min"] = jnp.where(
        has_valid,
        jnp.min(jnp.where(completion_mask > 0, sis, jnp.inf)),
        0.0,
    )
  else:
    aux["sampler_is/weight_mean"] = jnp.float32(1.0)
    aux["sampler_is/weight_min"] = jnp.float32(1.0)

  # We do not always compute KL divergence (e.g. when beta is 0.0 unless
  # force_compute_kl is True).
  if train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        algo_config.kl_loss_mode,
        clamp_value=algo_config.kl_clamp_value,
    )
    # Over `loss_mask`, like the policy loss: `total_loss` below adds the two
    # unreduced sums and divides by the policy loss's denominator, so a KL taken
    # over a wider mask would both scale beta up and give a dropped sequence a
    # reference-KL gradient.
    kl = jnp.where(loss_mask > 0, kl, 0.0)
    unreduced_kl = jnp.astype(jnp.sum(kl), jnp.float32)
    aux["kl"] = sft_utils.WeightedMetric(
        unreduced_kl, token_denom, min_denom=1.0
    )
    kl_loss = common.aggregate_loss(
        kl,
        loss_mask,
        loss_aggregation_mode,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )
    aux["kl_loss"] = kl_loss  # pyrefly: ignore[bad-assignment]
  if beta is not None and beta != 0.0:
    total_loss = sft_utils.WeightedMetric(
        unreduced_pg_loss.unreduced_sum + beta * kl_loss.unreduced_sum,  # pyrefly: ignore[unbound-name]
        unreduced_pg_loss.denominator,
        eps=unreduced_pg_loss.eps,
        min_denom=unreduced_pg_loss.min_denom,
    )

  token_entropy = jnp.where(
      (loss_mask > 0) & jnp.isfinite(token_entropy),
      jax.lax.stop_gradient(token_entropy),
      0.0,
  )
  entropy_loss = common.aggregate_loss(
      token_entropy,
      loss_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )
  aux["entropy"] = entropy_loss

  return sft_utils.LossOutput(primary_loss=total_loss, aux_metrics=aux)  # pyrefly: ignore[bad-argument-type]


@function_registry.register_advantage_estimator("grpo")
def compute_advantages(rewards: np.ndarray, num_generations: int) -> np.ndarray:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
  std_grouped_rewards = rewards.reshape(-1, num_generations).std(
      axis=-1, ddof=1
  )

  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
  return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)


# Fraction of a group's total squared deviation below which a leave-one-out
# variance is float rounding rather than spread. A few orders above float32
# epsilon, so it absorbs the rounding of the sum it is compared against without
# suppressing any variance a reward function could meaningfully produce.
_LOO_VAR_NOISE_FLOOR = 1e-5


@function_registry.register_advantage_estimator("grpo-loo")
def compute_grpo_loo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  """Group-relative advantages with a leave-one-out baseline and scale.

  Differs from `compute_advantages` in excluding each sample from the
  statistics used to judge it: both the baseline and the standard deviation
  come from the other generations for the same prompt. Including a sample in
  its own baseline scales its advantage by `1 - 1/G`, which under-scales every
  gradient by 6.25% at G=16.

  Differs from `compute_rloo_advantages` in dividing by the leave-one-out
  standard deviation, so advantages stay comparable across prompts of differing
  difficulty instead of letting high-variance prompts dominate the update.

  Groups are positional: `rewards` must be laid out with each prompt's
  generations contiguous, `[p0g0, p0g1, ..., p1g0, ...]`, as the rollout
  engines produce them. Interleaved input merges prompts into one group and
  destroys the normalisation rather than failing.

  Args:
    rewards: Per-sequence rewards, `[num_prompts * num_generations]`.
    num_generations: Generations per prompt.

  Returns:
    Advantages with the same shape as `rewards`.
  """
  if num_generations < 2:
    # No other sample to form a baseline from.
    return jnp.zeros_like(rewards)

  grouped = rewards.reshape(-1, num_generations)
  n_others = num_generations - 1
  # Everything below works on deviations from the group mean. The identities
  # are the same as on the raw rewards, but subtracting the mean once up front
  # keeps the variance from cancelling catastrophically in float32 when the
  # rewards are large relative to their spread.
  centered = grouped - grouped.mean(axis=-1, keepdims=True)
  # x_i - mean(x_{j != i}) reduces to the deviation scaled by (n + 1) / n.
  advantages = centered * ((n_others + 1) / n_others)

  # The leave-one-out set holds `n_others` samples, so an unbiased variance
  # needs at least two of them. At num_generations == 2 the variance is
  # undefined and no scaling is applied.
  if n_others >= 2:
    # sum_{j != i} (x_j - mean(x_{j != i}))^2, expanded around the group mean.
    sum_sq = jnp.square(centered).sum(axis=-1, keepdims=True)
    loo_var = (
        sum_sq - jnp.square(centered) * ((n_others + 1) / n_others)
    ) / (n_others - 1)
    # A sample whose leave-one-out set has no spread keeps its raw advantage
    # rather than being divided by a near-zero scale, which would manufacture a
    # large advantage out of rounding. The comparison is against the group's own
    # scale, not against zero: the expression above is a difference of two
    # similar positive numbers, so a genuinely zero variance comes out as float
    # noise. Decided per sample rather than per group, so a group can mix
    # normalised and raw advantages.
    loo_var = jnp.where(loo_var > _LOO_VAR_NOISE_FLOOR * sum_sq, loo_var, 0.0)
    loo_std = jnp.sqrt(loo_var)
    advantages = jnp.where(
        loo_std > 0, advantages / (loo_std + 1e-6), advantages
    )

  return advantages.flatten()


@function_registry.register_advantage_estimator("rloo")
def compute_rloo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  """Compute RLOO (REINFORCE Leave-One-Out) advantages.

  RLOO computes a baseline for each completion by averaging the rewards of all
  other completions to the same prompt.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    RLOO advantages.
  """
  if num_generations < 2:
    # RLOO requires at least 2 samples to calculate a baseline.
    return jnp.zeros_like(rewards)

  reshaped_rewards = rewards.reshape(-1, num_generations)
  loo_mean = (
      reshaped_rewards.sum(axis=-1, keepdims=True) - reshaped_rewards
  ) / (num_generations - 1)
  rloo_advantages = reshaped_rewards - loo_mean

  return rloo_advantages.flatten()


# ==============================================================================
# DrGRPO Core
# ==============================================================================


@function_registry.register_advantage_estimator("drgrpo")
def compute_drgrpo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  """Group relative advantages -- done right.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=1)
  return rewards - mean_grouped_rewards.repeat(num_generations)

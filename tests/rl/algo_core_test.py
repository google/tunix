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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core
from tunix.rl import common


class AlgoCoreTest(absltest.TestCase):

  def test_compute_rloo_advantages(self):
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=3)
    expected_value = jnp.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected_value)

  def test_compute_rloo_advantages_low_generations(self):
    rewards = jnp.array([1.0, 2.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=1)
    np.testing.assert_allclose(advantages, jnp.zeros_like(rewards))

  def test_grpo_compute_advantages(self):
    prev_val = jax.config.jax_threefry_partitionable
    self.addCleanup(jax.config.update, 'jax_threefry_partitionable', prev_val)
    jax.config.update('jax_threefry_partitionable', False)
    self.assertFalse(jax.config.jax_threefry_partitionable)

    rng = jax.random.PRNGKey(0)
    rewards = jax.random.uniform(rng, shape=(1, 6))
    advantages = algo_core.compute_advantages(rewards, num_generations=3)
    expected_value = jnp.array(
        [[0.307498, -1.117636, 0.810138, 1.094526, -0.228671, -0.865855]]
    )
    np.testing.assert_allclose(advantages, expected_value, rtol=1e-3, atol=1e-3)

  def test_grpo_loss_fn_packed_equals_unpacked(self):
    # P3.4 gate: grpo_loss_fn gives the SAME primary loss whether two sequences
    # are packed into one row (segment_ids set) or one-per-row (segment_ids
    # None). Proves segment_ids/num_segments are threaded into the loss
    # aggregation and the gspo-token per-segment pooling. old_per_token_logps is
    # None (is_ratio == 1), so the model output cancels and this isolates the
    # aggregation wiring: sequence-mean-token-mean over A (adv 1.5, 3 tokens) and
    # B (adv 3.0, 1 token) = (-1.5 + -3.0) / 2 = -2.25; a broken per-row
    # aggregation would instead give -1.875.
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import common  # pylint: disable=g-import-not-at-top

    class _SegAwareToy(nnx.Module):
      """Tiny model whose attention is confined to same-segment positions."""

      def __init__(self, *, vocab, dim, rngs):
        self.emb = nnx.Embed(vocab, dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=dim,
            qkv_features=dim,
            use_bias=False,
            decode=False,
            rngs=rngs,
        )
        self.head = nnx.Linear(dim, vocab, rngs=rngs)

      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        h = self.emb(x)
        if segment_ids is not None:
          same_seg = segment_ids[:, :, None] == segment_ids[:, None, :]
          h = self.attn(h, mask=same_seg[:, None, :, :]) + h
        else:
          h = self.attn(h) + h
        return self.head(h), cache

    model = _SegAwareToy(vocab=16, dim=8, rngs=nnx.Rngs(0))
    packed = common.TrainExample(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[3, 4, 5, 6]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1, 1]], jnp.float32),
        advantages=jnp.array([[1.5, 1.5, 1.5, 3.0]], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=jnp.array([[1, 1, 1, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 0]], jnp.int32),
        num_segments=3,
    )
    unpacked = common.TrainExample(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 0, 0]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1], [1, 0, 0]], jnp.float32),
        advantages=jnp.array([1.5, 3.0], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )
    for loss_algo in ('grpo', 'gspo-token'):
      cfg = SimpleNamespace(
          beta=0.0,
          epsilon=0.2,
          epsilon_high=0.2,
          epsilon_c=None,
          loss_algo=loss_algo,
          loss_agg_mode='sequence-mean-token-mean',
          temperature=1.0,
          kl_loss_mode='low_var_kl',
          kl_clamp_value=None,
          force_compute_kl=False,
      )
      lp = float(
          algo_core.grpo_loss_fn(
              model, packed, cfg, pad_id=0, eos_id=-1
          ).primary_loss.compute()
      )
      lu = float(
          algo_core.grpo_loss_fn(
              model, unpacked, cfg, pad_id=0, eos_id=-1
          ).primary_loss.compute()
      )
      with self.subTest(loss_algo=loss_algo):
        np.testing.assert_allclose(lp, lu, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(lp, -2.25, rtol=1e-4, atol=1e-4)


class GrpoLooAdvantagesTest(absltest.TestCase):
  """Leave-one-out group-relative advantages."""

  def _reference(self, rewards, num_generations):
    """Independent transcription of the intended formula, in matrix form.

    Written from the definition rather than from the implementation, so it
    fails if the implementation drifts toward something merely self-consistent.
    """
    r = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(r)
    g = num_generations
    for p in range(len(r) // g):
      idx = slice(p * g, (p + 1) * g)
      rr = r[idx]
      others = 1 - np.eye(g)  # exclude self
      n = np.float64(g - 1)
      base = others @ rr / n
      sq = others @ (rr**2) / n
      with np.errstate(divide='ignore', invalid='ignore'):
        std = np.nan_to_num(
            np.sqrt((sq - base**2) * (n / (n - np.float64(1)))), nan=0.0
        )
      adv = rr - base
      nz = std > 0
      adv[nz] = adv[nz] / (std[nz] + 1e-6)
      out[idx] = adv
    return out

  def test_matches_the_definition_across_group_sizes(self):
    rng = np.random.default_rng(0)
    for g in (2, 3, 4, 8, 16):
      rewards = rng.random(g * 3)
      with self.subTest(num_generations=g):
        np.testing.assert_allclose(
            algo_core.compute_grpo_loo_advantages(jnp.asarray(rewards), g),
            self._reference(rewards, g),
            rtol=1e-5,
            atol=1e-6,
        )

  def test_binary_rewards(self):
    # The realistic case: solved/unsolved within a group.
    rewards = jnp.array([1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        algo_core.compute_grpo_loo_advantages(rewards, 4),
        self._reference(np.asarray(rewards), 4),
        rtol=1e-5,
    )

  def test_excludes_self_from_the_baseline(self):
    # Including a sample in its own baseline shrinks its advantage by exactly
    # 1 - 1/G. That factor is the whole reason this estimator exists.
    grouped = np.arange(16, dtype=np.float64)
    plain = grouped - grouped.mean()
    loo_mean = (grouped.sum() - grouped) / 15
    np.testing.assert_allclose(
        plain / (grouped - loo_mean), np.full(16, 1 - 1 / 16), rtol=1e-6
    )

  def test_degenerate_groups(self):
    # G=1: no other sample to form a baseline from.
    np.testing.assert_array_equal(
        algo_core.compute_grpo_loo_advantages(jnp.array([1.0, 2.0]), 1),
        np.zeros(2),
    )
    # G=2: the leave-one-out set holds a single sample, so its variance is
    # undefined; advantages stay as raw differences.
    np.testing.assert_allclose(
        algo_core.compute_grpo_loo_advantages(jnp.array([1.0, 0.0]), 2),
        [1.0, -1.0],
        rtol=1e-6,
    )

  def test_zero_variance_group_is_not_sharpened(self):
    # All generations scored identically: advantages are 0 and must stay 0,
    # not become large numbers from dividing rounding by rounding.
    out = algo_core.compute_grpo_loo_advantages(jnp.full((8,), 0.7), 4)
    np.testing.assert_allclose(out, np.zeros(8), atol=1e-6)

  def test_groups_are_independent(self):
    # Changing one group must not move another.
    a = algo_core.compute_grpo_loo_advantages(
        jnp.array([1.0, 0.0, 1.0, 0.0, 0.9, 0.8, 0.7, 0.6]), 4
    )
    b = algo_core.compute_grpo_loo_advantages(
        jnp.array([1.0, 0.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4]), 4
    )
    np.testing.assert_allclose(a[:4], b[:4], rtol=1e-6)


class SequenceLossMaskTest(absltest.TestCase):
  """Sequence-level loss masking, and the denominator behaviour it implies."""

  def _mask(self, rows=4, tokens=5):
    return jnp.ones((rows, tokens), dtype=jnp.float32)

  def test_inactive_by_default(self):
    # No source enabled -> unrestricted outputs, so callers can use them
    # unconditionally and existing behaviour is unchanged.
    completion_mask = self._mask()
    result = algo_core.sequence_loss_mask(completion_mask)
    np.testing.assert_array_equal(result.sample_mask, np.ones(4))
    np.testing.assert_array_equal(result.loss_mask, completion_mask)
    self.assertIsNone(result.mult_prob_error)

  def test_overlong_ignored_unless_enabled(self):
    completion_mask = self._mask()
    result = algo_core.sequence_loss_mask(
        completion_mask,
        overlong=jnp.array([0.0, 1.0, 1.0, 0.0]),
        mask_overlong=False,
    )
    np.testing.assert_array_equal(result.loss_mask, completion_mask)

  def test_overlong_drops_whole_sequences(self):
    result = algo_core.sequence_loss_mask(
        self._mask(),
        overlong=jnp.array([0.0, 1.0, 1.0, 0.0]),
        mask_overlong=True,
    )
    np.testing.assert_array_equal(result.sample_mask, [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        result.loss_mask.sum(axis=-1), [5.0, 0.0, 0.0, 5.0]
    )

  def test_enabled_but_no_verdict_is_a_noop(self):
    # The rollout engine may report no truncation; that must not silently drop
    # everything or crash.
    completion_mask = self._mask()
    result = algo_core.sequence_loss_mask(
        completion_mask, overlong=None, mask_overlong=True
    )
    np.testing.assert_array_equal(result.loss_mask, completion_mask)

  def test_partial_completion_mask_is_preserved(self):
    completion_mask = jnp.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    result = algo_core.sequence_loss_mask(
        completion_mask,
        overlong=jnp.array([0.0, 0.0, 1.0]),
        mask_overlong=True,
    )
    np.testing.assert_array_equal(
        result.loss_mask, [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

  def test_dropped_sequences_leave_the_denominator(self):
    """The load-bearing property: a drop must not scale the loss down.

    Masking removes a sequence from numerator and denominator alike, so the
    survivors keep their full gradient magnitude. Zeroing a weight removes it
    from the numerator only, shrinking the loss by the drop fraction. Getting
    this backwards is a silent learning-rate change, so it is asserted against
    the real aggregators for every mode the loss can be configured with.
    """
    completion_mask = self._mask(rows=4, tokens=5)
    # Identical per-token loss everywhere, so any change in the aggregate can
    # only come from the denominator.
    per_token_loss = jnp.full((4, 5), 0.25, dtype=jnp.float32)
    result = algo_core.sequence_loss_mask(
        completion_mask,
        overlong=jnp.array([0.0, 1.0, 1.0, 0.0]),
        mask_overlong=True,
    )

    for mode in ('token-mean', 'sequence-mean-token-mean'):
      full = common.aggregate_loss(per_token_loss, completion_mask, mode)
      masked = common.aggregate_loss(per_token_loss, result.loss_mask, mode)
      self.assertAlmostEqual(
          float(full.compute()),
          float(masked.compute()),
          places=5,
          msg=f'{mode}: masking changed the aggregate loss',
      )
      # And the denominator really did shrink, i.e. the invariance above is
      # numerator and denominator falling together, not the mask doing nothing.
      self.assertAlmostEqual(
          float(masked.denominator) * 2.0,
          float(full.denominator),
          places=5,
          msg=f'{mode}: expected half the batch to leave the denominator',
      )

  def test_weight_zeroing_does_scale_the_loss_down(self):
    """Contrast case, so the distinction above is pinned rather than assumed."""
    completion_mask = self._mask(rows=4, tokens=5)
    per_token_loss = jnp.full((4, 5), 0.25, dtype=jnp.float32)
    keep = jnp.array([1.0, 0.0, 0.0, 1.0])[:, None]

    full = common.aggregate_loss(per_token_loss, completion_mask, 'token-mean')
    weighted = common.aggregate_loss(
        per_token_loss * keep, completion_mask, 'token-mean'
    )
    self.assertAlmostEqual(
        float(weighted.compute()), float(full.compute()) / 2.0, places=5
    )


class SequenceMultProbErrorTest(absltest.TestCase):
  """The log-probability disagreement gate."""

  def test_perfect_agreement_scores_one(self):
    np.testing.assert_allclose(
        algo_core.sequence_mult_prob_error(jnp.zeros((3, 4)), jnp.ones((3, 4))),
        [1.0, 1.0, 1.0],
    )

  def test_absolute_value_prevents_cancellation(self):
    # A sequence off by +d on half its tokens and -d on the other half has a
    # geometric-mean ratio of exactly 1 and looks perfectly on-policy. This is
    # a data-integrity check, so it must still see the disagreement.
    d = 0.5
    log_is = jnp.array([[d, -d, d, -d]])
    mask = jnp.ones((1, 4))
    self.assertAlmostEqual(
        float(jnp.exp((log_is * mask).sum() / mask.sum())), 1.0, places=6
    )
    self.assertAlmostEqual(
        float(algo_core.sequence_mult_prob_error(log_is, mask)[0]),
        float(np.exp(d)),
        places=5,
    )

  def test_masked_tokens_excluded(self):
    # A huge disagreement on a masked token must not leak in, nor overflow
    # through the exp.
    self.assertAlmostEqual(
        float(
            algo_core.sequence_mult_prob_error(
                jnp.array([[0.0, 0.0, 10.0]]), jnp.array([[1.0, 1.0, 0.0]])
            )[0]
        ),
        1.0,
        places=6,
    )

  def test_fully_masked_sequence_scores_zero(self):
    # Must not divide by zero, and must not trip a threshold.
    self.assertEqual(
        float(
            algo_core.sequence_mult_prob_error(
                jnp.array([[1.0, 1.0]]), jnp.zeros((1, 2))
            )[0]
        ),
        0.0,
    )

  def test_gate_drops_only_sequences_over_threshold(self):
    # Row errors: exp(0)=1.0, exp(0.5)=1.649, exp(1.5)=4.482.
    result = algo_core.sequence_loss_mask(
        jnp.ones((3, 4), dtype=jnp.float32),
        log_is=jnp.array([[0.0] * 4, [0.5] * 4, [1.5] * 4]),
        mult_prob_error_threshold=2.0,
    )
    np.testing.assert_allclose(
        result.mult_prob_error, [1.0, np.exp(0.5), np.exp(1.5)], rtol=1e-5
    )
    np.testing.assert_array_equal(result.sample_mask, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        result.loss_mask.sum(axis=-1), [4.0, 4.0, 0.0]
    )

  def test_gate_inactive_without_threshold_or_logps(self):
    completion_mask = jnp.ones((2, 3), dtype=jnp.float32)
    log_is = jnp.full((2, 3), 5.0)
    for kwargs in (
        {'log_is': log_is, 'mult_prob_error_threshold': None},
        {'log_is': None, 'mult_prob_error_threshold': 2.0},
    ):
      result = algo_core.sequence_loss_mask(completion_mask, **kwargs)
      np.testing.assert_array_equal(result.sample_mask, [1.0, 1.0])
      np.testing.assert_array_equal(result.loss_mask, completion_mask)
      self.assertIsNone(result.mult_prob_error)

  def test_overlong_applies_before_the_gate(self):
    # A truncated sequence contributes no tokens to its own error, so it
    # scores 0.0 and passes the threshold -- but must stay dropped regardless.
    result = algo_core.sequence_loss_mask(
        jnp.ones((2, 4), dtype=jnp.float32),
        overlong=jnp.array([1.0, 0.0]),
        mask_overlong=True,
        log_is=jnp.full((2, 4), 5.0),  # exp(5) >> threshold for both rows
        mult_prob_error_threshold=2.0,
    )
    self.assertEqual(float(result.mult_prob_error[0]), 0.0)
    np.testing.assert_array_equal(result.sample_mask, [0.0, 0.0])


class TruncatedImportanceWeightsTest(absltest.TestCase):
  """seq-mask-tis weights, and the denominator behaviour they imply."""

  _BAND = dict(band_min=0.999, band_max=1.002)

  def _call(self, log_is_raw, completion_mask=None, sample_mask=None):
    log_is_raw = jnp.asarray(log_is_raw)
    if completion_mask is None:
      completion_mask = jnp.ones_like(log_is_raw)
    if sample_mask is None:
      sample_mask = jnp.ones(log_is_raw.shape[0])
    log_is = jnp.nan_to_num(log_is_raw, nan=0.0, posinf=0.0, neginf=0.0)
    geomean, valid = algo_core.sequence_geomean_ratio(log_is, completion_mask)
    return algo_core.truncated_importance_weights(
        log_is_raw, geomean, valid, sample_mask, **self._BAND
    )

  def test_agreement_gives_unit_weights_and_no_drops(self):
    weights, oob = self._call(jnp.zeros((3, 4)))
    np.testing.assert_allclose(weights, np.ones((3, 4)), rtol=1e-6)
    self.assertAlmostEqual(float(oob), 0.0)

  def test_sequence_outside_band_is_zeroed_entirely(self):
    # Row 0 geomean exp(0.0005) = 1.0005 -> inside. Row 1 exp(0.01) -> outside.
    weights, oob = self._call(jnp.array([[0.0005] * 4, [0.01] * 4]))
    self.assertGreater(float(weights[0].min()), 0.0)
    np.testing.assert_array_equal(weights[1], np.zeros(4))
    self.assertAlmostEqual(float(oob), 0.5)

  def test_kept_sequences_keep_raw_untruncated_weights(self):
    # seq-mask-tis clips nothing inside the band: a token whose own ratio is
    # far from 1 survives intact as long as its sequence average is fine.
    log_is = jnp.array([[0.5, -0.5, 0.001, 0.001]])
    weights, oob = self._call(log_is)
    self.assertAlmostEqual(float(oob), 0.0)
    np.testing.assert_allclose(
        weights[0], np.exp(np.asarray(log_is[0])), rtol=1e-5
    )

  def test_infinite_ratio_becomes_zero_not_one(self):
    """nan_to_num goes on the weight, after the exp -- not on the log.

    Sanitising the log first would turn an infinite disagreement into log 0 and
    a weight of 1, silently relabelling a catastrophic token as perfect
    agreement. It has to become 0 so the token is discarded.
    """
    weights, _ = self._call(
        jnp.array([[jnp.inf, 0.0], [-jnp.inf, 0.0], [jnp.nan, 0.0]])
    )
    np.testing.assert_array_equal(weights[:, 0], np.zeros(3))

  def test_geomean_uses_completion_mask_not_padding(self):
    # Padding must not drag a sequence's geometric mean toward 1 and rescue it.
    weights, oob = self._call(
        jnp.array([[0.01, 0.01, 99.0, 99.0]]),
        completion_mask=jnp.array([[1.0, 1.0, 0.0, 0.0]]),
    )
    # Mean over the two real tokens is 0.01 -> outside the band -> dropped.
    self.assertAlmostEqual(float(oob), 1.0)
    np.testing.assert_array_equal(weights[0, :2], np.zeros(2))

  def test_oob_ratio_normalises_over_valid_sequences_only(self):
    # Sequences already dropped upstream leave both the numerator and the
    # denominator of the reported rate.
    _, oob = self._call(
        jnp.array([[0.01] * 3, [0.0] * 3, [0.0] * 3]),
        sample_mask=jnp.array([0.0, 1.0, 1.0]),  # row 0 already dropped
    )
    self.assertAlmostEqual(float(oob), 0.0)
    # Normalising over everything would have called it 1/3.

  def test_drop_scales_the_loss_down_unlike_a_loss_mask(self):
    """The keep-mask multiplies weights, so the denominator is frozen.

    This is the opposite of `sequence_loss_mask` and is the most consequential
    semantic here: getting it backwards is a silent learning-rate change in one
    direction or the other.
    """
    completion_mask = jnp.ones((2, 4), dtype=jnp.float32)
    weights, _ = self._call(
        jnp.array([[0.0] * 4, [0.01] * 4]),  # row 1 outside the band
        completion_mask=completion_mask,
    )
    per_token_loss = jnp.full((2, 4), 0.25, dtype=jnp.float32)

    full = common.aggregate_loss(per_token_loss, completion_mask, 'token-mean')
    corrected = common.aggregate_loss(
        per_token_loss * weights, completion_mask, 'token-mean'
    )
    # Half the sequences dropped -> half the loss, same denominator.
    self.assertAlmostEqual(
        float(corrected.compute()), float(full.compute()) / 2.0, places=5
    )
    self.assertAlmostEqual(
        float(corrected.denominator), float(full.denominator), places=5
    )


class SamplerIsLengthScalingTest(absltest.TestCase):
  """The offset's length-scaling diagnostic.

  The statistic has to separate two hypotheses that look identical in a batch
  average but imply opposite things at production sequence length: iid token
  noise, where the per-sequence offset shrinks as 1/sqrt(T), and a systematic
  within-sequence bias, where it does not shrink at all.
  """

  _EDGES = (256, 512, 1024)
  _LENGTHS = (128, 384, 768, 2048)  # one per bucket, incl. the open-ended one
  _N_PER_LENGTH = 400
  _TOKEN_SIGMA = 0.018

  def _synthesize(self, per_sequence_offset_sigma):
    """A batch of mixed-length sequences under a known hypothesis.

    Args:
      per_sequence_offset_sigma: Scale of a constant offset added to every
        token of a sequence. 0.0 gives pure iid token noise; a positive value
        adds the systematic component that sqrt(T) averaging cannot remove.

    Returns:
      `(log_is, completion_mask)`, right-padded to the longest length.
    """
    rng = np.random.default_rng(0)
    t_max = max(self._LENGTHS)
    rows = len(self._LENGTHS) * self._N_PER_LENGTH
    log_is = np.zeros((rows, t_max), dtype=np.float32)
    mask = np.zeros_like(log_is)
    for i, length in enumerate(self._LENGTHS):
      sl = slice(i * self._N_PER_LENGTH, (i + 1) * self._N_PER_LENGTH)
      tokens = rng.normal(
          0.0, self._TOKEN_SIGMA, size=(self._N_PER_LENGTH, length)
      )
      if per_sequence_offset_sigma:
        tokens += rng.normal(
            0.0, per_sequence_offset_sigma, size=(self._N_PER_LENGTH, 1)
        )
      log_is[sl, :length] = tokens
      mask[sl, :length] = 1.0
    return jnp.asarray(log_is), jnp.asarray(mask)

  def _bucket_sums(self, per_sequence_offset_sigma, overlong=None):
    log_is, completion_mask = self._synthesize(per_sequence_offset_sigma)
    seq_log_mean = (log_is * completion_mask).sum(axis=-1) / (
        completion_mask.sum(axis=-1) + 1e-8
    )
    return algo_core.sampler_is_length_bucket_sums(
        log_is,
        seq_log_mean,
        completion_mask,
        jnp.ones_like(seq_log_mean),
        self._EDGES,
        overlong=overlong,
    )

  def _excess_per_bucket(self, per_sequence_offset_sigma):
    """Applies the offline formulas documented on the emitting function."""
    sums = self._bucket_sums(per_sequence_offset_sigma)
    excess = []
    for name in algo_core.sampler_is_length_bucket_names(self._EDGES):
      prefix = f'{name}/complete'
      count = float(sums[f'{prefix}/count'])
      observed = np.sqrt(float(sums[f'{prefix}/logmean_sq_sum']) / count)
      iid = np.sqrt(float(sums[f'{prefix}/iid_var_sum']) / count)
      excess.append(observed / iid)
    return excess

  def test_bucket_names(self):
    self.assertEqual(
        algo_core.sampler_is_length_bucket_names((256, 1024)),
        ['le256', 'le1024', 'gt1024'],
    )
    self.assertEqual(algo_core.sampler_is_length_bucket_names(None), [])
    self.assertEqual(algo_core.sampler_is_length_bucket_names(()), [])

  def test_buckets_partition_the_batch_by_length(self):
    sums = self._bucket_sums(0.0)
    names = algo_core.sampler_is_length_bucket_names(self._EDGES)
    # Every sequence lands in exactly one bucket, and each bucket holds the
    # length it was built from. With no truncation verdict supplied, all of
    # them report as complete.
    self.assertEqual(
        [float(sums[f'{n}/complete/count']) for n in names],
        [float(self._N_PER_LENGTH)] * len(names),
    )
    self.assertEqual(
        [float(sums[f'{n}/truncated/count']) for n in names],
        [0.0] * len(names),
    )
    for name, length in zip(names, self._LENGTHS):
      self.assertAlmostEqual(
          float(sums[f'{name}/complete/len_sum']) / self._N_PER_LENGTH,
          length,
          places=3,
      )

  def test_completion_status_split_is_disjoint_and_additive(self):
    # Half the sequences marked truncated: the two series must partition the
    # bucket, so summing them recovers the unsplit total.
    n_rows = len(self._LENGTHS) * self._N_PER_LENGTH
    overlong = jnp.asarray(np.tile([0.0, 1.0], n_rows // 2), dtype=jnp.float32)
    split = self._bucket_sums(0.0, overlong=overlong)
    unsplit = self._bucket_sums(0.0)
    for name in algo_core.sampler_is_length_bucket_names(self._EDGES):
      self.assertEqual(
          float(split[f'{name}/complete/count']), self._N_PER_LENGTH / 2
      )
      self.assertEqual(
          float(split[f'{name}/truncated/count']), self._N_PER_LENGTH / 2
      )
      for metric in algo_core.SAMPLER_IS_LENGTH_BUCKET_METRICS:
        np.testing.assert_allclose(
            float(split[f'{name}/complete/{metric}'])
            + float(split[f'{name}/truncated/{metric}']),
            float(unsplit[f'{name}/complete/{metric}']),
            rtol=1e-5,
            err_msg=f'{name}/{metric} is not additive across the split',
        )

  def test_metric_names_cover_what_the_sums_emit(self):
    # The learner registers aggregators from the name list, so it must match
    # the keys the loss actually produces, exactly.
    self.assertCountEqual(
        algo_core.sampler_is_length_bucket_metric_names(self._EDGES),
        list(self._bucket_sums(0.0).keys()),
    )
    self.assertEmpty(algo_core.sampler_is_length_bucket_metric_names(None))

  def test_iid_noise_gives_flat_unit_excess(self):
    # Pure iid tokens: the observed per-sequence spread is exactly what
    # 1/sqrt(T) averaging predicts, at every length.
    for name, value in zip(
        algo_core.sampler_is_length_bucket_names(self._EDGES),
        self._excess_per_bucket(0.0),
    ):
      self.assertBetween(value, 0.9, 1.15, msg=f'bucket {name}')

  def test_systematic_offset_gives_excess_growing_with_length(self):
    # A per-sequence constant offset survives averaging, so the observed
    # spread stays flat in T while the iid prediction keeps falling; their
    # ratio must therefore grow with bucket length. This is the signature.
    excess = self._excess_per_bucket(per_sequence_offset_sigma=0.005)
    self.assertTrue(
        all(a < b for a, b in zip(excess, excess[1:])),
        msg=f'excess should increase with bucket length, got {excess}',
    )
    # 16x length range between the first and last bucket, so ~4x in sqrt(T).
    self.assertGreater(excess[-1] / excess[0], 3.0)


class SequenceGeomeanRatioTest(absltest.TestCase):

  def test_agreement_is_one(self):
    geomean, valid = algo_core.sequence_geomean_ratio(
        jnp.zeros((2, 3)), jnp.ones((2, 3))
    )
    np.testing.assert_allclose(geomean, [1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(valid, [1.0, 1.0])

  def test_geometric_not_arithmetic(self):
    # exp of the mean log, not the mean of the exps: for [+d, -d] those differ
    # (1.0 vs cosh(d)), and the geometric one is what composes correctly.
    d = 0.5
    geomean, _ = algo_core.sequence_geomean_ratio(
        jnp.array([[d, -d]]), jnp.ones((1, 2))
    )
    self.assertAlmostEqual(float(geomean[0]), 1.0, places=6)
    self.assertGreater(float(np.cosh(d)), 1.0)

  def test_empty_sequence_is_marked_invalid(self):
    geomean, valid = algo_core.sequence_geomean_ratio(
        jnp.zeros((2, 3)), jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(valid, [1.0, 0.0])
    self.assertTrue(np.isfinite(float(geomean[1])))


if __name__ == '__main__':
  absltest.main()

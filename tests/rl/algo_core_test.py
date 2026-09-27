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
from tunix.sft import utils as sft_utils


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

  def test_masked_padding_tokens_do_not_propagate_nan_in_loss_or_gradients(
      self,
  ):
    # Regression test for b/563191139 (paired with MaxText PR #5292):
    # Multiplicative masking (loss * completion_mask) evaluates 0.0 * Inf = NaN
    # in IEEE-754 when masked/padding positions carry non-finite values or
    # singular Jacobians. Using jnp.where(completion_mask > 0, ..., 0.0) severs
    # both forward and reverse-mode autodiff paths on masked positions.
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import common  # pylint: disable=g-import-not-at-top

    class _ToyModel(nnx.Module):

      def __init__(self, *, vocab, dim, rngs):
        self.emb = nnx.Embed(vocab, dim, rngs=rngs)
        self.head = nnx.Linear(dim, vocab, rngs=rngs)

      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        return self.head(self.emb(x)), cache

    model = _ToyModel(vocab=16, dim=8, rngs=nnx.Rngs(42))
    cfg = SimpleNamespace(
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        epsilon_c=None,
        loss_algo='grpo',
        loss_agg_mode='token-mean',
        temperature=1.0,
        kl_loss_mode='low_var_kl',
        kl_clamp_value=None,
        force_compute_kl=False,
    )

    clean_old_logps = jnp.array(
        [[-1.2, -0.8, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]], jnp.float32
    )
    corrupted_old_logps = jnp.array(
        [[-1.2, -0.8, jnp.inf, -jnp.inf], [-0.5, jnp.nan, jnp.inf, -jnp.inf]],
        jnp.float32,
    )
    completion_mask = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], jnp.float32
    )

    clean_ex = common.TrainExample(
        prompt_ids=jnp.array([[0, 3], [0, 4]], jnp.int32),
        prompt_mask=jnp.array([[0, 1], [0, 1]], jnp.int32),
        completion_ids=jnp.array([[5, 6, 0, 0], [7, 0, 0, 0]], jnp.int32),
        completion_mask=completion_mask,
        advantages=jnp.array([1.25, -0.75], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=clean_old_logps,
    )
    corrupted_ex = common.TrainExample(
        prompt_ids=jnp.array([[0, 3], [0, 4]], jnp.int32),
        prompt_mask=jnp.array([[0, 1], [0, 1]], jnp.int32),
        completion_ids=jnp.array([[5, 6, 0, 0], [7, 0, 0, 0]], jnp.int32),
        completion_mask=completion_mask,
        advantages=jnp.array([1.25, -0.75], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=corrupted_old_logps,
    )

    def _loss_scalar(m, ex):
      return algo_core.grpo_loss_fn(
          m, ex, cfg, pad_id=0, eos_id=-1
      ).primary_loss.compute()

    clean_loss, clean_grads = nnx.value_and_grad(_loss_scalar)(model, clean_ex)
    corrupt_loss, corrupt_grads = nnx.value_and_grad(_loss_scalar)(
        model, corrupted_ex
    )

    self.assertTrue(bool(jnp.isfinite(corrupt_loss)))
    np.testing.assert_allclose(corrupt_loss, clean_loss, rtol=1e-5, atol=1e-5)

    for g_corrupt, g_clean in zip(
        jax.tree_util.tree_leaves(corrupt_grads),
        jax.tree_util.tree_leaves(clean_grads),
    ):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_corrupt))))
      np.testing.assert_allclose(g_corrupt, g_clean, rtol=1e-5, atol=1e-5)


class GrpoLooAdvantagesTest(absltest.TestCase):
  """Tests for leave-one-out group-relative advantages."""

  def _reference(self, rewards, num_generations):
    """Transcribes the definition directly, independently of the code."""
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
    # The tolerance is set for float32 against a float64 reference.
    rng = np.random.default_rng(0)
    for g in (2, 3, 4, 8, 16):
      rewards = rng.random(g * 3)
      with self.subTest(num_generations=g):
        np.testing.assert_allclose(
            algo_core.compute_grpo_loo_advantages(jnp.asarray(rewards), g),
            self._reference(rewards, g),
            rtol=1e-3,
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
    # Including a sample in its own baseline scales its advantage by 1 - 1/G.
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

  def test_rewards_far_from_zero_are_still_normalized(self):
    # The spread here is four orders below the rewards themselves, so a
    # variance taken as E[x^2] - E[x]^2 cancels away entirely in float32 and
    # the normalization silently switches itself off.
    # The spread survives in float32 but only to about three digits, hence the
    # loose tolerance; the failure this guards against is a factor of ten.
    rewards = np.array([1000.0, 1000.1, 1000.2, 1000.3])
    np.testing.assert_allclose(
        algo_core.compute_grpo_loo_advantages(jnp.asarray(rewards), 4),
        self._reference(rewards, 4),
        rtol=5e-3,
    )

  def test_a_leave_one_out_set_with_no_spread_keeps_a_raw_advantage(self):
    # Sample 0's two peers scored identically, so its leave-one-out standard
    # deviation is exactly zero and its advantage is left unnormalized.
    out = algo_core.compute_grpo_loo_advantages(
        jnp.array([0.0, 100.0, 100.0]), 3
    )
    np.testing.assert_allclose(out[0], -100.0, rtol=1e-5)
    np.testing.assert_allclose(out[1:], [0.7071, 0.7071], rtol=1e-3)

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
  """Tests for sequence-level loss masking."""

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
    """A dropped sequence leaves the denominator, so the loss is unscaled.

    Checked against the real aggregators for every supported aggregation mode.
    Zeroing a weight instead, as `truncated_importance_weights` does, removes a
    sequence from the numerator only and does scale the loss down.
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

    for mode in (
        'token-mean',
        'sequence-mean-token-mean',
        'sequence-mean-token-scale',
        'seq-mean-token-sum',
    ):
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

  def test_a_fixed_normalizer_cannot_express_a_drop(self):
    # `sequence-mean-token-sum-norm` divides by a fixed factor rather than by
    # the mask, so a masked sequence leaves the numerator but not the
    # denominator. `grpo_loss_fn` refuses the combination for that reason.
    completion_mask = self._mask(rows=4, tokens=5)
    per_token_loss = jnp.full((4, 5), 0.25, dtype=jnp.float32)
    result = algo_core.sequence_loss_mask(
        completion_mask,
        overlong=jnp.array([0.0, 1.0, 1.0, 0.0]),
        mask_overlong=True,
    )
    mode = 'sequence-mean-token-sum-norm'
    full = common.aggregate_loss(per_token_loss, completion_mask, mode)
    masked = common.aggregate_loss(per_token_loss, result.loss_mask, mode)
    self.assertAlmostEqual(
        float(masked.denominator), float(full.denominator), places=5
    )
    self.assertAlmostEqual(
        float(masked.compute()), float(full.compute()) / 2.0, places=5
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


class TokenOutlierStatsTest(absltest.TestCase):
  """Tests for the extreme-value statistics of the per-token disagreement."""

  def test_uniform_small_noise_reports_no_outliers(self):
    log_is = jnp.full((4, 100), 0.08)
    absmax, frac, per_seq = algo_core.token_outlier_stats(
        log_is, jnp.ones((4, 100)), 4.0
    )
    self.assertAlmostEqual(float(absmax), 0.08, places=5)
    self.assertEqual(float(frac), 0.0)
    self.assertEqual(float(per_seq), 0.0)

  def test_one_catastrophic_token_per_sequence_is_counted(self):
    # The case this metric exists for: a defect that fires once per
    # conversation turn. Broad noise is small, but a few tokens are off by
    # tens of nats and dominate the sequence geometric mean.
    log_is = np.full((4, 100), 0.05, dtype=np.float32)
    log_is[:, ::25] = -24.0  # four per sequence, as turn boundaries would be
    absmax, frac, per_seq = algo_core.token_outlier_stats(
        jnp.asarray(log_is), jnp.ones((4, 100)), 4.0
    )
    self.assertAlmostEqual(float(absmax), 24.0, places=4)
    self.assertAlmostEqual(float(frac), 16 / 400, places=6)
    self.assertAlmostEqual(float(per_seq), 4.0, places=5)

  def test_absmean_alone_would_miss_it(self):
    # Guards the premise. Both arrays have a similar mean |log_is|, but only
    # one is pathological -- which is exactly why the mean cannot be the only
    # thing reported.
    spread = jnp.full((1, 100), 0.25)
    spiky = np.zeros((1, 100), dtype=np.float32)
    spiky[0, :1] = -25.0
    spiky = jnp.asarray(spiky)
    mask = jnp.ones((1, 100))
    self.assertAlmostEqual(
        float(jnp.abs(spread).mean()), float(jnp.abs(spiky).mean()), places=6
    )
    self.assertEqual(float(algo_core.token_outlier_stats(spread, mask, 1.0)[2]), 0.0)
    self.assertEqual(float(algo_core.token_outlier_stats(spiky, mask, 1.0)[2]), 1.0)

  def test_masked_tokens_are_ignored(self):
    log_is = jnp.array([[0.0, 0.0, -30.0]])
    mask = jnp.array([[1.0, 1.0, 0.0]])
    absmax, frac, per_seq = algo_core.token_outlier_stats(log_is, mask, 1.0)
    self.assertEqual(float(absmax), 0.0)
    self.assertEqual(float(frac), 0.0)
    self.assertEqual(float(per_seq), 0.0)

  def test_fully_masked_batch_is_neutral(self):
    absmax, frac, per_seq = algo_core.token_outlier_stats(
        jnp.full((2, 3), -40.0), jnp.zeros((2, 3)), 1.0
    )
    self.assertEqual(float(absmax), 0.0)
    self.assertEqual(float(frac), 0.0)
    self.assertEqual(float(per_seq), 0.0)


class SequenceMultProbErrorTest(absltest.TestCase):
  """Tests for the multiplicative probability error gate."""

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
        log_is_raw=jnp.array([[0.0] * 4, [0.5] * 4, [1.5] * 4]),
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
        {'log_is_raw': log_is, 'mult_prob_error_threshold': None},
        {'log_is_raw': None, 'mult_prob_error_threshold': 2.0},
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
        log_is_raw=jnp.full((2, 4), 5.0),  # exp(5) >> threshold for both rows
        mult_prob_error_threshold=2.0,
    )
    self.assertEqual(float(result.mult_prob_error[0]), 0.0)
    np.testing.assert_array_equal(result.sample_mask, [0.0, 0.0])

  def test_mult_prob_error_with_segment_ids(self):
    # Segment 1 (tokens 0-2): exp(0)=1.0
    # Segment 2 (tokens 3-4): exp(0.5)=1.6487
    # Segment 0 (token 5): pad, mask=0
    mask = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    log_is_raw = jnp.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.0]])
    segment_ids = jnp.array([[1, 1, 1, 2, 2, 0]])
    errors = algo_core.sequence_mult_prob_error(
        log_is_raw, mask, segment_ids=segment_ids, num_segments=3
    )
    self.assertEqual(errors.shape, (1, 3))
    np.testing.assert_allclose(errors[0, 0], 0.0)
    np.testing.assert_allclose(errors[0, 1], 1.0, rtol=1e-5)
    np.testing.assert_allclose(errors[0, 2], np.exp(0.5), rtol=1e-5)

  def test_overlong_drops_packed_segment(self):
    # Segment 1 (tokens 0-1): not overlong
    # Segment 2 (tokens 2-3): overlong
    mask = jnp.array([[1.0, 1.0, 1.0, 1.0]])
    segment_ids = jnp.array([[1, 1, 2, 2]])
    overlong = jnp.array([[0.0, 0.0, 1.0, 1.0]])
    result = algo_core.sequence_loss_mask(
        mask,
        overlong=overlong,
        mask_overlong=True,
        segment_ids=segment_ids,
        num_segments=3,
    )
    np.testing.assert_array_equal(result.sample_mask[0], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.loss_mask[0], [1.0, 1.0, 0.0, 0.0])

  def test_gate_drops_packed_segment_over_threshold(self):
    mask = jnp.array([[1.0, 1.0, 1.0, 1.0]])
    segment_ids = jnp.array([[1, 1, 2, 2]])
    log_is_raw = jnp.array([[0.0, 0.0, 1.5, 1.5]])
    result = algo_core.sequence_loss_mask(
        mask,
        log_is_raw=log_is_raw,
        mult_prob_error_threshold=2.0,
        segment_ids=segment_ids,
        num_segments=3,
    )
    np.testing.assert_array_equal(result.sample_mask[0], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.loss_mask[0], [1.0, 1.0, 0.0, 0.0])


class TruncatedImportanceWeightsTest(absltest.TestCase):
  """Tests for the seq-mask-tis importance weights."""

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
    """An infinite log ratio becomes a weight of 0, not of 1.

    `nan_to_num` runs on the weight rather than on the log ratio, so the token
    is discarded instead of being recorded as perfect agreement.
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
    """A rejected sequence stays in the denominator, so the loss scales down.

    The keep-mask multiplies the weights, not the loss mask, which is the
    opposite of what `sequence_loss_mask` does to the sequences it drops.
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

  def test_truncated_importance_weights_with_segment_ids(self):
    # 2 segments packed in row 0:
    # Segment 1 (tokens 0, 1): log_is = 0.0005 -> inside band
    # Segment 2 (tokens 2, 3): log_is = 0.01 -> outside band
    # Segment 0 (token 4): pad
    log_is_raw = jnp.array([[0.0005, 0.0005, 0.01, 0.01, 0.0]])
    completion_mask = jnp.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    segment_ids = jnp.array([[1, 1, 2, 2, 0]])
    num_segments = 3
    sample_mask = jnp.ones((1, num_segments))
    log_is = jnp.nan_to_num(log_is_raw, nan=0.0, posinf=0.0, neginf=0.0)
    geomean, valid = algo_core.sequence_geomean_ratio(
        log_is, completion_mask, segment_ids=segment_ids, num_segments=num_segments
    )
    weights, oob = algo_core.truncated_importance_weights(
        log_is_raw,
        geomean,
        valid,
        sample_mask,
        segment_ids=segment_ids,
        **self._BAND,
    )
    self.assertAlmostEqual(float(oob), 0.5)
    self.assertGreater(float(weights[0, 0]), 0.0)
    self.assertGreater(float(weights[0, 1]), 0.0)
    np.testing.assert_array_equal(weights[0, 2:4], [0.0, 0.0])
    np.testing.assert_array_equal(weights[0, 4], 0.0)


class SamplerIsLengthScalingTest(absltest.TestCase):
  """Tests for the length-scaling diagnostic of the sampler-trainer offset.

  The statistic separates two cases that a batch average cannot: iid token
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
      mean = float(sums[f'{prefix}/logmean_sum']) / count
      observed = float(sums[f'{prefix}/logmean_sq_sum']) / count - mean**2
      iid = float(sums[f'{prefix}/iid_var_sum']) / count
      excess.append(np.sqrt(observed / iid))
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
        # `atol` carries this, not `rtol`. `logmean_sum` adds ~400 signed
        # per-sequence means that largely cancel, leaving a total some two
        # orders smaller than the mass summed, so a relative tolerance on it
        # measures float32 accumulation order rather than additivity. The
        # absolute bound sits ~100x above that noise and ~1000x below the
        # smallest real violation, which would be a whole sequence counted
        # twice or not at all.
        np.testing.assert_allclose(
            float(split[f'{name}/complete/{metric}'])
            + float(split[f'{name}/truncated/{metric}']),
            float(unsplit[f'{name}/complete/{metric}']),
            rtol=1e-5,
            atol=1e-6,
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


class EmptyMicroBatchMetricsTest(absltest.TestCase):
  """Tests for metrics of a micro-batch that sequence masking left empty.

  One micro-batch is exactly one prompt group when
  `train_micro_batch_size == num_generations`, so masking can empty one
  outright. A degenerate value there propagates through the cross-micro-batch
  reducer into the step's number.
  """

  def test_weighted_mean_ignores_an_empty_micro_batch(self):
    # Two micro-batches score 1.06; the third has nothing scored. The correct
    # answer is 1.06, not 2/3 of it.
    scored = sft_utils.WeightedMetric(
        jnp.asarray(2 * 1.06), jnp.asarray(2.0), min_denom=1.0
    )
    empty = sft_utils.WeightedMetric(
        jnp.asarray(0.0), jnp.asarray(0.0), min_denom=1.0
    )
    self.assertAlmostEqual(
        float(common.global_weighted_mean([scored, empty])), 1.06, places=6
    )
    # The reducer this replaced is what produced the sub-1.0 readings.
    self.assertLess(float(common.mean_of_means([scored, empty])), 1.0)

  def test_all_empty_does_not_nan(self):
    empty = sft_utils.WeightedMetric(
        jnp.asarray(0.0), jnp.asarray(0.0), min_denom=1.0
    )
    self.assertTrue(np.isfinite(float(common.global_weighted_mean([empty]))))


class GrpoLossSequenceMaskingTest(absltest.TestCase):
  """Tests for `grpo_loss_fn` with the sequence-level options enabled."""

  def setUp(self):
    super().setUp()
    from flax import nnx  # pylint: disable=g-import-not-at-top

    class _Toy(nnx.Module):

      def __init__(self, *, vocab, dim, rngs):
        self.emb = nnx.Embed(vocab, dim, rngs=rngs)
        self.head = nnx.Linear(dim, vocab, rngs=rngs)

      def __call__(self, x, **kwargs):
        del kwargs
        return self.head(self.emb(x)), None

    self.model = _Toy(vocab=16, dim=8, rngs=nnx.Rngs(0))

  def _config(self, **overrides):
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top

    kwargs = dict(
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        epsilon_c=None,
        loss_algo='grpo',
        loss_agg_mode='token-mean',
        temperature=1.0,
        kl_loss_mode='low_var_kl',
        kl_clamp_value=None,
        force_compute_kl=False,
        overlong_loss_masking=False,
        seq_logprob_error_threshold=None,
        truncated_importance_sampling_type=None,
        truncated_importance_sampling_ratio_min=None,
        truncated_importance_sampling_ratio=None,
        sampler_is_length_buckets=None,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)

  def _example(self, **overrides):
    from tunix.rl import common  # pylint: disable=g-import-not-at-top

    kwargs = dict(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 7, 8]], jnp.int32),
        completion_mask=jnp.ones((2, 3), jnp.float32),
        advantages=jnp.array([1.5, -1.5], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )
    kwargs.update(overrides)
    return common.TrainExample(**kwargs)

  def _loss(self, example, config):
    return algo_core.grpo_loss_fn(
        self.model, example, config, pad_id=0, eos_id=-1
    )

  def test_overlong_masking_without_a_verdict_raises(self):
    with self.assertRaisesRegex(ValueError, 'truncation verdict'):
      self._loss(self._example(), self._config(overlong_loss_masking=True))

  def test_gates_without_rollout_logprobs_raise(self):
    with self.assertRaisesRegex(ValueError, 'per-token log-probabilities'):
      self._loss(
          self._example(), self._config(seq_logprob_error_threshold=2.0)
      )

  def test_dropping_a_sequence_does_not_rescale_the_kl_penalty(self):
    # The KL numerator and the policy-loss numerator are summed and divided by
    # the policy loss's denominator, so both have to be taken over the same
    # tokens. The two sequences here hold the same tokens, so their per-token
    # KL is the same and dropping one must not move the reported mean.
    example = self._example(
        completion_ids=jnp.array([[3, 4, 5], [3, 4, 5]], jnp.int32),
        ref_per_token_logps=jnp.full((2, 3), -1.0, jnp.float32),
        overlong=jnp.array([0.0, 1.0]),
    )
    kept = self._loss(example, self._config(beta=0.5))
    dropped = self._loss(
        example, self._config(beta=0.5, overlong_loss_masking=True)
    )
    self.assertAlmostEqual(
        float(kept.aux_metrics['kl'].compute()),
        float(dropped.aux_metrics['kl'].compute()),
        places=4,
    )

  def test_masking_with_a_fixed_normalizer_raises(self):
    example = self._example(overlong=jnp.array([0.0, 1.0]))
    config = self._config(
        overlong_loss_masking=True,
        loss_agg_mode='sequence-mean-token-sum-norm',
    )
    with self.assertRaisesRegex(ValueError, 'sequence-mean-token-sum-norm'):
      self._loss(example, config)

  def test_importance_weighting_with_a_fixed_normalizer_is_allowed(self):
    # Unlike a mask, the keep-band is meant to scale the loss down, so the
    # fixed normalizer is not a contradiction there.
    example = self._example(
        rollout_per_token_logps=jnp.full((2, 3), -1.0, jnp.float32)
    )
    config = self._config(
        loss_agg_mode='sequence-mean-token-sum-norm',
        truncated_importance_sampling_type='seq-mask-tis',
        truncated_importance_sampling_ratio_min=0.999,
        truncated_importance_sampling_ratio=1.002,
    )
    self.assertIsNotNone(self._loss(example, config))

  def test_an_empty_micro_batch_yields_the_reduction_identity(self):
    """An emptied micro-batch must not win the cross-micro-batch min/max.

    These are pooled over a step's micro-batches with the matching np.min /
    np.max, so the value emitted when nothing survives has to be the identity
    of that reduction. Any finite sentinel would replace the step's real value.
    """
    empty = self._loss(
        self._example(overlong=jnp.ones((2,))),
        self._config(overlong_loss_masking=True),
    ).aux_metrics
    full = self._loss(self._example(), self._config()).aux_metrics

    for name, reducer in (('is_ratio/min', np.min),
                          ('advantage/min', np.min),
                          ('advantage/max', np.max)):
      with self.subTest(metric=name):
        real = float(full[name])
        pooled = float(reducer([real, float(empty[name])]))
        self.assertAlmostEqual(
            pooled, real, places=5,
            msg=f'{name}: an empty micro-batch changed the pooled value')

  def test_a_non_finite_ratio_fails_the_error_gate(self):
    """A token the sampler gave zero probability is the worst disagreement.

    Sanitising the log ratio before the gate would turn it into exp(0) = 1 and
    read as perfect agreement, so the gate sees the raw ratio.
    """
    rollout = jnp.array([[-1.0, -1.0, -1.0], [-1.0, -jnp.inf, -1.0]],
                        jnp.float32)
    aux = self._loss(
        self._example(rollout_per_token_logps=rollout),
        self._config(seq_logprob_error_threshold=1e6),
    ).aux_metrics
    # The threshold is far above anything a finite ratio can reach, so the
    # only sequence that can fail it is the one holding the infinity.
    self.assertAlmostEqual(
        float(aux['sample_mask/kept_frac'].compute()), 0.5, places=5)

  def test_kept_frac_ignores_padding_rows(self):
    # Row 1 carries no scored tokens, as the assembler's trailing rows do not.
    example = self._example(
        completion_mask=jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        overlong=jnp.array([0.0, 0.0]),
    )
    aux = self._loss(
        example, self._config(overlong_loss_masking=True)
    ).aux_metrics
    self.assertAlmostEqual(
        float(aux['sample_mask/kept_frac'].compute()), 1.0, places=5
    )

  def test_options_left_unset_do_not_change_the_loss(self):
    example = self._example()
    base = self._loss(example, self._config())
    # The same config with the option fields absent altogether, as an
    # algo_config predating them would be.
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top

    legacy = SimpleNamespace(
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        epsilon_c=None,
        loss_algo='grpo',
        loss_agg_mode='token-mean',
        temperature=1.0,
        kl_loss_mode='low_var_kl',
        kl_clamp_value=None,
    )
    np.testing.assert_allclose(
        float(base.primary_loss.compute()),
        float(self._loss(example, legacy).primary_loss.compute()),
        rtol=1e-6,
    )
    self.assertAlmostEqual(
        float(base.aux_metrics['sample_mask/kept_frac'].compute()), 1.0
    )

  def test_packed_supports_all_mlperf_flags(self):
    example = self._example(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[7, 3, 4, 5, 7, 6, 7, 8]], jnp.int32),
        completion_mask=jnp.array([[0, 1, 1, 1, 0, 1, 1, 1]], jnp.float32),
        advantages=jnp.array([[0.0, 1.5, 1.5, 1.5, 0.0, -1.5, -1.5, -1.5]], jnp.float32),
        rollout_per_token_logps=jnp.array([[0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0]], jnp.float32),
        overlong=jnp.zeros((1, 8), jnp.float32),
        segment_ids=jnp.array([[1, 1, 1, 1, 2, 2, 2, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]], jnp.int32),
        num_segments=3,
    )
    # With generous threshold and band, both segments pass
    config = self._config(
        overlong_loss_masking=True,
        seq_logprob_error_threshold=20.0,
        truncated_importance_sampling_type='seq-mask-tis',
        truncated_importance_sampling_ratio_min=0.01,
        truncated_importance_sampling_ratio=100.0,
    )
    out = self._loss(example, config)
    self.assertTrue(np.isfinite(float(out.primary_loss.compute())))
    self.assertAlmostEqual(
        float(out.aux_metrics['sample_mask/kept_frac'].compute()), 1.0
    )

    # With strict threshold, both segments are dropped by the gate
    strict_config = self._config(
        overlong_loss_masking=True,
        seq_logprob_error_threshold=1.0,
    )
    strict_out = self._loss(example, strict_config)
    self.assertAlmostEqual(
        float(strict_out.aux_metrics['sample_mask/kept_frac'].compute()), 0.0
    )

  def test_packed_equals_unpacked_with_all_flags(self):
    unpacked = self._example(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 7, 8]], jnp.int32),
        completion_mask=jnp.ones((2, 3), jnp.float32),
        advantages=jnp.array([1.5, -1.5], jnp.float32),
        rollout_per_token_logps=jnp.full((2, 3), -1.0, jnp.float32),
        overlong=jnp.array([0.0, 0.0], jnp.float32),
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )
    packed = self._example(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[7, 3, 4, 5, 7, 6, 7, 8]], jnp.int32),
        completion_mask=jnp.array([[0, 1, 1, 1, 0, 1, 1, 1]], jnp.float32),
        advantages=jnp.array([[0.0, 1.5, 1.5, 1.5, 0.0, -1.5, -1.5, -1.5]], jnp.float32),
        rollout_per_token_logps=jnp.array([[0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0]], jnp.float32),
        overlong=jnp.zeros((1, 8), jnp.float32),
        segment_ids=jnp.array([[1, 1, 1, 1, 2, 2, 2, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]], jnp.int32),
        num_segments=3,
    )
    config = self._config(
        loss_agg_mode='token-mean',
        overlong_loss_masking=True,
        seq_logprob_error_threshold=20.0,
        truncated_importance_sampling_type='seq-mask-tis',
        truncated_importance_sampling_ratio_min=0.01,
        truncated_importance_sampling_ratio=100.0,
    )
    unpacked_loss = float(self._loss(unpacked, config).primary_loss.compute())
    packed_loss = float(self._loss(packed, config).primary_loss.compute())
    np.testing.assert_allclose(packed_loss, unpacked_loss, rtol=1e-5, atol=1e-5)

  def test_packed_equals_unpacked_with_overlong_drop(self):
    unpacked = self._example(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 7, 8]], jnp.int32),
        completion_mask=jnp.ones((2, 3), jnp.float32),
        advantages=jnp.array([1.5, -1.5], jnp.float32),
        rollout_per_token_logps=jnp.full((2, 3), -1.0, jnp.float32),
        overlong=jnp.array([0.0, 1.0], jnp.float32),
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )
    packed = self._example(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[7, 3, 4, 5, 7, 6, 7, 8]], jnp.int32),
        completion_mask=jnp.array([[0, 1, 1, 1, 0, 1, 1, 1]], jnp.float32),
        advantages=jnp.array([[0.0, 1.5, 1.5, 1.5, 0.0, -1.5, -1.5, -1.5]], jnp.float32),
        rollout_per_token_logps=jnp.array([[0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0]], jnp.float32),
        overlong=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], jnp.float32),
        segment_ids=jnp.array([[1, 1, 1, 1, 2, 2, 2, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]], jnp.int32),
        num_segments=3,
    )
    config = self._config(
        loss_agg_mode='token-mean',
        overlong_loss_masking=True,
    )
    unpacked_out = self._loss(unpacked, config)
    packed_out = self._loss(packed, config)
    np.testing.assert_allclose(
        float(packed_out.primary_loss.compute()),
        float(unpacked_out.primary_loss.compute()),
        rtol=1e-5,
        atol=1e-5,
    )
    self.assertAlmostEqual(
        float(packed_out.aux_metrics['sample_mask/kept_frac'].compute()),
        float(unpacked_out.aux_metrics['sample_mask/kept_frac'].compute()),
    )
    self.assertAlmostEqual(
        float(packed_out.aux_metrics['sample_mask/kept_frac'].compute()), 0.5
    )

  def test_packed_kept_frac_ignores_padding_segments(self):
    # Segment 1 has 2 tokens, segment 0 has 2 pad tokens.
    example = self._example(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[3, 4, 0, 0]], jnp.int32),
        completion_mask=jnp.array([[1.0, 1.0, 0.0, 0.0]], jnp.float32),
        advantages=jnp.array([[1.5, 1.5, 0.0, 0.0]], jnp.float32),
        overlong=jnp.zeros((1, 4), jnp.float32),
        segment_ids=jnp.array([[1, 1, 0, 0]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 0, 0]], jnp.int32),
        num_segments=3,
    )
    aux = self._loss(
        example, self._config(overlong_loss_masking=True)
    ).aux_metrics
    self.assertAlmostEqual(
        float(aux['sample_mask/kept_frac'].compute()), 1.0, places=5
    )

  def test_truncated_importance_weights_bounds_and_nan_safety(self):
    log_is_raw = jnp.array([[0.0, 60.0, -jnp.inf, jnp.nan]], dtype=jnp.float32)
    seq_geomean = jnp.array([1.0], dtype=jnp.float32)
    seq_valid = jnp.array([1.0], dtype=jnp.float32)
    sample_mask = jnp.array([1.0], dtype=jnp.float32)
    weights, oob = algo_core.truncated_importance_weights(
        log_is_raw,
        seq_geomean,
        seq_valid,
        sample_mask,
        band_min=0.5,
        band_max=2.0,
    )
    self.assertEqual(float(oob), 0.0)
    self.assertTrue(bool(jnp.all(jnp.isfinite(weights))))
    # Squared weight must remain finite in float32 (no overflow at log_is=60).
    self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.square(weights)))))
    self.assertAlmostEqual(float(weights[0, 0]), 1.0, places=5)
    self.assertAlmostEqual(float(weights[0, 1]), float(jnp.exp(20.0)), places=1)
    self.assertEqual(float(weights[0, 2]), 0.0)
    self.assertEqual(float(weights[0, 3]), 0.0)

  def test_dropped_sequence_severs_backward_gradient_under_extreme_drift(self):
    from flax import nnx  # pylint: disable=g-import-not-at-top

    example = self._example(
        rollout_per_token_logps=jnp.array(
            [[-1.0, -1.0, -1.0], [-1e9, -1e9, -1e9]], jnp.float32
        ),
        old_per_token_logps=jnp.array(
            [[-1.0, -1.0, -1.0], [-1e9, -1e9, -1e9]], jnp.float32
        ),
        ref_per_token_logps=jnp.array(
            [[-1.0, -1.0, -1.0], [1e9, 1e9, 1e9]], jnp.float32
        ),
    )
    for loss_algo in ('grpo', 'gspo-token'):
      with self.subTest(loss_algo=loss_algo):
        config = self._config(
            loss_algo=loss_algo,
            beta=0.01,
            seq_logprob_error_threshold=2.0,
        )
        loss_scalar_fn = lambda model, cfg=config: algo_core.grpo_loss_fn(
            model, example, cfg, pad_id=0, eos_id=-1
        ).primary_loss.compute()
        loss_val, grads = nnx.value_and_grad(loss_scalar_fn)(self.model)
        self.assertTrue(bool(jnp.isfinite(loss_val)))
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for g in grad_leaves:
          self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

  def test_grpo_loss_fn_ignores_non_finite_advantages_on_masked_rows(self):
    from flax import nnx  # pylint: disable=g-import-not-at-top

    example = self._example(
        completion_mask=jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], jnp.float32),
        advantages=jnp.array([1.5, jnp.nan], jnp.float32),
    )
    config = self._config(beta=0.0)
    out = algo_core.grpo_loss_fn(self.model, example, config, pad_id=0, eos_id=-1)
    self.assertTrue(bool(jnp.isfinite(out.primary_loss.compute())))
    self.assertTrue(bool(jnp.isfinite(out.aux_metrics['entropy'].compute())))
    loss_scalar_fn = lambda model: algo_core.grpo_loss_fn(
        model, example, config, pad_id=0, eos_id=-1
    ).primary_loss.compute()
    loss_val, grads = nnx.value_and_grad(loss_scalar_fn)(self.model)
    self.assertTrue(bool(jnp.isfinite(loss_val)))
    for g in jax.tree_util.tree_leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


if __name__ == '__main__':
  absltest.main()


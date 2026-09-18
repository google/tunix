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

import functools

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


class SamplerTrainerDiagnosticsTest(absltest.TestCase):
  """Covers the `sampler_trainer/*` c_t metrics in `grpo_loss_fn`.

  The model emits uniform logits, so the trainer's per-token log-prob is exactly
  -log(vocab) at every position. That pins c_t to a number we can write down,
  which is the only way to test a metric without re-deriving it from the code
  under test.
  """

  _VOCAB = 8

  def _model(self):
    from flax import nnx  # pylint: disable=g-import-not-at-top

    class _Uniform(nnx.Module):

      def __init__(self, vocab):
        # A parameter is required: the loss differentiates the forward pass.
        self.scale = nnx.Param(jnp.zeros(()))

        self.vocab = vocab

      def __call__(self, x, **unused_kwargs):
        logits = jnp.zeros(x.shape + (self.vocab,)) + self.scale[...]
        return logits, None

    return _Uniform(self._VOCAB)

  def _config(self):
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top

    return SimpleNamespace(
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        epsilon_c=None,
        loss_algo='grpo',
        loss_agg_mode='sequence-mean-token-mean',
        temperature=1.0,
        kl_loss_mode='low_var_kl',
        kl_clamp_value=None,
        force_compute_kl=False,
    )

  def _payload(self, sampler_logps):
    from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top

    return datatypes.RLTrainerPayload(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 0, 0]], jnp.int32),
        # Row 1 is mostly padding: masked positions must not reach the metrics.
        completion_mask=jnp.array([[1, 1, 1], [1, 0, 0]], jnp.float32),
        advantages=jnp.array([1.5, 3.0], jnp.float32),
        sampler_per_token_logps=sampler_logps,
    )

  def test_absent_sampler_logps_emits_no_metrics(self):
    aux = algo_core.grpo_loss_fn(
        self._model(), self._payload(None), self._config(), pad_id=0, eos_id=-1
    ).aux_metrics
    self.assertEmpty([k for k in aux if k.startswith('sampler_trainer/')])

  def test_metrics_match_closed_form(self):
    trainer_lp = -np.log(self._VOCAB)
    # log c_t = trainer_lp - sampler_lp. Masked cells are deliberately absurd.
    deltas = np.array([[0.1, -0.3, 0.0], [0.5, 99.0, -99.0]], np.float32)
    sampler = jnp.asarray(trainer_lp - deltas)

    aux = algo_core.grpo_loss_fn(
        self._model(),
        self._payload(sampler),
        self._config(),
        pad_id=0,
        eos_id=-1,
    ).aux_metrics

    live = np.abs([0.1, -0.3, 0.0, 0.5])
    np.testing.assert_allclose(
        aux['sampler_trainer/logp_diff_mean'], live.mean(), atol=1e-5
    )
    np.testing.assert_allclose(
        aux['sampler_trainer/logp_diff_max'], 0.5, atol=1e-5
    )
    np.testing.assert_allclose(
        aux['sampler_trainer/logp_diff_p99'],
        np.quantile(live, 0.99),
        atol=1e-5,
    )
    # Band is [0.8, 1.2]. Live ratios are exp([0.1, -0.3, 0.0, 0.5]) =
    # [1.105, 0.741, 1.000, 1.649]; the 2nd and 4th escape it.
    np.testing.assert_allclose(
        aux['sampler_trainer/would_clip_frac'], 0.5, atol=1e-5
    )

  def test_measurement_does_not_touch_the_loss(self):
    # The whole point of keeping sampler log-probs out of `old_per_token_logps`:
    # the objective must be bit-identical whether or not we are measuring.
    cfg, model = self._config(), self._model()
    sampler = jnp.full((2, 3), -4.2, jnp.float32)

    def run(payload):
      out = algo_core.grpo_loss_fn(model, payload, cfg, pad_id=0, eos_id=-1)
      return float(out.primary_loss.compute())

    self.assertEqual(
        run(self._payload(None)), run(self._payload(sampler))
    )


class GrpoLooAdvantagesTest(absltest.TestCase):
  """Leave-one-out GRPO advantages."""

  def _reference(self, rewards, num_generations):
    """Independent transcription of the intended formula, in matrix form."""
    r = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(r)
    g = num_generations
    for p in range(len(r) // g):
      idx = slice(p * g, (p + 1) * g)
      rr = r[idx]
      others = 1 - np.eye(g)
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
            rtol=1e-3,
            atol=1e-6,
        )

  def test_binary_rewards(self):
    rewards = jnp.array([1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        algo_core.compute_grpo_loo_advantages(rewards, 4),
        self._reference(np.asarray(rewards), 4),
        rtol=1e-5,
    )

  def test_excludes_self_from_the_baseline(self):
    grouped = np.arange(16, dtype=np.float64)
    plain = grouped - grouped.mean()
    loo_mean = (grouped.sum() - grouped) / 15
    np.testing.assert_allclose(
        plain / (grouped - loo_mean), np.full(16, 1 - 1 / 16), rtol=1e-6
    )

  def test_degenerate_groups(self):
    np.testing.assert_array_equal(
        algo_core.compute_grpo_loo_advantages(jnp.array([1.0, 2.0]), 1),
        np.zeros(2),
    )
    np.testing.assert_allclose(
        algo_core.compute_grpo_loo_advantages(jnp.array([1.0, 0.0]), 2),
        [1.0, -1.0],
        rtol=1e-6,
    )

  def test_zero_variance_group_is_not_sharpened(self):
    out = algo_core.compute_grpo_loo_advantages(jnp.full((8,), 0.7), 4)
    np.testing.assert_allclose(out, np.zeros(8), atol=1e-6)

  def test_groups_are_independent(self):
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
    completion_mask = self._mask(rows=4, tokens=5)
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
      self.assertAlmostEqual(
          float(masked.denominator) * 2.0,
          float(full.denominator),
          places=5,
          msg=f'{mode}: expected half the batch to leave the denominator',
      )

  def test_weight_zeroing_does_scale_the_loss_down(self):
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
  """Separating a few catastrophic tokens from broadly noisy ones."""

  def test_uniform_small_noise_reports_no_outliers(self):
    log_is = jnp.full((4, 100), 0.08)
    absmax, n_outliers, n_scored = algo_core.token_outlier_stats(
        log_is, jnp.ones((4, 100))
    )
    self.assertAlmostEqual(float(absmax), 0.08, places=5)
    self.assertEqual(float(n_outliers), 0.0)
    self.assertEqual(float(n_scored), 400.0)

  def test_one_catastrophic_token_per_sequence_is_counted(self):
    log_is = np.full((4, 100), 0.05, dtype=np.float32)
    log_is[:, ::25] = -24.0
    absmax, n_outliers, n_scored = algo_core.token_outlier_stats(
        jnp.asarray(log_is), jnp.ones((4, 100))
    )
    self.assertAlmostEqual(float(absmax), 24.0, places=4)
    self.assertEqual(float(n_outliers), 16.0)
    self.assertEqual(float(n_scored), 400.0)

  def test_absmean_alone_would_miss_it(self):
    spread = jnp.full((1, 100), 0.25)
    spiky = np.zeros((1, 100), dtype=np.float32)
    spiky[0, :1] = -25.0
    spiky = jnp.asarray(spiky)
    mask = jnp.ones((1, 100))
    self.assertAlmostEqual(
        float(jnp.abs(spread).mean()), float(jnp.abs(spiky).mean()), places=6
    )
    self.assertEqual(float(algo_core.token_outlier_stats(spread, mask)[1]), 0.0)
    self.assertEqual(float(algo_core.token_outlier_stats(spiky, mask)[1]), 1.0)

  def test_masked_tokens_are_ignored(self):
    log_is = jnp.array([[0.0, 0.0, -30.0]])
    mask = jnp.array([[1.0, 1.0, 0.0]])
    absmax, n_outliers, n_scored = algo_core.token_outlier_stats(log_is, mask)
    self.assertEqual(float(absmax), 0.0)
    self.assertEqual(float(n_outliers), 0.0)
    self.assertEqual(float(n_scored), 2.0)

  def test_fully_masked_batch_reports_zero_over_zero(self):
    # The denominator must be 0, not 1: an empty micro-batch has to be
    # ignorable by the weighted mean rather than contributing a 0.0 sample.
    absmax, n_outliers, n_scored = algo_core.token_outlier_stats(
        jnp.full((2, 3), -40.0), jnp.zeros((2, 3))
    )
    self.assertEqual(float(absmax), 0.0)
    self.assertEqual(float(n_outliers), 0.0)
    self.assertEqual(float(n_scored), 0.0)

  def test_empty_batches_do_not_dilute_a_pooled_fraction(self):
    # Two micro-batches: one with 1 outlier in 10 scored tokens, one empty.
    # The pooled fraction must be 1/10, not the mean of 0.1 and 0.0.
    spiky = np.zeros((1, 10), dtype=np.float32)
    spiky[0, 0] = -40.0
    _, n_out_a, n_scored_a = algo_core.token_outlier_stats(
        jnp.asarray(spiky), jnp.ones((1, 10))
    )
    _, n_out_b, n_scored_b = algo_core.token_outlier_stats(
        jnp.zeros((1, 10)), jnp.zeros((1, 10))
    )
    pooled = common.global_weighted_mean([
        sft_utils.WeightedMetric(n_out_a, n_scored_a, min_denom=1.0),
        sft_utils.WeightedMetric(n_out_b, n_scored_b, min_denom=1.0),
    ])
    self.assertAlmostEqual(float(pooled), 0.1, places=6)


class SequenceMultProbErrorTest(absltest.TestCase):
  """The log-probability disagreement gate."""

  def test_perfect_agreement_scores_one(self):
    np.testing.assert_allclose(
        algo_core.sequence_mult_prob_error(jnp.zeros((3, 4)), jnp.ones((3, 4))),
        [1.0, 1.0, 1.0],
    )

  def test_absolute_value_prevents_cancellation(self):
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
    self.assertEqual(
        float(
            algo_core.sequence_mult_prob_error(
                jnp.array([[1.0, 1.0]]), jnp.zeros((1, 2))
            )[0]
        ),
        0.0,
    )

  def test_gate_drops_only_sequences_over_threshold(self):
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
    result = algo_core.sequence_loss_mask(
        jnp.ones((2, 4), dtype=jnp.float32),
        overlong=jnp.array([1.0, 0.0]),
        mask_overlong=True,
        log_is=jnp.full((2, 4), 5.0),
        mult_prob_error_threshold=2.0,
    )
    self.assertEqual(float(result.mult_prob_error[0]), 0.0)
    np.testing.assert_array_equal(result.sample_mask, [0.0, 0.0])


class TruncatedImportanceWeightsTest(absltest.TestCase):
  """seq-mask-tis weights, and the denominator behaviour they imply."""

  _BAND = dict(band_min=0.999, band_max=1.002)

  def _call(self, log_is_raw, completion_mask=None, sample_mask=None):
    """Returns `(weights, oob_ratio)`.

    `truncated_importance_weights` returns raw counts; the ratio is formed the
    same way `grpo_loss_fn` forms it, so these tests exercise the real pooling
    path rather than a reimplementation of it.
    """
    weights, n_oob, n_judged = self._call_raw(
        log_is_raw, completion_mask, sample_mask
    )
    oob = sft_utils.WeightedMetric(n_oob, n_judged, min_denom=1.0).compute()
    return weights, oob

  def _call_raw(self, log_is_raw, completion_mask=None, sample_mask=None):
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

  def test_returns_counts_not_a_ratio(self):
    # The counts are the contract: a pre-divided ratio cannot be pooled across
    # micro-batches, and would report 1.0 (100% out of band) for an empty one.
    _, n_oob, n_judged = self._call_raw(
        jnp.array([[0.0] * 4, [0.01] * 4, [0.01] * 4])
    )
    self.assertEqual(float(n_oob), 2.0)
    self.assertEqual(float(n_judged), 3.0)

  def test_empty_batch_judges_nothing(self):
    _, n_oob, n_judged = self._call_raw(
        jnp.zeros((2, 4)), sample_mask=jnp.zeros(2)
    )
    self.assertEqual(float(n_oob), 0.0)
    self.assertEqual(float(n_judged), 0.0)

  def test_agreement_gives_unit_weights_and_no_drops(self):
    weights, oob = self._call(jnp.zeros((3, 4)))
    np.testing.assert_allclose(weights, np.ones((3, 4)), rtol=1e-6)
    self.assertAlmostEqual(float(oob), 0.0)

  def test_sequence_outside_band_is_zeroed_entirely(self):
    weights, oob = self._call(jnp.array([[0.0005] * 4, [0.01] * 4]))
    self.assertGreater(float(weights[0].min()), 0.0)
    np.testing.assert_array_equal(weights[1], np.zeros(4))
    self.assertAlmostEqual(float(oob), 0.5)

  def test_kept_sequences_keep_raw_untruncated_weights(self):
    log_is = jnp.array([[0.5, -0.5, 0.001, 0.001]])
    weights, oob = self._call(log_is)
    self.assertAlmostEqual(float(oob), 0.0)
    np.testing.assert_allclose(
        weights[0], np.exp(np.asarray(log_is[0])), rtol=1e-5
    )

  def test_apply_token_weights_false_acts_purely_as_sequence_mask(self):
    # First sequence is in band (geomean ~ exp(0.0005)), second is out of band.
    log_is_raw = jnp.array([[0.5, -0.499, 0.0005, 0.0005], [0.01] * 4])
    completion_mask = jnp.ones_like(log_is_raw)
    sample_mask = jnp.ones(2)
    log_is = jnp.nan_to_num(log_is_raw, nan=0.0, posinf=0.0, neginf=0.0)
    geomean, valid = algo_core.sequence_geomean_ratio(log_is, completion_mask)
    weights, n_oob, n_judged = algo_core.truncated_importance_weights(
        log_is_raw,
        geomean,
        valid,
        sample_mask,
        apply_token_weights=False,
        **self._BAND,
    )
    self.assertEqual(float(n_oob), 1.0)
    self.assertEqual(float(n_judged), 2.0)
    # In-band sequence gets exactly 1.0 (not exp(0.5)), out-of-band gets 0.0.
    np.testing.assert_array_equal(weights[0], np.ones(4))
    np.testing.assert_array_equal(weights[1], np.zeros(4))

  def test_infinite_ratio_becomes_zero_not_one(self):
    weights, _ = self._call(
        jnp.array([[jnp.inf, 0.0], [-jnp.inf, 0.0], [jnp.nan, 0.0]])
    )
    np.testing.assert_array_equal(weights[:, 0], np.zeros(3))

  def test_geomean_uses_completion_mask_not_padding(self):
    weights, oob = self._call(
        jnp.array([[0.01, 0.01, 99.0, 99.0]]),
        completion_mask=jnp.array([[1.0, 1.0, 0.0, 0.0]]),
    )
    self.assertAlmostEqual(float(oob), 1.0)
    np.testing.assert_array_equal(weights[0, :2], np.zeros(2))

  def test_oob_ratio_normalises_over_valid_sequences_only(self):
    _, oob = self._call(
        jnp.array([[0.01] * 3, [0.0] * 3, [0.0] * 3]),
        sample_mask=jnp.array([0.0, 1.0, 1.0]),
    )
    self.assertAlmostEqual(float(oob), 0.0)

  def test_drop_scales_the_loss_down_unlike_a_loss_mask(self):
    completion_mask = jnp.ones((2, 4), dtype=jnp.float32)
    weights, _ = self._call(
        jnp.array([[0.0] * 4, [0.01] * 4]),
        completion_mask=completion_mask,
    )
    per_token_loss = jnp.full((2, 4), 0.25, dtype=jnp.float32)

    full = common.aggregate_loss(per_token_loss, completion_mask, 'token-mean')
    corrected = common.aggregate_loss(
        per_token_loss * weights, completion_mask, 'token-mean'
    )
    self.assertAlmostEqual(
        float(corrected.compute()), float(full.compute()) / 2.0, places=5
    )
    self.assertAlmostEqual(
        float(corrected.denominator), float(full.denominator), places=5
    )


class SequenceGeomeanRatioTest(absltest.TestCase):

  def test_agreement_is_one(self):
    geomean, valid = algo_core.sequence_geomean_ratio(
        jnp.zeros((2, 3)), jnp.ones((2, 3))
    )
    np.testing.assert_allclose(geomean, [1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(valid, [1.0, 1.0])

  def test_geometric_not_arithmetic(self):
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

  def test_weighted_mean_ignores_an_empty_micro_batch(self):
    scored = sft_utils.WeightedMetric(
        jnp.asarray(2 * 1.06), jnp.asarray(2.0), min_denom=1.0
    )
    empty = sft_utils.WeightedMetric(
        jnp.asarray(0.0), jnp.asarray(0.0), min_denom=1.0
    )
    self.assertAlmostEqual(
        float(common.global_weighted_mean([scored, empty])), 1.06, places=6
    )
    self.assertLess(float(common.mean_of_means([scored, empty])), 1.0)

  def test_all_empty_does_not_nan(self):
    empty = sft_utils.WeightedMetric(
        jnp.asarray(0.0), jnp.asarray(0.0), min_denom=1.0
    )
    self.assertTrue(np.isfinite(float(common.global_weighted_mean([empty]))))


class MaskedExtremumTest(absltest.TestCase):
  """Tests for `algo_core.masked_extremum`.

  The predecessor of this class reimplemented the `jnp.where` guard inside the
  test body and asserted that the test's own code worked. It stayed green while
  `algo_core` shipped the unguarded reduction. Every test here calls the real
  function.
  """

  def test_empty_mask_returns_empty_value_not_inf(self):
    values = jnp.ones((2, 3))
    mask = jnp.zeros((2, 3))

    largest = algo_core.masked_extremum(values, mask, largest=True, empty=0.0)
    smallest = algo_core.masked_extremum(values, mask, largest=False, empty=1.0)

    self.assertTrue(np.isfinite(float(largest)))
    self.assertTrue(np.isfinite(float(smallest)))
    self.assertEqual(float(largest), 0.0)
    self.assertEqual(float(smallest), 1.0)

  def test_partial_mask_ignores_masked_out_entries(self):
    # The dropped row holds both the global max and the global min, so a
    # missing mask would be visible in either direction.
    values = jnp.array([[2.0, 3.0, -1.0], [99.0, -99.0, 0.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    largest = algo_core.masked_extremum(values, mask, largest=True, empty=0.0)
    smallest = algo_core.masked_extremum(values, mask, largest=False, empty=0.0)

    self.assertEqual(float(largest), 3.0)
    self.assertEqual(float(smallest), 2.0)

  def test_full_mask_matches_plain_reduction(self):
    values = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    mask = jnp.ones((2, 3))

    self.assertEqual(
        float(algo_core.masked_extremum(values, mask, largest=True, empty=0.0)),
        float(jnp.max(values)),
    )
    self.assertEqual(
        float(
            algo_core.masked_extremum(values, mask, largest=False, empty=0.0)
        ),
        float(jnp.min(values)),
    )

  def test_empty_value_is_not_clamped_into_the_data_range(self):
    # `empty` is a sentinel, not a floor: it must be returned verbatim even
    # when it sits outside the range of `values`.
    values = jnp.array([[5.0, 6.0]])
    mask = jnp.zeros((1, 2))

    self.assertEqual(
        float(
            algo_core.masked_extremum(values, mask, largest=True, empty=-7.0)
        ),
        -7.0,
    )

  def test_mask_broadcasts_over_rows(self):
    values = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    per_row_mask = jnp.array([[1.0], [0.0]])

    self.assertEqual(
        float(
            algo_core.masked_extremum(
                values, per_row_mask, largest=True, empty=0.0
            )
        ),
        3.0,
    )

  def test_is_jittable(self):
    fn = jax.jit(
        functools.partial(algo_core.masked_extremum, largest=True, empty=0.0)
    )
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    self.assertEqual(float(fn(values, jnp.array([[1.0, 1.0], [0.0, 0.0]]))), 2.0)
    self.assertEqual(float(fn(values, jnp.zeros((2, 2)))), 0.0)


if __name__ == '__main__':
  absltest.main()

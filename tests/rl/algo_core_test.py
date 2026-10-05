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

  def test_compute_advantages_valid_mask_all_valid_matches_legacy(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(
        algo_core.compute_advantages(
            rewards, num_generations=4, valid_mask=np.ones(4, dtype=bool)
        ),
        algo_core.compute_advantages(rewards, num_generations=4),
        rtol=1e-5,
    )

  def test_compute_advantages_valid_mask_excludes_masked(self):
    # The 4th trajectory was masked out, so its artificial 0.0 reward must not
    # drag the group baseline down: mean/std come from [1, 2, 3] only.
    rewards = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    valid_mask = np.array([True, True, True, False])

    advantages = algo_core.compute_advantages(
        rewards, num_generations=4, valid_mask=valid_mask
    )

    np.testing.assert_allclose(
        advantages, [-1.0, 0.0, 1.0, 0.0], rtol=1e-4, atol=1e-4
    )
    # The legacy (unmasked) baseline would have been mean=1.5, so the third
    # trajectory must not look as good as it does without the mask.
    legacy = algo_core.compute_advantages(rewards, num_generations=4)
    self.assertLess(advantages[2], legacy[2])

  def test_compute_advantages_valid_mask_degenerate_group_is_zeroed(self):
    # A sample std (ddof=1) is undefined for a single valid trajectory.
    rewards = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for valid_mask in (
        np.array([True, False, False, False]),
        np.zeros(4, dtype=bool),
    ):
      with self.subTest(num_valid=int(np.sum(valid_mask))):
        advantages = algo_core.compute_advantages(
            rewards, num_generations=4, valid_mask=valid_mask
        )
        np.testing.assert_array_equal(advantages, np.zeros(4, dtype=np.float32))

  def test_compute_rloo_advantages_valid_mask(self):
    # Leave-one-out baseline of the first trajectory averages its valid peers
    # ([2, 3] -> 2.5) rather than all peers ([2, 3, 0] -> 5/3).
    rewards = jnp.array([1.0, 2.0, 3.0, 0.0])
    valid_mask = np.array([True, True, True, False])

    advantages = algo_core.compute_rloo_advantages(
        rewards, num_generations=4, valid_mask=valid_mask
    )

    np.testing.assert_allclose(
        advantages, [-1.5, 0.0, 1.5, 0.0], rtol=1e-4, atol=1e-4
    )

  def test_compute_rloo_advantages_valid_mask_degenerate_group_is_zeroed(self):
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    advantages = algo_core.compute_rloo_advantages(
        rewards,
        num_generations=4,
        valid_mask=np.array([True, False, False, False]),
    )
    np.testing.assert_array_equal(advantages, jnp.zeros(4))

  def test_compute_drgrpo_advantages_valid_mask(self):
    rewards = jnp.array([1.0, 2.0, 3.0, 0.0])
    valid_mask = np.array([True, True, True, False])

    advantages = algo_core.compute_drgrpo_advantages(
        rewards, num_generations=4, valid_mask=valid_mask
    )

    # Valid-only mean is 2.0; DrGRPO skips the std normalization.
    np.testing.assert_allclose(
        advantages, [-1.0, 0.0, 1.0, 0.0], rtol=1e-4, atol=1e-4
    )

  def test_compute_drgrpo_advantages_valid_mask_single_valid_is_zero(self):
    # DrGRPO needs no peer variance, but a lone survivor still sits exactly on
    # its own mean, so the advantage is 0.0 either way.
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    advantages = algo_core.compute_drgrpo_advantages(
        rewards,
        num_generations=4,
        valid_mask=np.array([True, False, False, False]),
    )
    np.testing.assert_allclose(advantages, jnp.zeros(4), atol=1e-6)

  def test_valid_mask_estimators_are_finite_for_empty_group(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    valid_mask = np.zeros(4, dtype=bool)
    for estimator in (
        algo_core.compute_advantages,
        algo_core.compute_rloo_advantages,
        algo_core.compute_drgrpo_advantages,
    ):
      with self.subTest(estimator=estimator.__name__):
        advantages = estimator(
            jnp.asarray(rewards), num_generations=4, valid_mask=valid_mask
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(advantages)))))
        np.testing.assert_array_equal(advantages, np.zeros(4, dtype=np.float32))

  def test_valid_mask_matches_subslice_unmasked_computation(self):
    # Fundamental invariant: for any subset of k >= 2 valid trajectories in a
    # group of G, estimator(rewards, G, valid_mask=mask)[mask] must exactly
    # equal calling the unmasked estimator(rewards[mask], k, valid_mask=None),
    # regardless of extreme values in the masked-out slots.
    rewards = np.array([1.25, -1e6, 3.5, 1e6, -0.75, 2.0], dtype=np.float32)
    masks = [
        np.array([True, False, True, False, False, False]),  # k = 2
        np.array([True, False, True, False, True, False]),   # k = 3
        np.array([True, False, True, False, True, True]),    # k = 4
        np.array([True, True, True, False, True, True]),     # k = 5
    ]
    for estimator in (
        algo_core.compute_advantages,
        algo_core.compute_rloo_advantages,
        algo_core.compute_drgrpo_advantages,
    ):
      for mask in masks:
        k = int(np.sum(mask))
        with self.subTest(estimator=estimator.__name__, k=k):
          masked_adv = np.asarray(
              estimator(
                  jnp.asarray(rewards), num_generations=6, valid_mask=mask
              )
          )
          subslice_adv = np.asarray(
              estimator(
                  jnp.asarray(rewards[mask]), num_generations=k, valid_mask=None
              )
          )
          np.testing.assert_allclose(
              masked_adv[mask], subslice_adv, rtol=1e-5, atol=1e-5
          )
          np.testing.assert_array_equal(
              masked_adv[~mask], np.zeros(6 - k, dtype=np.float32)
          )

  def test_valid_mask_multi_group_heterogeneous_batch(self):
    # 4 groups of G=4 in one batch:
    # - Group 0: 4/4 valid ([1, 2, 3, 4], mean=2.5, sample std=sqrt(5/3))
    # - Group 1: 2/4 valid ([10, _, 20, _], mean=15.0, sample std=sqrt(50))
    # - Group 2: 1/4 valid (degenerate -> all 0.0)
    # - Group 3: 0/4 valid (empty -> all 0.0)
    rewards = np.array(
        [
            1.0, 2.0, 3.0, 4.0,
            10.0, -999.0, 20.0, 999.0,
            5.0, 1.0, 2.0, 3.0,
            1.0, 2.0, 3.0, 4.0,
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([
        True, True, True, True,
        True, False, True, False,
        True, False, False, False,
        False, False, False, False,
    ])
    s0 = np.sqrt(5.0 / 3.0)
    s1 = np.sqrt(50.0)
    expected_by_estimator = [
        (
            algo_core.compute_advantages,
            np.array(
                [
                    -1.5 / s0, -0.5 / s0, 0.5 / s0, 1.5 / s0,
                    -5.0 / s1, 0.0, 5.0 / s1, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
        ),
        (
            algo_core.compute_rloo_advantages,
            np.array(
                [
                    -2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0,
                    -10.0, 0.0, 10.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
        ),
        (
            algo_core.compute_drgrpo_advantages,
            np.array(
                [
                    -1.5, -0.5, 0.5, 1.5,
                    -5.0, 0.0, 5.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
        ),
        (
            algo_core.compute_grpo_loo_advantages,
            np.array(
                [
                    -2.0 / s0, -0.5 * (4.0 / 3.0) / s0, 0.5 * (4.0 / 3.0) / s0, 2.0 / s0,
                    -10.0 / s1, 0.0, 10.0 / s1, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
        ),
    ]
    for estimator, expected in expected_by_estimator:
      with self.subTest(estimator=estimator.__name__):
        batch_adv = np.asarray(
            estimator(
                jnp.asarray(rewards), num_generations=4, valid_mask=valid_mask
            )
        )
        np.testing.assert_allclose(batch_adv, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(batch_adv[8:16], np.zeros(8))
        for g in range(4):
          sl = slice(g * 4, (g + 1) * 4)
          single_adv = np.asarray(
              estimator(
                  jnp.asarray(rewards[sl]),
                  num_generations=4,
                  valid_mask=valid_mask[sl],
              )
          )
          np.testing.assert_allclose(
              batch_adv[sl], single_adv, rtol=1e-5, atol=1e-5
          )

  def test_valid_mask_constant_valid_rewards_ignores_invalid_variance(self):
    # All 3 valid peers scored 1.0, while 1 invalid peer got 0.0.
    # Without valid_mask, the 0.0 creates fake variance and positive advantages
    # for the 1.0 trajectories. With valid_mask, valid variance is 0 -> all 0.0.
    rewards = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    valid_mask = np.array([True, True, True, False])
    for estimator in (
        algo_core.compute_advantages,
        algo_core.compute_rloo_advantages,
        algo_core.compute_drgrpo_advantages,
        algo_core.compute_grpo_loo_advantages,
    ):
      with self.subTest(estimator=estimator.__name__):
        adv = np.asarray(
            estimator(
                jnp.asarray(rewards), num_generations=4, valid_mask=valid_mask
            )
        )
        np.testing.assert_allclose(
            adv, np.zeros(4, dtype=np.float32), atol=1e-6
        )

  def test_grpo_loo_removes_self_inclusion_attenuation_and_matches_relation_to_grpo(
      self,
  ):
    """Verify A_i^{grpo-loo} = (r_i - mean_{-i}) / (std + 1e-6) = (k/(k-1)) * A_i^{grpo}."""
    # k=2 binary rewards [1, 0]:
    #   loo_mean = [0, 1], r - loo_mean = [1, -1]
    #   sample std (ddof=1) of [1, 0] = sqrt((0.25 + 0.25) / 1) = sqrt(0.5)
    #   Plain GRPO gives (r - 0.5) / sqrt(0.5) = [1/sqrt(2), -1/sqrt(2)] (~0.7071),
    #   attenuated by (k-1)/k = 1/2 relative to the LOO residual.
    #   GRPO-LOO gives [1 / sqrt(0.5), -1 / sqrt(0.5)] = [sqrt(2), -sqrt(2)].
    rewards_k2 = np.array([1.0, 0.0], dtype=np.float32)
    adv_loo_k2 = algo_core.compute_grpo_loo_advantages(
        rewards_k2, num_generations=2
    )
    adv_grpo_k2 = algo_core.compute_advantages(rewards_k2, num_generations=2)
    expected_k2 = np.array(
        [1.0 / (np.sqrt(0.5) + 1e-6), -1.0 / (np.sqrt(0.5) + 1e-6)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(adv_loo_k2, expected_k2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        adv_loo_k2, 2.0 * adv_grpo_k2, rtol=1e-5, atol=1e-5
    )

    # k=4 arbitrary rewards: verify exact LOO formula and (k/(k-1)) = 4/3 ratio.
    rewards_k4 = np.array([1.0, 2.0, 4.0, 9.0, -3.0, 0.0, 3.0, 12.0], np.float32)
    adv_loo_k4 = algo_core.compute_grpo_loo_advantages(
        rewards_k4, num_generations=4
    )
    adv_grpo_k4 = algo_core.compute_advantages(rewards_k4, num_generations=4)
    np.testing.assert_allclose(
        adv_loo_k4, (4.0 / 3.0) * adv_grpo_k4, rtol=1e-5, atol=1e-5
    )
    # Sum of advantages within each group is zero.
    np.testing.assert_allclose(
        adv_loo_k4.reshape(2, 4).sum(axis=-1), [0.0, 0.0], atol=1e-5
    )

  def test_grpo_loo_valid_mask_excludes_invalid_from_loo_mean_and_std(self):
    """Invalid trajectories must not enter peer LOO means or group std."""
    # Group of G=4 with 2 valid peers (rewards 10.0 and 20.0) and 2 invalid
    # peers carrying extreme garbage. The k=2 valid slice must match running
    # compute_grpo_loo_advantages on [10.0, 20.0] with G=2.
    rewards_4 = np.array([10.0, -999.0, 20.0, 1e6], dtype=np.float32)
    valid_mask_4 = np.array([True, False, True, False])
    adv_4 = algo_core.compute_grpo_loo_advantages(
        rewards_4, num_generations=4, valid_mask=valid_mask_4
    )
    adv_2 = algo_core.compute_grpo_loo_advantages(
        np.array([10.0, 20.0], dtype=np.float32), num_generations=2
    )
    self.assertEqual(float(adv_4[1]), 0.0)
    self.assertEqual(float(adv_4[3]), 0.0)
    np.testing.assert_allclose(
        [adv_4[0], adv_4[2]], adv_2, rtol=1e-5, atol=1e-5
    )
    # And for k_valid=3 out of G=4, ratio to masked GRPO is 3/2 (not 4/3).
    rewards_3v = np.array([1.0, 2.0, 6.0, -500.0], dtype=np.float32)
    mask_3v = np.array([True, True, True, False])
    adv_loo_3v = algo_core.compute_grpo_loo_advantages(
        rewards_3v, num_generations=4, valid_mask=mask_3v
    )
    adv_grpo_3v = algo_core.compute_advantages(
        rewards_3v, num_generations=4, valid_mask=mask_3v
    )
    np.testing.assert_allclose(
        adv_loo_3v, (3.0 / 2.0) * adv_grpo_3v, rtol=1e-5, atol=1e-5
    )
    self.assertEqual(float(adv_loo_3v[3]), 0.0)

  def test_grpo_loo_degenerate_groups_and_zero_variance(self):
    # num_generations < 2 -> all zeros.
    adv_g1 = algo_core.compute_grpo_loo_advantages(
        np.array([5.0, -2.0], dtype=np.float32), num_generations=1
    )
    np.testing.assert_array_equal(adv_g1, np.zeros(2, dtype=np.float32))

    # 1 valid and 0 valid in G=4 -> all zeros, no NaN/Inf.
    rewards = np.array([42.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0], np.float32)
    mask = np.array([True, False, False, False, False, False, False, False])
    adv = algo_core.compute_grpo_loo_advantages(
        rewards, num_generations=4, valid_mask=mask
    )
    self.assertTrue(np.all(np.isfinite(adv)))
    np.testing.assert_array_equal(adv, np.zeros(8, dtype=np.float32))

  def test_grpo_loo_registered_in_function_registry(self):
    from tunix.rl import function_registry  # pylint: disable=g-import-not-at-top

    fn = function_registry.get_advantage_estimator('grpo-loo')
    self.assertIs(fn, algo_core.compute_grpo_loo_advantages)

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

  def test_fused_sampler_trainer_agreement_matches_two_pass_reference(self):
    """Fused in-loss agreement/TIS/masking produces identical loss, grads, and metrics to 2-pass."""
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import common  # pylint: disable=g-import-not-at-top

    class _ToyModel(nnx.Module):

      def __init__(self, *, vocab_size, rngs):
        self.emb = nnx.Embed(vocab_size, 8, rngs=rngs)
        self.head = nnx.Linear(8, vocab_size, rngs=rngs)

      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        del segment_ids, positions, attention_mask
        return self.head(self.emb(x)), cache

    model = _ToyModel(vocab_size=16, rngs=nnx.Rngs(42))
    prompt_ids = jnp.array([[1, 2], [3, 4]], jnp.int32)
    prompt_mask = jnp.ones_like(prompt_ids, jnp.int32)
    completion_ids = jnp.array([[5, 6, 7], [8, 9, 10]], jnp.int32)
    completion_mask = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], jnp.float32)
    advantages = jnp.array([1.2, -0.8], jnp.float32)
    rollout_logps = jnp.array(
        [[-1.5, -2.0, -1.8], [-5.0, -4.5, 0.0]], jnp.float32
    )

    graphdef, state = nnx.split(model)
    trainer_logps = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=0,
        eos_id=-1,
        stop_gradient=True,
        return_entropy=False,
    )

    for sampler_is, seq_err_thresh in [
        (None, None),
        ('token', None),
        (None, 2.0),
        ('token', 2.0),
    ]:
      with self.subTest(sampler_is=sampler_is, seq_err_thresh=seq_err_thresh):
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
            use_rollout_logps=True,
            sampler_is=sampler_is,
            sampler_is_threshold=2.0,
            seq_logprob_error_threshold=seq_err_thresh,
        )
        # 1. Fused single-pass example (raw rollout_logps)
        ex_fused = common.TrainExample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            advantages=advantages,
            ref_per_token_logps=None,
            old_per_token_logps=rollout_logps,
        )
        # 2. Two-pass precomputed reference example
        ref_metrics, ref_is_weights, ref_filtered_mask = (
            common.sampler_trainer_agreement(
                rollout_logps,
                trainer_logps,
                completion_mask,
                sampler_is=sampler_is,
                sampler_is_threshold=2.0,
                seq_logprob_error_threshold=seq_err_thresh,
            )
        )
        ref_old_logps = (
            trainer_logps
            if (sampler_is == 'token' or seq_err_thresh is not None)
            else rollout_logps
        )
        ex_ref = common.TrainExample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=ref_filtered_mask,
            advantages=advantages,
            ref_per_token_logps=None,
            old_per_token_logps=ref_old_logps,
            sampler_is_weights=ref_is_weights,
            sampler_agreement_applied=True,
        )

        def _loss_and_aux(m, ex):
          out = algo_core.grpo_loss_fn(m, ex, cfg, pad_id=0, eos_id=-1)
          return out.primary_loss.compute(), out.aux_metrics

        (loss_fused, aux_fused), grads_fused = nnx.value_and_grad(
            _loss_and_aux, has_aux=True
        )(model, ex_fused)
        (loss_ref, aux_ref), grads_ref = nnx.value_and_grad(
            _loss_and_aux, has_aux=True
        )(model, ex_ref)

        np.testing.assert_allclose(loss_fused, loss_ref, rtol=1e-6, atol=1e-6)
        for g_f, g_r in zip(
            jax.tree_util.tree_leaves(grads_fused),
            jax.tree_util.tree_leaves(grads_ref),
        ):
          np.testing.assert_allclose(g_f, g_r, rtol=1e-6, atol=1e-6)
        for k, (val, _) in ref_metrics.items():
          self.assertIn(k, aux_fused)
          fused_val = getattr(aux_fused[k], 'compute', lambda: aux_fused[k])()
          ref_val = getattr(val, 'compute', lambda: val)()
          np.testing.assert_allclose(
              float(np.asarray(fused_val)),
              float(np.asarray(ref_val)),
              rtol=1e-5,
              atol=1e-5,
          )
        for k in ('reduced_pg_loss', 'is_ratio/mean', 'ppo_kl'):
          np.testing.assert_allclose(
              float(np.asarray(getattr(aux_fused[k], 'compute', lambda: aux_fused[k])())),
              float(np.asarray(getattr(aux_ref[k], 'compute', lambda: aux_ref[k])())),
              rtol=1e-6,
              atol=1e-6,
          )


if __name__ == '__main__':
  absltest.main()

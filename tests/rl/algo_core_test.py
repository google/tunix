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
    # - Group 0: 4/4 valid
    # - Group 1: 2/4 valid (with extreme garbage in invalid slots)
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
    for estimator in (
        algo_core.compute_advantages,
        algo_core.compute_rloo_advantages,
        algo_core.compute_drgrpo_advantages,
    ):
      with self.subTest(estimator=estimator.__name__):
        batch_adv = np.asarray(
            estimator(
                jnp.asarray(rewards), num_generations=4, valid_mask=valid_mask
            )
        )
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
        np.testing.assert_array_equal(batch_adv[8:16], np.zeros(8))

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


if __name__ == '__main__':
  absltest.main()

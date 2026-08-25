# Copyright 2025 Google LLC
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
import optax
from tunix.sft import utils


class UtilsTest(absltest.TestCase):

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array([
        [True, True, True, True],
        [True, True, True, False],
        [False, True, True, False],
    ])
    attn_mask = utils.make_causal_attn_mask(input_mask)
    expected_value = jnp.array([
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, False],
        ],
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
            [False, True, True, False],
        ],
    ])
    np.testing.assert_allclose(attn_mask, expected_value)

  def test_build_positions_from_mask(self):
    input_mask = jnp.array(
        [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]]
    )
    positions = utils.build_positions_from_mask(input_mask)
    expected_value = jnp.array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 1, 2, 2],
        [0, 0, 1, 1],
    ])
    np.testing.assert_array_equal(positions, expected_value)


class StableGlobalNormTest(absltest.TestCase):

  @staticmethod
  def _p63_env(*, frozenlake: bool = False) -> dict[str, str]:
    if frozenlake:
      profile_file = (
          "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
      )
      profile = "qwen3-8b-dp8-tp8-frozenlake-v1-hp"
      workload = "frozenlake-dp8-tp8"
      warn_only = {"CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0"}
    else:
      profile_file = (
          "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env"
      )
      profile = "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp"
      workload = "gsm8k"
      warn_only = {"CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0"}
    return {
        "CANON_P63_OVERFLOW_SAFE_CLIP": "1",
        "CANON_PROFILE_FILE": profile_file,
        "CANON_PROFILE": profile,
        "CANON_P32_WORKLOAD": workload,
        "CANON_V1_HP_FULL": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "full",
        "CANON_P33_NO_COMMIT": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
        **warn_only,
    }

  def test_finite_large_gradient_does_not_overflow(self):
    gradient = {"x": jnp.asarray([1.0e20, -1.0e20], jnp.float32)}
    naive = jnp.sqrt(jnp.sum(jnp.square(gradient["x"])))
    self.assertTrue(np.isinf(float(naive)))

    norm = float(utils.stable_global_norm(gradient))
    self.assertTrue(np.isfinite(norm))
    self.assertAlmostEqual(norm / 1.0e20, np.sqrt(2.0), places=5)

  def test_nonfinite_gradient_remains_nonfinite(self):
    for value in (jnp.inf, jnp.nan):
      with self.subTest(value=value):
        norm = float(utils.stable_global_norm({"x": jnp.asarray([value])}))
        self.assertFalse(np.isfinite(norm))

  def test_hybrid_clip_stock_finite_path_is_byte_exact(self):
    for dtype in (jnp.float32, jnp.bfloat16):
      for values in ([0.25, -0.5, 0.75], [2.0, -3.0, 4.0]):
        with self.subTest(dtype=dtype, values=values):
          gradient = {"x": jnp.asarray(values, dtype=dtype)}
          stock = optax.clip_by_global_norm(1.0)
          hybrid = utils.overflow_safe_clip_by_global_norm(1.0)
          stock_update, _ = stock.update(
              gradient, stock.init(gradient)
          )
          hybrid_update, _ = hybrid.update(
              gradient, hybrid.init(gradient)
          )
          np.testing.assert_array_equal(
              np.asarray(hybrid_update["x"]),
              np.asarray(stock_update["x"]),
          )
          stats = utils.hybrid_global_norm_stats(
              gradient, max_norm=1.0
          )
          self.assertTrue(bool(stats["naive_norm_finite"]))
          self.assertFalse(bool(stats["fallback_used"]))

  def test_hybrid_clip_overflow_matches_fp64_oracle(self):
    gradient = {"x": jnp.asarray([1.0e20, -1.0e20], jnp.float32)}
    stock = optax.clip_by_global_norm(1.0)
    hybrid = utils.overflow_safe_clip_by_global_norm(1.0)
    stock_update, _ = stock.update(gradient, stock.init(gradient))
    hybrid_update, _ = hybrid.update(gradient, hybrid.init(gradient))
    np.testing.assert_array_equal(
        np.asarray(stock_update["x"]), np.zeros((2,), np.float32)
    )
    oracle_input = np.asarray(gradient["x"], dtype=np.float64)
    oracle = oracle_input / np.linalg.norm(oracle_input)
    np.testing.assert_allclose(
        np.asarray(hybrid_update["x"]), oracle, rtol=2e-6, atol=0.0
    )
    stats = utils.hybrid_global_norm_stats(gradient, max_norm=1.0)
    self.assertTrue(bool(stats["all_finite"]))
    self.assertFalse(bool(stats["naive_norm_finite"]))
    self.assertTrue(bool(stats["fallback_used"]))
    self.assertGreater(float(stats["clip_factor"]), 0.0)

  def test_hybrid_clip_does_not_sanitize_nonfinite_tree(self):
    transform = utils.overflow_safe_clip_by_global_norm(1.0)
    for value in (jnp.inf, jnp.nan):
      with self.subTest(value=value):
        gradient = {"x": jnp.asarray([value], jnp.float32)}
        update, _ = transform.update(
            gradient, transform.init(gradient)
        )
        self.assertFalse(np.all(np.isfinite(np.asarray(update["x"]))))
        stats = utils.hybrid_global_norm_stats(gradient, max_norm=1.0)
        self.assertFalse(bool(stats["all_finite"]))
        self.assertFalse(bool(stats["fallback_used"]))

  def test_p63_context_is_exact_and_default_off(self):
    self.assertIsNone(utils.canonical_overflow_safe_clip_max_norm({}))
    self.assertIsNone(utils.canonical_overflow_safe_clip_max_norm({
        "CANON_P63_OVERFLOW_SAFE_CLIP": "0"
    }))
    self.assertEqual(
        utils.canonical_overflow_safe_clip_max_norm(self._p63_env()), 1.0
    )
    self.assertEqual(
        utils.canonical_overflow_safe_clip_max_norm(
            self._p63_env(frozenlake=True)
        ),
        100.0,
    )
    for key, value in (
        ("CANON_P63_OVERFLOW_SAFE_CLIP", ""),
        ("CANON_PROFILE", "foreign-profile"),
        ("CANON_P33_NO_COMMIT", "1"),
        ("CANON_VLLM_ENABLE_PREFIX_CACHING", "1"),
    ):
      with self.subTest(key=key):
        env = self._p63_env()
        env[key] = value
        with self.assertRaisesRegex(ValueError, "P63|OVERFLOW_SAFE_CLIP"):
          utils.canonical_overflow_safe_clip_max_norm(env)

  def test_numeric_stats_distinguish_finite_huge_from_nonfinite(self):
    gradient = {"x": jnp.asarray([1.0e20, -1.0e20], jnp.float32)}
    receipt = utils.tree_numeric_receipt(gradient)
    self.assertTrue(receipt["all_finite"])
    self.assertFalse(receipt["naive_norm_finite"])
    self.assertAlmostEqual(receipt["max_abs"] / 1.0e20, 1.0, places=6)
    self.assertIsNone(receipt["first_nonfinite"])

    nonfinite = utils.tree_numeric_receipt({
        "good": jnp.asarray([1.0], jnp.float32),
        "bad": jnp.asarray([jnp.nan], jnp.float32),
    })
    self.assertFalse(nonfinite["all_finite"])
    self.assertIsNotNone(nonfinite["first_nonfinite"])
    self.assertIn("bad", nonfinite["first_nonfinite"]["path"])

  def test_scaled_numeric_stats_expose_multiplier(self):
    gradient = {"x": jnp.asarray([8.0, -4.0], jnp.float32)}
    stats = utils.scaled_tree_numeric_stats(
        gradient, jnp.asarray(1.0 / 16.0, jnp.float32)
    )
    receipt = utils.tree_numeric_receipt(gradient, stats=stats)
    self.assertAlmostEqual(receipt["max_abs"], 0.5)
    self.assertAlmostEqual(
        receipt["stable_norm"], np.sqrt(0.5**2 + 0.25**2), places=6
    )

  def test_ranked_numeric_stats_identify_rank_and_leaf(self):
    tree = {
        "a": jnp.asarray([[1.0], [2.0]], jnp.float32),
        "b": jnp.asarray([[3.0], [jnp.inf]], jnp.float32),
    }
    receipt = utils.tree_numeric_receipt(tree, ranked=True)
    self.assertFalse(receipt["all_finite"])
    self.assertEqual(receipt["rank_count"], 2)
    self.assertEqual(receipt["first_nonfinite_rank"]["rank"], 1)
    self.assertIn("b", receipt["first_nonfinite_rank"]["path"])


class WeightedMetricTest(absltest.TestCase):
  """Isolated tests for WeightedMetric's deferred division and safeguards.

  WeightedMetric stores an unreduced sum and a denominator and only divides in
  compute(). compute_scale() builds 1 / denominator with three safeguards, in
  order: add eps (if set), clamp to min_denom (if set), then a zero-guard that
  maps denominator == 0 to a 0.0 scale (so an empty batch contributes nothing
  and does not poison gradients with NaN/Inf).
  """

  def _compute(self, sum_val, denom, **kwargs):
    metric = utils.WeightedMetric(
        jnp.array(sum_val, dtype=jnp.float32),
        jnp.array(denom, dtype=jnp.float32),
        **kwargs,
    )
    return float(metric.compute())

  def test_basic_compute(self):
    metric = utils.WeightedMetric(
        jnp.array(6.0, dtype=jnp.float32), jnp.array(3.0, dtype=jnp.float32)
    )
    self.assertAlmostEqual(float(metric.compute()), 2.0, places=5)
    self.assertAlmostEqual(float(metric.compute_scale()), 1.0 / 3.0, places=5)

  def test_zero_denominator_is_safe(self):
    metric = utils.WeightedMetric(
        jnp.array(5.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    )
    scale = float(metric.compute_scale())
    value = float(metric.compute())
    self.assertEqual(scale, 0.0)
    self.assertEqual(value, 0.0)
    self.assertFalse(np.isnan(value) or np.isinf(value))

  def test_eps_is_negligible_for_nonzero_denominator(self):
    self.assertAlmostEqual(self._compute(6.0, 3.0, eps=1e-6), 2.0, places=4)

  def test_eps_bypasses_zero_guard(self):
    # With eps set, denominator == 0 becomes eps (not 0), so the zero-guard is
    # NOT triggered: the result is sum / eps (a large, finite number), not 0.
    # eps prevents NaN, it does not zero out empty batches.
    value = self._compute(6.0, 0.0, eps=1e-6)
    self.assertAlmostEqual(value, 6.0 / 1e-6, delta=1.0)
    self.assertFalse(np.isinf(value))
    self.assertNotEqual(value, 0.0)

  def test_min_denom_clamps_small_denominator(self):
    self.assertAlmostEqual(self._compute(6.0, 1.0, min_denom=3.0), 2.0, places=5)

  def test_min_denom_noop_when_denominator_is_larger(self):
    self.assertAlmostEqual(self._compute(6.0, 5.0, min_denom=3.0), 1.2, places=5)

  def test_min_denom_acts_as_zero_safeguard(self):
    self.assertAlmostEqual(self._compute(6.0, 0.0, min_denom=2.0), 3.0, places=5)

  def test_eps_and_min_denom_applied_in_order(self):
    # denominator 0 -> +eps (1e-6) -> max(1e-6, 2.0) = 2.0 -> 6 / 2 = 3.0.
    value = self._compute(6.0, 0.0, eps=1e-6, min_denom=2.0)
    self.assertAlmostEqual(value, 3.0, places=5)

  def test_gradient_is_finite_at_zero_denominator(self):
    # The zero-guard uses a double jnp.where so the backward pass stays finite:
    # gradient of the numerator at denominator == 0 is exactly 0, not NaN.
    def loss(sum_val):
      metric = utils.WeightedMetric(sum_val, jnp.array(0.0, dtype=jnp.float32))
      return metric.compute()

    grad = float(jax.grad(loss)(jnp.array(5.0, dtype=jnp.float32)))
    self.assertEqual(grad, 0.0)
    self.assertFalse(np.isnan(grad))

  def test_is_a_pytree_with_two_dynamic_leaves(self):
    # sum and denominator are dynamic leaves; eps and min_denom are static.
    metric = utils.WeightedMetric(
        jnp.array(6.0, dtype=jnp.float32),
        jnp.array(3.0, dtype=jnp.float32),
        eps=1e-8,
        min_denom=1.0,
    )
    leaves = jax.tree_util.tree_leaves(metric)
    self.assertLen(leaves, 2)


if __name__ == '__main__':
  absltest.main()

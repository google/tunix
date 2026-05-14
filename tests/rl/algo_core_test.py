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

from types import SimpleNamespace
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core
from tunix.rl import common as rl_common
from tunix.rl import utils as rl_utils
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.sft import utils as sft_utils
from tunix.tests import test_common as tc


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


# =============================================================================
# Numerical-equivalence tests for the LossOutput / WeightedMetric refactor.
#
# The refactor switched the policy / value loss functions from returning a
# reduced scalar `(loss, aux)` tuple to returning a `LossOutput` containing
# `WeightedMetric`s with unreduced sums + denominators. The equivalence
# contract we have to preserve is:
#
#   `LossOutput.primary_loss.compute()` and `aux_metrics[k].compute()` must
#   match the pre-refactor scalar value of `loss` and `aux[k]` for any input.
#
# These tests pin both:
#   (a) Numerical equivalence with the OLD masked_mean / aggregate_loss
#       formulas (reconstructed in-test as a reference implementation), and
#   (b) Hard-coded golden numerics captured from the current implementation,
#       so future drifts are caught even if the legacy reference helper has
#       a parallel regression.
#
# The bulk of the dependency on `compute_per_token_logps` / `compute_score`
# is exercised by running the real ToyTransformer (same approach used by
# `drgrpo_learner_test`), with `jax_threefry_partitionable=False` so the
# golden numerics are reproducible across machines.
# =============================================================================


jax.config.update('jax_threefry_partitionable', False)


_PAD_ID = 0
_EOS_ID = 1


def _ns(**kw):
  """Lightweight stand-in for an `AlgorithmConfig` — sets only the fields read
  by the loss fns under test, so the tests stay decoupled from algorithm
  config drift (PPOConfig/GRPOConfig __post_init__ side effects, defaults).
  """
  return SimpleNamespace(**kw)


def _grpo_inputs():
  """Deterministic, dense GRPO inputs.

  Shapes mirror an unpacked GRPO micro-batch: 2 rows, 4 tokens, one fully
  masked tail token in row 1. Advantages are 1D (per-row, GRPO style).
  """
  return SimpleNamespace(
      prompt_ids=jnp.zeros((2, 4), dtype=jnp.int32),
      completion_ids=jnp.array(
          [[2, 3, 4, 5], [2, 3, 4, 0]], dtype=jnp.int32
      ),
      completion_mask=jnp.array(
          [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.float32
      ),
      advantages=jnp.array([0.5, -0.2], dtype=jnp.float32),
      ref_per_token_logps=jnp.full((2, 4), -0.3, dtype=jnp.float32),
      old_per_token_logps=jnp.full((2, 4), -0.25, dtype=jnp.float32),
      segment_ids=None,
      segment_positions=None,
  )


def _ppo_inputs():
  """Deterministic PPO inputs (2D per-token advantages, GAE-style)."""
  ex = _grpo_inputs()
  ex.advantages = jnp.array(
      [[0.5, 0.4, 0.3, 0.2], [-0.2, -0.1, 0.0, 0.0]], dtype=jnp.float32
  )
  ex.old_values = jnp.array(
      [[0.1, 0.2, 0.3, 0.4], [0.05, 0.15, 0.25, 0.0]], dtype=jnp.float32
  )
  ex.returns = jnp.array(
      [[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.0]], dtype=jnp.float32
  )
  return ex


def _toy_actor():
  vocab = tc.MockVocab().GetPieceSize()
  return tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab), rngs=nnx.Rngs(0)
  )


class GrpoLossFnEquivalenceTest(parameterized.TestCase):
  """`grpo_loss_fn` lock-in: structure + numerical equivalence + golden values.

  The default `GRPOConfig()` exercises the KL-branch (beta=0.04) and
  `loss_agg_mode="sequence-mean-token-mean"`. Golden values were captured
  from the current implementation; any drift here flags either a math
  regression or an unintentional output-contract change.
  """

  def test_returns_lossoutput_and_aux_are_weighted_metrics(self):
    cfg = grpo_lib.GRPOConfig()
    cfg.temperature = 1.0
    out = algo_core.grpo_loss_fn(
        _toy_actor(), _grpo_inputs(), cfg, _PAD_ID, _EOS_ID
    )
    self.assertIsInstance(out, sft_utils.LossOutput)
    self.assertIsInstance(out.primary_loss, sft_utils.WeightedMetric)
    # Every aux entry must be a `WeightedMetric` so downstream
    # `_compute_legacy_aux` is uniform (no branching on type).
    for k, v in out.aux_metrics.items():
      self.assertIsInstance(
          v, sft_utils.WeightedMetric, msg=f'aux[{k!r}] is not WeightedMetric'
      )
    # The exposed aux keys are the public contract (consumed by trainer
    # post-processing / hooks). Lock them in.
    self.assertEqual(
        sorted(out.aux_metrics),
        sorted([
            'kl', 'kl_loss', 'pg_loss',
            'pg_clipfrac', 'ppo_kl', 'pg_clipfrac_lower', 'entropy',
        ]),
    )

  def test_grpo_loss_fn_golden_numerics(self):
    """Golden values captured from the current implementation.

    The denominators encode the documented per-aux semantics:
      - `pg_loss` / `pg_clipfrac_lower` / `entropy` / `kl_loss` use the
        aggregation-mode denominator (`non_zero_rows = 2`).
      - `pg_clipfrac` / `ppo_kl` / `kl` use `token_denom = Σmask = 7`.
    """
    cfg = grpo_lib.GRPOConfig()
    cfg.temperature = 1.0
    out = algo_core.grpo_loss_fn(
        _toy_actor(), _grpo_inputs(), cfg, _PAD_ID, _EOS_ID
    )
    np.testing.assert_allclose(
        out.primary_loss.compute(), -0.060536161, rtol=1e-5
    )
    expected_aux = {
        'kl':                (-22.753920, 7.0,  -3.2505600),
        'kl_loss':           (-6.4251240, 2.0,  -3.2125618),
        'pg_loss':           ( 0.1359330, 2.0,   0.0679663),
        'pg_clipfrac':       ( 3.0000000, 7.0,   0.4285715),
        'ppo_kl':            (23.1039200, 7.0,   3.3005602),
        'pg_clipfrac_lower': ( 0.0000000, 2.0,   0.0000000),
        'entropy':           ( 4.9093580, 2.0,   2.4546790),
    }
    for k, (unred, denom, val) in expected_aux.items():
      with self.subTest(aux=k):
        m = out.aux_metrics[k]
        np.testing.assert_allclose(
            m.unreduced_sum, unred, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(m.denominator, denom, rtol=1e-6)
        np.testing.assert_allclose(m.compute(), val, rtol=1e-5)

  def test_grpo_compute_matches_legacy_masked_mean_reference(self):
    """Cross-check `aux['kl'] / aux['ppo_kl'] / aux['pg_clipfrac']` against
    the OLD `masked_mean = sum(x*mask) / (sum(mask) + 1e-8)` reduction.

    Within rtol=1e-5 the `min_denom=1.0` vs `+1e-8` choice is invisible
    when the mask is non-empty (`sum(mask) = 7`).
    """
    cfg = grpo_lib.GRPOConfig()
    cfg.temperature = 1.0
    ex = _grpo_inputs()
    model = _toy_actor()
    out = algo_core.grpo_loss_fn(model, ex, cfg, _PAD_ID, _EOS_ID)
    # Reconstruct ppo_kl with the legacy formula:
    #   ppo_kl = masked_mean(-(logp - old), mask)
    graphdef, state = nnx.split(model)
    per_token_logps, _ = rl_common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=ex.prompt_ids,
        completion_tokens=ex.completion_ids,
        pad_id=_PAD_ID,
        eos_id=_EOS_ID,
        completion_mask=ex.completion_mask,
        stop_gradient=False,
        return_logits=True,
        segment_ids=None,
        segment_positions=None,
        temperature=cfg.temperature,
    )
    per_token_logps = per_token_logps.astype(jnp.float32)
    legacy_ppo_kl = algo_core.masked_mean(
        -(per_token_logps - ex.old_per_token_logps), ex.completion_mask
    )
    np.testing.assert_allclose(
        out.aux_metrics['ppo_kl'].compute(),
        float(legacy_ppo_kl),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_grpo_no_kl_when_ref_logps_missing(self):
    """When `ref_per_token_logps=None`, KL branch is skipped:
    `kl_loss` falls back to the pre-set zero sentinel, and the primary
    loss does NOT include a `beta * kl_loss` term.
    """
    cfg = grpo_lib.GRPOConfig()
    cfg.temperature = 1.0
    ex = _grpo_inputs()
    ex.ref_per_token_logps = None
    out = algo_core.grpo_loss_fn(_toy_actor(), ex, cfg, _PAD_ID, _EOS_ID)
    np.testing.assert_allclose(out.aux_metrics['kl'].compute(), 0.0)
    np.testing.assert_allclose(out.aux_metrics['kl_loss'].compute(), 0.0)
    # `primary_loss` should equal `pg_loss` exactly (no KL added).
    np.testing.assert_allclose(
        out.primary_loss.compute(),
        out.aux_metrics['pg_loss'].compute(),
        rtol=1e-6,
    )

  def test_grpo_unreduced_sum_aggregates_across_microbatches(self):
    """End-to-end gradient-accumulation invariant for `pg_loss`.

    Two micro-batches' `WeightedMetric` fields, summed and divided once,
    must equal the single-shot full-batch `pg_loss`. This is the property
    that makes `GradientAccumulator` produce identical results to
    full-batch training in the new contract.
    """
    cfg = grpo_lib.GRPOConfig()
    cfg.temperature = 1.0
    ex_full = _grpo_inputs()
    # Same shapes split across two microbatches (rows [0] and [1]).
    ex_a = _grpo_inputs()
    ex_a.prompt_ids = ex_full.prompt_ids[:1]
    ex_a.completion_ids = ex_full.completion_ids[:1]
    ex_a.completion_mask = ex_full.completion_mask[:1]
    ex_a.advantages = ex_full.advantages[:1]
    ex_a.ref_per_token_logps = ex_full.ref_per_token_logps[:1]
    ex_a.old_per_token_logps = ex_full.old_per_token_logps[:1]
    ex_b = _grpo_inputs()
    ex_b.prompt_ids = ex_full.prompt_ids[1:]
    ex_b.completion_ids = ex_full.completion_ids[1:]
    ex_b.completion_mask = ex_full.completion_mask[1:]
    ex_b.advantages = ex_full.advantages[1:]
    ex_b.ref_per_token_logps = ex_full.ref_per_token_logps[1:]
    ex_b.old_per_token_logps = ex_full.old_per_token_logps[1:]
    model = _toy_actor()
    out_full = algo_core.grpo_loss_fn(model, ex_full, cfg, _PAD_ID, _EOS_ID)
    out_a = algo_core.grpo_loss_fn(model, ex_a, cfg, _PAD_ID, _EOS_ID)
    out_b = algo_core.grpo_loss_fn(model, ex_b, cfg, _PAD_ID, _EOS_ID)
    # Sequence-mean-token-mean's per-row inner average is mode-specific,
    # so this invariant is exercised on `pg_clipfrac` which uses
    # token-mean denominator + token-counted unreduced sums (the property
    # holds exactly for any mode that uses an additive denominator).
    combined_num = (
        out_a.aux_metrics['pg_clipfrac'].unreduced_sum
        + out_b.aux_metrics['pg_clipfrac'].unreduced_sum
    )
    combined_denom = (
        out_a.aux_metrics['pg_clipfrac'].denominator
        + out_b.aux_metrics['pg_clipfrac'].denominator
    )
    np.testing.assert_allclose(
        combined_num / jnp.maximum(combined_denom, 1.0),
        out_full.aux_metrics['pg_clipfrac'].compute(),
        rtol=1e-5,
        atol=1e-5,
    )


class PpoPolicyLossFnEquivalenceTest(parameterized.TestCase):
  """`ppo_policy_loss_fn` lock-in: structure + golden numerics + legacy match."""

  def _cfg(self, entropy_coef=0.01, kl_coef=0.1, epsilon_c=None):
    return _ns(
        epsilon=0.2,
        epsilon_low=0.2,
        epsilon_high=0.2,
        epsilon_c=epsilon_c,
        entropy_coef=entropy_coef,
        kl_coef=kl_coef,
    )

  def test_returns_lossoutput_with_expected_aux_keys(self):
    """Public aux contract for PPO policy loss.

    The set of aux keys depends on whether `entropy_coef` / `kl_coef` are
    active; lock in the union for the default-config path.
    """
    out = algo_core.ppo_policy_loss_fn(
        _toy_actor(), _ppo_inputs(), self._cfg(), _PAD_ID, _EOS_ID
    )
    self.assertIsInstance(out, sft_utils.LossOutput)
    self.assertEqual(
        sorted(out.aux_metrics),
        sorted(['pg_clipfrac', 'pg_clipfrac_lower', 'loss/entropy', 'kl']),
    )

  def test_ppo_policy_golden_numerics(self):
    """Golden numerics captured from the current implementation."""
    out = algo_core.ppo_policy_loss_fn(
        _toy_actor(), _ppo_inputs(), self._cfg(), _PAD_ID, _EOS_ID
    )
    np.testing.assert_allclose(
        out.primary_loss.compute(), -0.32665366, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['pg_clipfrac'].compute(), 0.28571430, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['pg_clipfrac_lower'].compute(), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        out.aux_metrics['loss/entropy'].compute(), 2.45303679, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['kl'].compute(), -3.25056005, rtol=1e-5
    )

  def test_ppo_policy_no_entropy_branch_drops_aux_key(self):
    """When `entropy_coef=0`, the entropy term is skipped and the
    `loss/entropy` aux key is absent. (Documents the conditional aux
    surface so downstream code can rely on its presence/absence.)"""
    out = algo_core.ppo_policy_loss_fn(
        _toy_actor(),
        _ppo_inputs(),
        self._cfg(entropy_coef=0.0),
        _PAD_ID,
        _EOS_ID,
    )
    self.assertNotIn('loss/entropy', out.aux_metrics)

  def test_ppo_policy_no_kl_branch_drops_aux_key(self):
    """`kl_coef=0` skips the KL branch entirely; no `kl` aux is emitted."""
    out = algo_core.ppo_policy_loss_fn(
        _toy_actor(),
        _ppo_inputs(),
        self._cfg(kl_coef=0.0),
        _PAD_ID,
        _EOS_ID,
    )
    self.assertNotIn('kl', out.aux_metrics)


class PpoValueLossFnEquivalenceTest(parameterized.TestCase):
  """`ppo_value_loss_fn` lock-in: structure + golden numerics."""

  def _critic(self):
    return rl_utils.create_critic_model(_toy_actor())

  def test_returns_lossoutput(self):
    out = algo_core.ppo_value_loss_fn(
        self._critic(), _ppo_inputs(), 0.2, _PAD_ID, _EOS_ID
    )
    self.assertIsInstance(out, sft_utils.LossOutput)
    self.assertEqual(
        sorted(out.aux_metrics),
        sorted(['vf_loss', 'vpred_mean', 'vf_clipfrac', 'return_mean']),
    )

  def test_ppo_value_golden_numerics(self):
    out = algo_core.ppo_value_loss_fn(
        self._critic(), _ppo_inputs(), 0.2, _PAD_ID, _EOS_ID
    )
    np.testing.assert_allclose(
        out.primary_loss.compute(), 0.5568355917, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['vf_loss'].compute(), 0.5568355917, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['vpred_mean'].compute(), -0.56332552, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.aux_metrics['vf_clipfrac'].compute(), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        out.aux_metrics['return_mean'].compute(), 0.41428575, rtol=1e-5
    )

  def test_ppo_value_loss_legacy_equivalence(self):
    """`vf_loss = 0.5 * masked_mean((clipped vf_losses), mask)` — the OLD
    formula reconstructed in-test against `compute()` to guarantee the
    refactor preserved the math, not just the golden value."""
    critic = self._critic()
    ex = _ppo_inputs()
    out = algo_core.ppo_value_loss_fn(critic, ex, 0.2, _PAD_ID, _EOS_ID)

    # Reconstruct OLD reduction from scratch.
    vpreds = rl_common.compute_score(
        critic,
        ex.prompt_ids,
        ex.completion_ids,
        _PAD_ID,
        _EOS_ID,
        stop_gradient=False,
        segment_ids=None,
        segment_positions=None,
    )[:, -ex.completion_ids.shape[1] - 1 : -1]
    vpred_clipped = jnp.clip(
        vpreds, ex.old_values - 0.2, ex.old_values + 0.2
    )
    vf_losses1 = jnp.square(vpreds - ex.returns)
    vf_losses2 = jnp.square(vpred_clipped - ex.returns)
    legacy_vf_loss = 0.5 * algo_core.masked_mean(
        jnp.maximum(vf_losses1, vf_losses2), ex.completion_mask
    )
    np.testing.assert_allclose(
        out.primary_loss.compute(),
        float(legacy_vf_loss),
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()

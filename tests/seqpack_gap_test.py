"""Gap-coverage tests for unreduced-loss + stream grad-accumulation (c2f07d42).

Complements the existing GradientAccumulatorTest (which proves the accumulator
*unit*, including denom-aware Sigma-g/Sigma-d). These tests fill four gaps found
by coverage review:

  G1  WeightedMetric.compute_scale/compute edge cases (loss-side safe divide).
  G2  pack_sequences is_update_step tagging cadence (rl_utils_test not updated
      by c2f07d42).
  G4  Trainer-integration "currently mean-of-means" pin: emulates the exact ops
      _train_step performs on the accumulator (peft_trainer local-scale + add
      denom=1.0) and pins that the e2e path is still mean-of-means, NOT the
      packing-correct global weighting (gated behind b/491970038).
  G3  Agentic is_update_step cadence — spec test of the counter logic mirrored
      from agentic_rl_learner (e2e coverage is via the GKE run).

Run: JAX_PLATFORMS=cpu PYTHONPATH=. <venv>/bin/python -m pytest \
       tests/seqpack_gap_test.py -q
"""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import utils as rl_utils
from tunix.sft import peft_trainer
from tunix.sft import utils


def _unwrap(state):
  return jax.tree_util.tree_map(
      lambda v: v[...] if isinstance(v, nnx.Variable) else v,
      state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )


# ======================================================================
# G1 — WeightedMetric loss-side safe divide
# ======================================================================
class WeightedMetricEdgeCaseTest(parameterized.TestCase):

  def test_pure_denom_zero_scale_and_value_are_zero(self):
    m = utils.WeightedMetric(jnp.asarray(5.0), jnp.asarray(0.0))
    self.assertEqual(float(m.compute_scale()), 0.0)
    self.assertEqual(float(m.compute()), 0.0)

  def test_pure_denom_zero_grad_is_finite(self):
    # d(compute)/d(sum) == scale == 0 at denom=0; must be finite (no 0/0 NaN).
    def f(s):
      return utils.WeightedMetric(s, jnp.asarray(0.0)).compute()

    g = jax.grad(f)(jnp.asarray(5.0))
    self.assertTrue(bool(jnp.isfinite(g)))
    self.assertEqual(float(g), 0.0)

  def test_eps_matches_legacy_formula(self):
    s, d, eps = 10.0, 4.0, 1e-8
    m = utils.WeightedMetric(jnp.asarray(s), jnp.asarray(d), eps=eps)
    np.testing.assert_allclose(float(m.compute()), s / (d + eps), rtol=1e-6)

  def test_min_denom_clamps_small_denominator(self):
    s, d = 3.0, 0.5  # d < min_denom
    m = utils.WeightedMetric(jnp.asarray(s), jnp.asarray(d), min_denom=1.0)
    np.testing.assert_allclose(float(m.compute()), s / max(d, 1.0), rtol=1e-6)

  def test_min_denom_noop_when_denominator_large(self):
    s, d = 8.0, 4.0  # d > min_denom
    m = utils.WeightedMetric(jnp.asarray(s), jnp.asarray(d), min_denom=1.0)
    np.testing.assert_allclose(float(m.compute()), s / 4.0, rtol=1e-6)

  def test_eps_then_min_denom_order(self):
    s, d, eps, md = 6.0, 2.0, 1e-8, 1.0
    m = utils.WeightedMetric(
        jnp.asarray(s), jnp.asarray(d), eps=eps, min_denom=md
    )
    np.testing.assert_allclose(
        float(m.compute()), s / max(d + eps, md), rtol=1e-6
    )


# ======================================================================
# G2 — pack_sequences is_update_step tagging cadence
# ======================================================================
class PackSequencesIsUpdateStepTest(parameterized.TestCase):

  def _item(self, prompt_len, completion_len):
    return common.TrainExample(
        prompt_ids=jnp.ones((1, prompt_len), dtype=jnp.int32),
        prompt_mask=jnp.ones((1, prompt_len), dtype=jnp.int32),
        completion_ids=jnp.ones((1, completion_len), dtype=jnp.int32) * 2,
        completion_mask=jnp.ones((1, completion_len), dtype=jnp.int32),
        advantages=jnp.array([1.5], dtype=jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
    )

  def test_target_cadence_marks_update_every_k_items(self):
    # 4 tiny items, huge budget so only target_items_per_update triggers flush.
    items = iter([[self._item(1, 1)] for _ in range(4)])
    packs = list(
        rl_utils.pack_sequences(
            items, max_token_budget=1000, pad_id=0, target_items_per_update=2
        )
    )
    flags = [bool(np.asarray(p[0].is_update_step)) for p in packs]
    # After every 2 items a flush fires with is_update_step=True.
    self.assertEqual(flags, [True, True])

  def test_budget_flush_is_false_and_final_flush_is_true(self):
    # Two items of 6 tokens, budget 8 -> second item forces a budget flush of
    # the first (is_update=False); the trailing buffer flushes at end (True).
    items = iter([[self._item(3, 3)], [self._item(3, 3)]])
    packs = list(
        rl_utils.pack_sequences(items, max_token_budget=8, pad_id=0)
    )
    flags = [bool(np.asarray(p[0].is_update_step)) for p in packs]
    self.assertEqual(flags, [False, True])


# ======================================================================
# G4 — Trainer-integration "currently mean-of-means" pin
# ======================================================================
class TrainStepMeanOfMeansPinTest(parameterized.TestCase):
  """Pins that the e2e grad-accum path is still mean-of-means.

  peft_trainer._train_step currently does (lines ~479, ~485):
      grads = grads * aux.primary_loss.compute_scale()   # local 1/denom
      grad_accumulator.add(grads, denom=1.0)             # TODO b/491970038
  so get() = mean_i(grad(sum_i)/denom_i) — mean of LOCAL means, identical to the
  old optax.MultiSteps. It is NOT the packing-correct Sigma grad(sum_i)/Sigma
  denom_i. When b/491970038 lands (drop the local scale, pass denom=size) this
  test's expectation must flip to the global-weighted tree.
  """

  def _model_and_data(self, n, seed=0):
    rngs = nnx.Rngs(seed)
    model = nnx.Linear(in_features=4, out_features=2, rngs=rngs)
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    x = jax.random.normal(k[0], (n, 4))
    y = jax.random.normal(k[1], (n, 2))
    return model, x, y

  @staticmethod
  def _loss_sum(model, x, y):
    return jnp.sum((model(x) - y) ** 2)

  def test_train_step_recipe_is_mean_of_means_not_global(self):
    d1, d2 = 2, 6  # uneven micro-batch sizes
    model, x, y = self._model_and_data(d1 + d2)
    grad_fn = nnx.value_and_grad(self._loss_sum)
    _, g1 = grad_fn(model, x[:d1], y[:d1])
    _, g2 = grad_fn(model, x[d1:], y[d1:])
    g1, g2 = _unwrap(g1), _unwrap(g2)

    # Emulate _train_step exactly: local scale by 1/denom, add with denom=1.0.
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)
    acc.add(jax.tree_util.tree_map(lambda a: a * (1.0 / d1), g1), denom=jnp.asarray(1.0))
    acc.add(jax.tree_util.tree_map(lambda a: a * (1.0 / d2), g2), denom=jnp.asarray(1.0))
    result = _unwrap(acc.get())

    mean_of_means = jax.tree_util.tree_map(
        lambda a, b: (a / d1 + b / d2) / 2.0, g1, g2
    )
    global_weighted = jax.tree_util.tree_map(
        lambda a, b: (a + b) / (d1 + d2), g1, g2
    )

    # Current behavior == mean-of-means.
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        result,
        mean_of_means,
    )
    # ... and it is NOT the packing-correct global weighting (uneven denoms).
    diff = jax.tree_util.tree_map(
        lambda a, b: jnp.max(jnp.abs(a - b)), result, global_weighted
    )
    max_diff = jax.tree_util.tree_reduce(
        jnp.maximum, diff, initializer=jnp.asarray(0.0)
    )
    self.assertGreater(float(max_diff), 1e-3)


# ======================================================================
# G3 — Agentic is_update_step cadence (spec of the counter logic)
# ======================================================================
class AgenticIsUpdateCadenceSpecTest(parameterized.TestCase):
  """Spec test of the agentic is_update cadence.

  Mirrors (does not import) agentic_rl_learner:
      unpacked_micro_step_counter += 1
      is_update = unpacked_micro_step_counter % grad_acc_steps == 0
  End-to-end coverage (that frozenlake's agentic loop actually flips on this)
  is via the GKE run, not CPU.
  """

  @parameterized.named_parameters(
      dict(testcase_name='ga4_8micro', grad_acc_steps=4, n=8,
           expected=[False, False, False, True, False, False, False, True]),
      dict(testcase_name='ga1_every_step', grad_acc_steps=1, n=3,
           expected=[True, True, True]),
      dict(testcase_name='ga3_7micro', grad_acc_steps=3, n=7,
           expected=[False, False, True, False, False, True, False]),
  )
  def test_cadence(self, grad_acc_steps, n, expected):
    counter = 0
    got = []
    for _ in range(n):
      counter += 1
      got.append(counter % grad_acc_steps == 0)
    self.assertEqual(got, expected)


# ======================================================================
# G5 — Packed logp == padded logp (unified-seqpack P1.0 gate)
# ======================================================================
_VOCAB, _D = 16, 8


class _ToySegmentAttn(nnx.Module):
  """1-layer segment-aware attention + LM head.

  Builds a block-diagonal causal mask from ``segment_ids`` (packed path) or from
  ``attention_mask`` (padded path). This is the CPU stand-in for the real model:
  Qwen's segment isolation lives in the splash-attention (TPU-only) path, so a
  toy model is the only way to exercise compute_per_token_logps' packed gather
  (process_ids packed branch / logits_to_keep / front-pad alignment) on CPU.
  """

  def __init__(self, rngs):
    self.embed = nnx.Embed(_VOCAB, _D, rngs=rngs)
    self.q = nnx.Linear(_D, _D, rngs=rngs)
    self.k = nnx.Linear(_D, _D, rngs=rngs)
    self.v = nnx.Linear(_D, _D, rngs=rngs)
    self.o = nnx.Linear(_D, _D, rngs=rngs)
    self.head = nnx.Linear(_D, _VOCAB, rngs=rngs)

  def __call__(
      self, input_tokens, positions=None, attention_mask=None, cache=None,
      segment_ids=None, **kwargs
  ):
    x = self.embed(input_tokens)
    b, t, _ = x.shape
    q, k, v = self.q(x), self.k(x), self.v(x)
    scores = jnp.einsum('btd,bsd->bts', q, k) / jnp.sqrt(_D)
    idx = jnp.arange(t)
    causal = idx[:, None] >= idx[None, :]
    if segment_ids is not None:
      mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & causal[None]
    elif attention_mask is not None:
      mask = attention_mask
      if mask.ndim == 4:
        mask = mask[:, 0]
    else:
      mask = jnp.broadcast_to(causal[None], (b, t, t))
    scores = jnp.where(mask, scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1)
    return self.head(self.o(jnp.einsum('bts,bsd->btd', attn, v))), None


class PackedLogpEqualsPaddedTest(parameterized.TestCase):
  """Pins that compute_per_token_logps' packed path == the padded path.

  Basis of the unified-seqpack design: logp and training share one packed
  representation only if packed logp is numerically identical to the per-sequence
  padded logp. Packed output is token-aligned (common.py:436-441 front-pads 0.0
  so logp[i] matches token[i]); each segment's first token is a prompt
  (completion_mask=0) and is excluded.
  """

  def _model(self, seed=0):
    return nnx.split(_ToySegmentAttn(nnx.Rngs(seed)))

  def test_packed_equals_padded_and_isolation_is_necessary(self):
    graphdef, state = self._model()
    pad_id, eos_id = 0, 1

    def padded(prompt, comp):
      return np.asarray(common.compute_per_token_logps(
          graphdef, state,
          prompt_tokens=jnp.array(prompt)[None, :],
          completion_tokens=jnp.array(comp)[None, :],
          pad_id=pad_id, eos_id=eos_id))[0]

    # seq1: prompt [3] + completion [5,7,9]; seq2: prompt [2] + completion [4,6].
    lp1 = padded([3], [5, 7, 9])
    lp2 = padded([2], [4, 6])

    # Packed: full prompt+completion of both seqs in one buffer, prompt empty.
    buf = [3, 5, 7, 9, 2, 4, 6]
    seg = [1, 1, 1, 1, 2, 2, 2]
    pos = [0, 1, 2, 3, 0, 1, 2]
    lp_packed = np.asarray(common.compute_per_token_logps(
        graphdef, state,
        prompt_tokens=jnp.zeros((1, 0), jnp.int32),
        completion_tokens=jnp.array(buf)[None, :],
        pad_id=pad_id, eos_id=eos_id,
        segment_ids=jnp.array(seg)[None, :],
        segment_positions=jnp.array(pos)[None, :]))[0]

    # Token-aligned: completion tokens indexed by their own buffer positions.
    np.testing.assert_allclose(lp_packed[[1, 2, 3]], lp1, atol=1e-4)
    np.testing.assert_allclose(lp_packed[[5, 6]], lp2, atol=1e-4)

    # Negative control: collapse to one segment (no isolation) -> seq2 attends
    # seq1 -> seq2 logp must change, proving the match above is not a coincidence.
    lp_noiso = np.asarray(common.compute_per_token_logps(
        graphdef, state,
        prompt_tokens=jnp.zeros((1, 0), jnp.int32),
        completion_tokens=jnp.array(buf)[None, :],
        pad_id=pad_id, eos_id=eos_id,
        segment_ids=jnp.ones((1, len(buf)), jnp.int32),
        segment_positions=jnp.arange(len(buf))[None, :]))[0]
    self.assertFalse(np.allclose(lp_noiso[[5, 6]], lp2, atol=1e-4))


if __name__ == '__main__':
  absltest.main()

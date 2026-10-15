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

"""Tests for the sharded chunked log-prob head (TUNIX_LOGPS_SHARDED_HEAD).

ShardedLogpsHeadTest tests the head, in its "local" and "fused" modes, on a
mesh of 8 CPU devices; FusedScoringTest tests the "fused" mode's chunk scoring
on one device. JAX fixes the device count when its backend starts. This module
never sets the count at import, because in a shared pytest process that would
change it for every other test module. Run directly, it sets 8 CPU devices for
its own process. Under pytest, ShardedLogpsHeadTest runs in the current process
only if that process already has 8 CPU devices; otherwise it is skipped and
ChildProcessTest runs this file in a child process that has them.
"""

import contextlib
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tunix.rl import common
from tunix.tests import test_common as tc

_NUM_DEVICES = 8
# Set in the environment of the child process that ChildProcessTest starts.
_CHILD_ENV = 'TUNIX_SHARDED_LOGPS_HEAD_TEST_CHILD'
_B, _T, _H, _V = 4, 64, 32, 128
_AXES = ('data', 'fsdp', 'context', 'expert')
_RULES = (
    ('activation_batch', ('data', 'fsdp', 'expert')),
    ('activation_length', 'context'),
)


def _has_devices():
  """Whether this process can build the test mesh: 8 devices on CPU."""
  return jax.default_backend() == 'cpu' and jax.device_count() >= _NUM_DEVICES


class _HeadOnly(nnx.Module):
  """Trainable stand-in hidden states into a MaxText-like head.

  `h` is a Param, so dL/dh is checked. The head is an RMSNorm, then a float32
  table cast to `dtype` for the dot.
  """

  def __init__(self, dtype, batch=_B, seq=_T):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    self.dtype = dtype
    self.h = nnx.Param(
        jax.random.normal(keys[0], (batch, seq, _H)).astype(dtype)
    )
    self.scale = nnx.Param(
        0.2 * jax.random.normal(keys[1], (_H,)).astype(dtype)
    )
    self.table = nnx.Param(3.0 * jax.random.normal(keys[2], (_H, _V)) / _H**0.5)

  def __call__(
      self,
      input_tokens,
      positions,
      cache,
      attention_mask,
      segment_ids=None,
      skip_lm_head=False,
  ):
    del input_tokens, positions, cache, attention_mask, segment_ids
    h = self.h[...]
    return (h if skip_lm_head else self.compute_final_logits(h)), None

  def compute_final_logits(self, h):
    x = h.astype(jnp.float32)
    y = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
    y = (y * (self.scale[...].astype(jnp.float32) + 1.0)).astype(self.dtype)
    out = jnp.dot(y, self.table[...].astype(self.dtype))
    return out.astype(jnp.float32)


def _mesh():
  devices = np.array(jax.devices()[:_NUM_DEVICES]).reshape(1, 2, 2, 2)
  return Mesh(devices, _AXES)


class ShardedLogpsHeadTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not _has_devices():
      if os.environ.get(_CHILD_ENV):
        raise RuntimeError(f'child process has no {_NUM_DEVICES} CPU devices')
      raise unittest.SkipTest(
          f'needs {_NUM_DEVICES} CPU devices; ChildProcessTest runs these'
          ' tests in a child process'
      )

  def setUp(self):
    super().setUp()
    self.mesh = _mesh()

  def _inputs(self, batch, seq):
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(1, _V, (batch, seq)), jnp.int32)
    segment_ids = jnp.ones((batch, seq), jnp.int32)
    positions = jnp.asarray(np.tile(np.arange(seq), (batch, 1)), jnp.int32)
    weights = jnp.asarray(rng.normal(size=(2, batch, seq)), jnp.float32)
    # Shard like _RULES when it divides: 4 = data x fsdp x expert, 2 = context.
    if batch % 4 == 0 and seq % 2 == 0:
      sharding = NamedSharding(
          self.mesh, P(('data', 'fsdp', 'expert'), 'context')
      )
      tokens, segment_ids, positions = (
          jax.device_put(x, sharding) for x in (tokens, segment_ids, positions)
      )
    return tokens, segment_ids, positions, weights

  @contextlib.contextmanager
  def _head_switch(self, mode, rules=_RULES, expect=None, legacy_mesh=False):
    """Sets the switch, mesh and rules, then checks the head's engagement.

    The switch is read when compute_per_token_logps is traced and is not part
    of its jit cache key, so the caches are cleared first.

    Args:
      mode: TUNIX_LOGPS_SHARDED_HEAD value, or None to unset it.
      rules: flax logical-axis rules.
      expect: Whether each call of maybe_compute_sharded_chunked_logps in the
        body engaged, in order. Default: one call, engaged iff `mode` is set.
      legacy_mesh: Enter the mesh with `with mesh:` instead of `jax.set_mesh`.

    Yields:
      Nothing.
    """
    calls = []
    real = common.maybe_compute_sharded_chunked_logps

    def spy(*args, **kwargs):
      out = real(*args, **kwargs)
      calls.append(out is not None)
      return out

    jax.clear_caches()
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.dict(os.environ))
      if mode is None:
        os.environ.pop(common.SHARDED_HEAD_ENV, None)
      else:
        os.environ[common.SHARDED_HEAD_ENV] = mode
      stack.enter_context(
          mock.patch.object(common, 'maybe_compute_sharded_chunked_logps', spy)
      )
      stack.enter_context(self.mesh if legacy_mesh else jax.set_mesh(self.mesh))
      stack.enter_context(nn.logical_axis_rules(rules))
      yield
    self.assertEqual(calls, [mode is not None] if expect is None else expect)

  def _run(
      self,
      mode,
      dtype=jnp.float32,
      chunk_size=8,
      rules=_RULES,
      temperature=0.7,
      batch=_B,
      seq=_T,
      return_entropy=True,
      expect=None,
      legacy_mesh=False,
      differentiate=('logps', 'entropy'),
      nonfinite=(),
      lower=False,
  ):
    """Returns (log-probs, entropy, grads) of a loss weighting both outputs.

    Args:
      differentiate: The outputs the loss is differentiated through; the others
        enter it under `stop_gradient` (GRPO differentiates the log-probs only).
      nonfinite: (row, position) pairs whose hidden state is NaN. The scores
        computed from them get loss weight 0, as padding does.
      lower: Return the StableHLO text of the jitted value_and_grad instead of
        running it.
    """
    model = _HeadOnly(dtype, batch, seq)
    if nonfinite:
      h = np.array(model.h[...], np.float32)
      for row, pos in nonfinite:
        h[row, pos] = np.nan
      model.h.value = jnp.asarray(h, dtype)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, weights = self._inputs(batch, seq)
    if nonfinite:
      # Position t scores token t + 1, which sits at output index t + 1.
      w = np.array(weights)
      for row, pos in nonfinite:
        if pos + 1 < seq:
          w[:, row, pos + 1] = 0.0
      weights = jnp.asarray(w)

    def loss(state):
      out = common.compute_per_token_logps(
          graphdef,
          state,
          jnp.zeros((tokens.shape[0], 0), jnp.int32),
          tokens,
          pad_id=0,
          eos_id=-1,
          stop_gradient=False,
          return_entropy=return_entropy,
          segment_ids=segment_ids,
          segment_positions=positions,
          temperature=temperature,
          chunk_size=chunk_size,
      )
      logps, entropy = out if return_entropy else (out, jnp.zeros_like(out))
      lp, en = (
          x if name in differentiate else jax.lax.stop_gradient(x)
          for name, x in (('logps', logps), ('entropy', entropy))
      )
      value = jnp.sum(lp * weights[0]) + jnp.sum(en * weights[1])
      return value, (logps, entropy)

    with self._head_switch(mode, rules, expect, legacy_mesh):
      step = jax.jit(jax.value_and_grad(loss, has_aux=True))
      if lower:
        return step.lower(state).as_text()
      (_, (logps, entropy)), grads = step(state)
    return logps, entropy, grads

  def _assert_parity(self, got, want, entropy=True):
    """f32 parity of log-probs, entropy and the h, scale and table grads."""
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=2e-5)
    if entropy:
      np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=2e-5)
    for name in ('h', 'scale', 'table'):
      np.testing.assert_allclose(
          got[2][name][...],
          want[2][name][...],
          rtol=1e-4,
          atol=1e-5,
          err_msg=name,
      )

  @parameterized.named_parameters(
      ('chunk8', 8),
      ('uneven_chunk12', 12),
      ('one_step64', 64),
  )
  def test_matches_scan_head(self, chunk_size):
    want = self._run(None, chunk_size=chunk_size)
    got = self._run('local', chunk_size=chunk_size)
    self._assert_parity(got, want)
    # The tolerance is tight enough to see a one-position misalignment.
    self.assertFalse(
        np.allclose(np.roll(got[0], 1, axis=1), want[0], rtol=1e-5, atol=2e-5)
    )

  @parameterized.named_parameters(
      ('logps_only_as_in_grpo', ('logps',)),
      ('entropy_only', ('entropy',)),
  )
  def test_one_differentiated_output_matches_scan_head(self, differentiate):
    # The other output's cotangent reaches the backward as a symbolic zero.
    want = self._run(None, differentiate=differentiate)
    got = self._run('local', differentiate=differentiate)
    self._assert_parity(got, want)

  def test_nonfinite_hidden_states_at_masked_positions(self):
    # The head zeroes non-finite hidden states, as compute_chunked_logps does,
    # so NaN at positions the loss masks out leaves every gradient finite.
    bad = ((0, 5), (1, 31), (1, 32), (2, 40), (3, 63), (3, 0))
    want = self._run(None, nonfinite=bad, differentiate=('logps',))
    got = self._run('local', nonfinite=bad, differentiate=('logps',))
    for name in ('h', 'scale', 'table'):
      self.assertTrue(np.all(np.isfinite(got[2][name][...])), name)
    self.assertTrue(np.all(np.isfinite(got[0])))
    self._assert_parity(got, want)

  @parameterized.named_parameters(
      ('two_axes', ('data', 'fsdp'), ('expert', 'context')),
      ('three_axes', 'data', ('expert', 'fsdp', 'context')),
  )
  def test_sequence_axes_out_of_mesh_order(self, batch_rule, seq_rule):
    # jax.lax.ppermute numbers the sequence shards in mesh order. A rule that
    # lists the sequence axes in another order must not move the one-token
    # shifts across the wrong shards.
    rules = (('activation_batch', batch_rule), ('activation_length', seq_rule))
    want = self._run(None, rules=rules)
    got = self._run('local', rules=rules)
    self._assert_parity(got, want)

  def test_logps_only_matches_scan_head(self):
    # return_entropy=False: one output, and no entropy cotangent in the bwd.
    want = self._run(None, return_entropy=False)
    got = self._run('local', return_entropy=False)
    self._assert_parity(got, want, entropy=False)

  @parameterized.parameters(True, False)
  def test_forward_only_matches_scan_head(self, return_entropy):
    # No gradient: the custom_vjp's primal body runs instead of fwd/bwd.
    model = _HeadOnly(jnp.float32)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, _ = self._inputs(_B, _T)

    def logps(mode):
      with self._head_switch(mode):
        return jax.jit(
            lambda st: common.compute_per_token_logps(
                graphdef,
                st,
                jnp.zeros((_B, 0), jnp.int32),
                tokens,
                pad_id=0,
                eos_id=-1,
                stop_gradient=True,
                return_entropy=return_entropy,
                segment_ids=segment_ids,
                segment_positions=positions,
                temperature=1.0,
                chunk_size=8,
            )
        )(state)

    want, got = logps(None), logps('local')
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      np.testing.assert_allclose(g, w, rtol=1e-5, atol=2e-5)

  def test_bf16_matches_scan_head(self):
    # Same per-token math; XLA may order the per-token f32 reductions
    # differently for the smaller per-device chunk, so allow f32 rounding.
    want = self._run(None, dtype=jnp.bfloat16)
    got = self._run('local', dtype=jnp.bfloat16)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(got[2]['h'][...], np.float32),
        np.asarray(want[2]['h'][...], np.float32),
        rtol=1e-2,
        atol=1e-6,
    )
    for name in ('scale', 'table'):
      g = np.asarray(got[2][name][...], np.float32)
      w = np.asarray(want[2][name][...], np.float32)
      self.assertLess(np.linalg.norm(g - w) / np.linalg.norm(w), 2.0**-6, name)

  def test_mesh_order_layout_without_rules(self):
    want = self._run(None, rules=())
    got = self._run('local', rules=())
    self._assert_parity(got, want)

  def test_engages_on_mesh(self):
    self._run('local', expect=[True])
    self._run(None, expect=[False])

  def test_sequence_only_layout_when_batch_does_not_divide(self):
    # 3 rows cannot be split over fsdp x expert, so all 8 devices take a
    # slice of every row's sequence instead.
    want = self._run(None, batch=3)
    got = self._run('local', batch=3)
    self._assert_parity(got, want)

  def test_falls_back_when_no_layout(self):
    # 3 rows of 63 tokens divide over no mesh axis: compute_chunked_logps runs.
    want = self._run(None, batch=3, seq=63, chunk_size=7)
    with self.assertLogs(level='WARNING') as logs:
      got = self._run('local', batch=3, seq=63, chunk_size=7, expect=[False])
    self.assertIn('no token layout', '\n'.join(logs.output))
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      np.testing.assert_array_equal(g, w)

  def test_falls_back_without_mesh(self):
    model = _HeadOnly(jnp.float32)
    tokens = self._inputs(_B, _T)[0]
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: '1'}):
      with self.assertLogs(level='WARNING') as logs:
        self.assertIsNone(
            common.maybe_compute_sharded_chunked_logps(
                model, model.h[...], tokens, 1.0, 8, True
            )
        )
    self.assertIn('no mesh in context', '\n'.join(logs.output))

  def test_legacy_mesh_context_falls_back(self):
    # A legacy `with mesh:`, as tunix's RL cluster enters it, sets no abstract
    # mesh, so the head falls back to compute_chunked_logps.
    want = self._run(None, legacy_mesh=True)
    with self.assertLogs(level='WARNING') as logs:
      got = self._run('local', legacy_mesh=True, expect=[False])
    self.assertIn('no mesh in context', '\n'.join(logs.output))
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      np.testing.assert_array_equal(g, w)

  def test_unpacked_rows_do_not_consult_head(self):
    # Without segment_ids the switch is not read and the head is not called.
    model = _HeadOnly(jnp.float32)
    graphdef, state = nnx.split(model)
    tokens = self._inputs(_B, _T)[0]
    with self._head_switch('local', expect=[]):
      jax.jit(
          lambda st: common.compute_per_token_logps(
              graphdef,
              st,
              tokens[:, :8],
              tokens[:, 8:],
              pad_id=0,
              eos_id=-1,
              chunk_size=8,
          )
      )(state)

  def test_chunk_size_zero_does_not_consult_head(self):
    model = _HeadOnly(jnp.float32)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, _ = self._inputs(_B, _T)
    with self._head_switch('local', expect=[]):
      jax.jit(
          lambda st: common.compute_per_token_logps(
              graphdef,
              st,
              jnp.zeros((_B, 0), jnp.int32),
              tokens,
              pad_id=0,
              eos_id=-1,
              segment_ids=segment_ids,
              segment_positions=positions,
              chunk_size=0,
          )
      )(state)

  def test_unknown_switch_value_raises(self):
    with self.assertRaisesRegex(ValueError, common.SHARDED_HEAD_ENV):
      self._run('locla')

  @parameterized.parameters(
      ('', None),
      ('0', None),
      ('false', None),
      ('off', None),
      ('no', None),
      ('1', 'local'),
      ('true', 'local'),
      ('on', 'local'),
      ('yes', 'local'),
      ('local', 'local'),
      (' Local ', 'local'),
      ('TRUE', 'local'),
      ('fused', 'fused'),
      (' Fused ', 'fused'),
  )
  def test_mode_parsing(self, value, want):
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: value}):
      self.assertEqual(common.sharded_head_mode(), want)

  @parameterized.parameters('locla', '2', 'fuse', 'local,fused')
  def test_mode_parsing_rejects_unknown(self, value):
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: value}):
      with self.assertRaisesRegex(ValueError, common.SHARDED_HEAD_ENV):
        common.sharded_head_mode()

  @parameterized.named_parameters(
      (
          'rules',
          _RULES,
          4,
          64,
          (('data', 'fsdp', 'expert'), ('context',), 'logical-axis-rules'),
      ),
      ('no_rules', (), 4, 64, (('fsdp', 'context'), ('expert',), 'mesh-order')),
      (
          'batch_does_not_divide',
          _RULES,
          3,
          64,
          ((), ('fsdp', 'context', 'expert'), 'mesh-order'),
      ),
      ('no_layout', _RULES, 3, 63, None),
  )
  def test_token_layout(self, rules, batch, seq, want):
    # The layouts the parity and fallback tests above rely on.
    with nn.logical_axis_rules(rules):
      got = common._token_partition(self.mesh, batch, seq)  # pylint: disable=protected-access
    self.assertEqual(got, want)

  def _assert_fused_parity(self, got, want, entropy=True):
    """Asserts that "fused" matches "local": log-probs, entropy and grads.

    The log-probs and gradients are the same f32 operations, but the exp-sum is
    reduced in another fusion, so they may differ in their last bits; the
    entropy comes from another formula.

    Args:
      got: (log-probs, entropy, grads) with "fused".
      want: The same with "local".
      entropy: Whether to compare the entropy.
    """
    np.testing.assert_allclose(got[0], want[0], rtol=1e-6, atol=2e-6)
    if entropy:
      np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=1e-6)
    for name in ('h', 'scale', 'table'):
      g = np.asarray(got[2][name][...], np.float32)
      w = np.asarray(want[2][name][...], np.float32)
      np.testing.assert_allclose(
          g, w, rtol=1e-5, atol=1e-6 * np.abs(w).max(), err_msg=name
      )

  @parameterized.named_parameters(
      ('logps_only_as_in_grpo', ('logps',), 8, 0.7),
      ('logps_only_uneven_chunk', ('logps',), 12, 1.0),
      ('logps_only_one_step', ('logps',), 64, 0.7),
      ('logps_and_entropy', ('logps', 'entropy'), 8, 0.7),
      ('entropy_only', ('entropy',), 12, 1.0),
  )
  def test_fused_matches_local(self, differentiate, chunk_size, temperature):
    # With only the log-probs differentiated the backward forms the logits'
    # cotangent in one pass; a differentiated entropy takes the autodiff
    # backward. Chunk 12 does not divide the 32 local positions (padding).
    kwargs = dict(
        differentiate=differentiate,
        chunk_size=chunk_size,
        temperature=temperature,
    )
    want = self._run('local', **kwargs)
    got = self._run('fused', **kwargs)
    self._assert_fused_parity(got, want)

  def test_fused_matches_scan_head(self):
    want = self._run(None, differentiate=('logps',))
    got = self._run('fused', differentiate=('logps',))
    self._assert_parity(got, want)

  def test_fused_logps_only_output(self):
    # return_entropy=False: one output, and _head_logps_stats skips the entropy.
    want = self._run('local', return_entropy=False, differentiate=('logps',))
    got = self._run('fused', return_entropy=False, differentiate=('logps',))
    self._assert_fused_parity(got, want, entropy=False)

  def test_fused_nonfinite_hidden_states_at_masked_positions(self):
    bad = ((0, 5), (1, 31), (1, 32), (2, 40), (3, 63), (3, 0))
    want = self._run('local', nonfinite=bad, differentiate=('logps',))
    got = self._run('fused', nonfinite=bad, differentiate=('logps',))
    for name in ('h', 'scale', 'table'):
      self.assertTrue(np.all(np.isfinite(got[2][name][...])), name)
    self.assertTrue(np.all(np.isfinite(got[0])))
    self._assert_fused_parity(got, want)

  def test_fused_bf16_matches_local(self):
    want = self._run('local', dtype=jnp.bfloat16, differentiate=('logps',))
    got = self._run('fused', dtype=jnp.bfloat16, differentiate=('logps',))
    self._assert_fused_parity(got, want)

  @parameterized.parameters(True, False)
  def test_fused_forward_only_matches_local(self, return_entropy):
    # No gradient: the custom_vjp's primal body runs, scoring with
    # _head_logps_stats and keeping no max or exp-sum.
    model = _HeadOnly(jnp.float32)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, _ = self._inputs(_B, _T)

    def logps(mode):
      with self._head_switch(mode):
        return jax.jit(
            lambda st: common.compute_per_token_logps(
                graphdef,
                st,
                jnp.zeros((_B, 0), jnp.int32),
                tokens,
                pad_id=0,
                eos_id=-1,
                stop_gradient=True,
                return_entropy=return_entropy,
                segment_ids=segment_ids,
                segment_positions=positions,
                temperature=0.7,
                chunk_size=8,
            )
        )(state)

    want, got = logps('local'), logps('fused')
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    got_logps, want_logps = jax.tree.leaves(got)[0], jax.tree.leaves(want)[0]
    np.testing.assert_allclose(got_logps, want_logps, rtol=1e-6, atol=2e-6)
    if return_entropy:
      np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=1e-6)

  @parameterized.named_parameters(
      ('fused_logps_only_as_in_grpo', 'fused', ('logps',), False),
      ('fused_entropy_differentiated', 'fused', ('logps', 'entropy'), True),
      ('local', 'local', ('logps',), True),
  )
  def test_scatter_only_in_autodiff_backward(
      self, mode, differentiate, scatter
  ):
    # "fused" reads the target logit with a one-hot select-sum, and when only
    # the log-probs are differentiated its backward forms the logits' cotangent
    # in one pass behind one optimization barrier: the program gathers nothing
    # and scatters nothing. A differentiated entropy, like "local", takes the
    # autodiff backward, which transposes take_along_axis into a scatter.
    text = self._run(mode, differentiate=differentiate, lower=True)
    count = {
        op: len(re.findall(rf'\bstablehlo\.{op}\b', text))
        for op in ('gather', 'scatter', 'optimization_barrier')
    }
    if scatter:
      self.assertGreater(count['scatter'], 0, count)
      self.assertEqual(count['optimization_barrier'], 0, count)
    else:
      self.assertEqual(
          count, {'gather': 0, 'scatter': 0, 'optimization_barrier': 1}
      )
    if mode == 'local':
      self.assertGreater(count['gather'], 0, count)


class FusedScoringTest(parameterized.TestCase):
  """The "fused" head's chunk scoring against what it replaces.

  `_head_logps_stats` and `_head_logps_cotangent` against
  `selective_log_softmax`, `compute_entropy_from_logits` and `jax.vjp` of
  `selective_log_softmax`, on logits with non-finite entries and on target ids
  outside the vocabulary. One device is enough.
  """

  _V = 50

  def _inputs(self):
    """[2, 9, 50] logits, [2, 9] target ids and [2, 9] log-prob cotangents."""
    rng = np.random.default_rng(1)
    logits = 4.0 * rng.normal(size=(2, 9, self._V)).astype(np.float32)
    logits[0, 0, 7] = np.inf
    logits[0, 1, 8] = -np.inf
    logits[0, 2, 9] = np.nan
    logits[1, 4, :] = -np.inf  # no finite logit in the row
    ids = rng.integers(0, self._V, (2, 9)).astype(np.int32)
    # Targets on the non-finite entries; ids counting from the end; and ids
    # outside [-V, V), which take_along_axis reads as NaN.
    ids[0] = [7, 8, 9, self._V, -1, -self._V, -self._V - 1, 2**30, 0]
    g = rng.normal(size=(2, 9)).astype(np.float32)
    return jnp.asarray(logits), jnp.asarray(ids), jnp.asarray(g)

  def test_target_hit(self):
    ids = jnp.array([0, 4, 5, -1, -5, -6, 2**30], jnp.int32)
    hit, valid = common._target_hit(ids, (7, 5))  # pylint: disable=protected-access
    want = np.zeros((7, 5), bool)
    for row, v in ((0, 0), (1, 4), (3, 4), (4, 0)):
      want[row, v] = True
    np.testing.assert_array_equal(hit, want)
    np.testing.assert_array_equal(
        valid, [True, True, False, True, True, False, False]
    )

  @parameterized.parameters(True, False)
  def test_stats_match_selective_log_softmax(self, with_entropy):
    logits, ids, _ = self._inputs()
    stats = jax.jit(common._head_logps_stats, static_argnums=2)  # pylint: disable=protected-access
    logps, entropy, m, s = stats(logits, ids, with_entropy)
    want = common.selective_log_softmax(logits, ids)
    # NaN exactly where take_along_axis reads outside the vocabulary.
    np.testing.assert_array_equal(np.isnan(logps), np.isnan(want))
    self.assertEqual(int(np.isnan(want).sum()), 3)
    np.testing.assert_allclose(logps, want, rtol=1e-6, atol=1e-6)
    x = np.where(np.isfinite(logits), logits, 0.0)
    np.testing.assert_array_equal(m, x.max(-1))
    shifted = (x - x.max(-1, keepdims=True)).astype(np.float64)
    np.testing.assert_allclose(s, np.exp(shifted).sum(-1), rtol=1e-5)
    if with_entropy:
      np.testing.assert_allclose(
          entropy,
          common.compute_entropy_from_logits(logits),
          rtol=1e-5,
          atol=1e-6,
      )
    else:
      self.assertIsNone(entropy)

  def test_cotangent_matches_autodiff(self):
    logits, ids, g = self._inputs()
    _, _, m, s = common._head_logps_stats(logits, ids, False)  # pylint: disable=protected-access
    got = jax.jit(common._head_logps_cotangent)(logits, ids, g, m, s)  # pylint: disable=protected-access
    _, vjp = jax.vjp(lambda x: common.selective_log_softmax(x, ids), logits)
    (want,) = vjp(g)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    got = np.asarray(got)
    # A non-finite logit gets no cotangent (the transpose of the isfinite mask).
    self.assertTrue(np.all(got[~np.isfinite(logits)] == 0.0))
    # Over a row of finite logits the cotangent sums to g - g * sum(p) = 0, or
    # to -g when the id is outside the vocabulary, whose g the gather's
    # transpose drops.
    outside = np.zeros(g.shape, bool)
    outside[0, [3, 6, 7]] = True  # ids V, -V - 1 and 2**30
    finite = np.all(np.isfinite(logits), axis=-1)
    np.testing.assert_allclose(
        got.sum(-1)[outside], -np.asarray(g)[outside], rtol=1e-5
    )
    np.testing.assert_allclose(
        got.sum(-1)[finite & ~outside], 0.0, rtol=0, atol=1e-5
    )


class ChildProcessTest(absltest.TestCase):
  """Runs ShardedLogpsHeadTest in a child process when this one cannot."""

  def test_runs_head_tests_in_child_process(self):
    if os.environ.get(_CHILD_ENV) or _has_devices():
      self.skipTest('ShardedLogpsHeadTest runs in this process')
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    env[_CHILD_ENV] = '1'
    # The child imports the same tunix as this process.
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(common.__file__)))
    )
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (root, env.get('PYTHONPATH')) if p
    )
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    self.assertEqual(result.returncode, 0, result.stderr[-8000:])
    ran = re.search(r'^Ran (\d+) tests? in', result.stderr, re.M)
    self.assertIsNotNone(ran, result.stderr[-8000:])
    self.assertGreater(int(ran.group(1)), 1)


if __name__ == '__main__':
  tc.safe_set_n_cpu_devices(_NUM_DEVICES)
  absltest.main()

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

"""Tests for the sharded chunked log-prob head (TUNIX_LOGPS_SHARDED_HEAD)."""

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

from unittest import mock  # pylint: disable=g-import-not-at-top

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tunix.rl import common

_B, _T, _H, _V = 4, 64, 32, 128
_AXES = ('data', 'fsdp', 'context', 'expert')
_RULES = (
    ('activation_batch', ('data', 'fsdp', 'expert')),
    ('activation_length', 'context'),
)


class _HeadOnly(nnx.Module):
  """Frozen hidden states into a MaxText-like head (RMSNorm, f32 table)."""

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
  devices = np.array(jax.devices()[:8]).reshape(1, 2, 2, 2)
  return Mesh(devices, _AXES)


class ShardedLogpsHeadTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = _mesh()

  def _inputs(self, batch, seq):
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(1, _V, (batch, seq)), jnp.int32)
    segment_ids = jnp.ones((batch, seq), jnp.int32)
    positions = jnp.asarray(np.tile(np.arange(seq), (batch, 1)), jnp.int32)
    weights = jnp.asarray(rng.normal(size=(2, batch, seq)), jnp.float32)
    if batch % 4 == 0 and seq % 2 == 0:
      sharding = NamedSharding(
          self.mesh, P(('data', 'fsdp', 'expert'), 'context')
      )
      tokens, segment_ids, positions = (
          jax.device_put(x, sharding) for x in (tokens, segment_ids, positions)
      )
    return tokens, segment_ids, positions, weights

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
  ):
    model = _HeadOnly(dtype, batch, seq)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, weights = self._inputs(batch, seq)

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
      value = jnp.sum(logps * weights[0]) + jnp.sum(entropy * weights[1])
      return value, (logps, entropy)

    jax.clear_caches()
    env = {} if mode is None else {common.SHARDED_HEAD_ENV: mode}
    with mock.patch.dict(os.environ, env, clear=False):
      if mode is None:
        os.environ.pop(common.SHARDED_HEAD_ENV, None)
      with jax.set_mesh(self.mesh), nn.logical_axis_rules(rules):
        (_, (logps, entropy)), grads = jax.jit(
            jax.value_and_grad(loss, has_aux=True)
        )(state)
    return logps, entropy, grads

  @parameterized.named_parameters(
      ('local', 'local', 8),
      ('local_uneven_chunk', 'local', 12),
      ('local_one_step', 'local', 64),
  )
  def test_matches_scan_head(self, mode, chunk_size):
    want = self._run(None, chunk_size=chunk_size)
    got = self._run(mode, chunk_size=chunk_size)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=2e-5)
    for name in ('h', 'scale', 'table'):
      np.testing.assert_allclose(
          got[2][name][...],
          want[2][name][...],
          rtol=1e-4,
          atol=1e-5,
          err_msg=name,
      )

  def test_logps_only_matches_scan_head(self):
    # return_entropy=False: one output, and no entropy cotangent in the bwd.
    want = self._run(None, return_entropy=False)
    got = self._run('local', return_entropy=False)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=2e-5)
    for name in ('h', 'scale', 'table'):
      np.testing.assert_allclose(
          got[2][name][...],
          want[2][name][...],
          rtol=1e-4,
          atol=1e-5,
          err_msg=name,
      )

  @parameterized.parameters(True, False)
  def test_forward_only_matches_scan_head(self, return_entropy):
    # No gradient: the custom_vjp's primal body runs instead of fwd/bwd.
    model = _HeadOnly(jnp.float32)
    graphdef, state = nnx.split(model)
    tokens, segment_ids, positions, _ = self._inputs(_B, _T)

    def logps(mode):
      jax.clear_caches()
      env = {} if mode is None else {common.SHARDED_HEAD_ENV: mode}
      with mock.patch.dict(os.environ, env, clear=False):
        if mode is None:
          os.environ.pop(common.SHARDED_HEAD_ENV, None)
        with jax.set_mesh(self.mesh), nn.logical_axis_rules(_RULES):
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
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(
        got[2]['table'][...], want[2]['table'][...], rtol=1e-4, atol=1e-5
    )

  def test_engages_on_mesh(self):
    calls = []
    real = common.maybe_compute_sharded_chunked_logps

    def spy(*args, **kwargs):
      out = real(*args, **kwargs)
      calls.append(out is not None)
      return out

    with mock.patch.object(common, 'maybe_compute_sharded_chunked_logps', spy):
      self._run('local')
      self._run(None)
    self.assertEqual(calls, [True, False])

  def _engaged(self, **kwargs):
    calls = []
    real = common.maybe_compute_sharded_chunked_logps

    def spy(*args, **kw):
      out = real(*args, **kw)
      calls.append(out is not None)
      return out

    with mock.patch.object(common, 'maybe_compute_sharded_chunked_logps', spy):
      got = self._run('local', **kwargs)
    return calls, got

  def test_sequence_only_layout_when_batch_does_not_divide(self):
    # 3 rows cannot be split over fsdp x expert, so all 8 devices take a
    # slice of every row's sequence instead.
    calls, got = self._engaged(batch=3)
    want = self._run(None, batch=3)
    self.assertEqual(calls, [True])
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(
        got[2]['table'][...], want[2]['table'][...], rtol=1e-4, atol=1e-5
    )

  def test_falls_back_when_no_layout(self):
    # 3 rows of 63 tokens divide over no mesh axis: compute_chunked_logps runs.
    calls, got = self._engaged(batch=3, seq=63, chunk_size=7)
    want = self._run(None, batch=3, seq=63, chunk_size=7)
    self.assertEqual(calls, [False])
    np.testing.assert_array_equal(got[0], want[0])

  def test_falls_back_without_mesh(self):
    model = _HeadOnly(jnp.float32)
    tokens = self._inputs(_B, _T)[0]
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: '1'}):
      self.assertIsNone(
          common.maybe_compute_sharded_chunked_logps(
              model, model.h[...], tokens, 1.0, 8, True
          )
      )

  @parameterized.parameters(
      ('', None),
      ('0', None),
      ('false', None),
      ('1', 'local'),
      ('local', 'local'),
  )
  def test_mode_parsing(self, value, want):
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: value}):
      self.assertEqual(common.sharded_head_mode(), want)

  @parameterized.parameters('locla', 'per_step')
  def test_mode_parsing_rejects_unknown(self, value):
    with mock.patch.dict(os.environ, {common.SHARDED_HEAD_ENV: value}):
      with self.assertRaises(ValueError):
        common.sharded_head_mode()


if __name__ == '__main__':
  absltest.main()

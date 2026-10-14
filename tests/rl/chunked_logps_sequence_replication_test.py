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

"""Tests for the sequence replication in `common.compute_chunked_logps`.

Above `_MAX_CHUNK_WITHOUT_SEQUENCE_REPLICATION` (or with
TUNIX_LOGPS_REPLICATE_SEQUENCE=always) the chunk loop's int32 targets and f32
log-probs are replicated over the sequence axes, so the partitioner no longer
needs the 32-bit all-to-alls whose relayout the TPU compiler fails to lower.
These tests run on an 8-device CPU mesh shaped like the RL trainer's (batch
over `fsdp`, sequence over `context`) and check, each with a control that shows
the check can fail:

* the traced program is unchanged at or below the threshold and without a
  mesh, and the constraints land on exactly the 32-bit loop arrays above it;
* log-probs, entropies and gradients are bit-identical to the unreplicated
  program at the same chunk size, and match today's chunk 1024 and a float64
  reference.

XLA:CPU partitions these reshards without all-to-alls, so the removal of the
all-to-alls themselves is checked by an AOT compile for a TPU topology, not
here.
"""

import functools
import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import sharding
import numpy as np
from tunix.rl import common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# pylint: disable=protected-access
_ENV = common._SEQUENCE_REPLICATION_ENV
_THRESHOLD = common._MAX_CHUNK_WITHOUT_SEQUENCE_REPLICATION
_engaged = common._replicate_sequence_in_chunk_loop
_replicate = common._replicate_sequence_dims
# pylint: enable=protected-access

# One sequence per `fsdp` shard and 4,096 tokens per `context` shard, so chunk
# 4096 leaves one chunk per device and chunk 1024 four, like the bench cell.
_B, _L, _H, _V = 2, 16384, 32, 128
_OUTPUTS = ("logps", "entropy", "d_kernel", "d_hidden")


class _Head:
  """Stands in for MaxText's `compute_final_logits`: RMSNorm, vocab dot."""

  def __init__(self, kernel):
    self.kernel = kernel

  def compute_final_logits(self, y):
    yf = y.astype(jnp.float32)
    yf = yf * jax.lax.rsqrt(jnp.mean(yf * yf, axis=-1, keepdims=True) + 1e-6)
    return yf.astype(self.kernel.dtype) @ self.kernel


def _mesh():
  devices = np.array(jax.devices()[:8]).reshape(2, 4)
  return sharding.Mesh(devices, ("fsdp", "context"))


def _in_shardings(mesh):
  ker = sharding.NamedSharding(
      mesh, sharding.PartitionSpec(("fsdp", "context"))
  )
  hid = sharding.NamedSharding(mesh, sharding.PartitionSpec("fsdp", "context"))
  tok = sharding.NamedSharding(mesh, sharding.PartitionSpec("fsdp", "context"))
  return ker, hid, tok, tok


def _inputs(seq_len=_L, dtype=np.float32, seed=0):
  rng = np.random.default_rng(seed)
  kernel = (rng.normal(size=(_H, _V)) * 0.3).astype(dtype)
  hidden = rng.normal(size=(_B, seq_len, _H)).astype(dtype)
  ids = rng.integers(0, _V, size=(_B, seq_len), dtype=np.int32)
  weights = rng.normal(size=(_B, seq_len)).astype(np.float32)
  return kernel, hidden, ids, weights


def _fwd_bwd(
    kernel, hidden, ids, weights, *, chunk_size, return_entropy, packed=False
):
  """Log-probs (and entropies) plus gradients w.r.t. kernel and hidden."""
  if packed:
    # As compute_per_token_logps in packed mode: drop the first target, so the
    # sequence is one short of a chunk multiple and gets padded.
    ids, weights = ids[:, 1:], weights[:, 1:]

  def loss_fn(kernel, hidden):
    if packed:
      hidden = hidden[:, :-1]
    out = common.compute_chunked_logps(
        _Head(kernel), hidden, ids, 1.0, chunk_size, return_entropy
    )
    logps, entropy = out if return_entropy else (out, jnp.zeros_like(out))
    loss = jnp.sum(logps * weights) + jnp.sum(entropy * weights)
    return loss, (logps, entropy)

  (_, (logps, entropy)), (d_kernel, d_hidden) = jax.value_and_grad(
      loss_fn, argnums=(0, 1), has_aux=True
  )(kernel, hidden)
  return logps, entropy, d_kernel, d_hidden


def _jitted(chunk_size, return_entropy, mesh, packed=False):
  fn = functools.partial(
      _fwd_bwd,
      chunk_size=chunk_size,
      return_entropy=return_entropy,
      packed=packed,
  )
  if mesh is None:
    return jax.jit(fn)
  return jax.jit(fn, in_shardings=_in_shardings(mesh))


def _run(
    chunk_size,
    mode,
    *,
    return_entropy=False,
    use_mesh=True,
    args=None,
    packed=False,
):
  """[logps, entropy, d_kernel, d_hidden] as float64 numpy arrays."""
  args = _inputs() if args is None else args
  with mock.patch.dict(os.environ, {_ENV: mode}):
    if not use_mesh:
      out = _jitted(chunk_size, return_entropy, None, packed)(*args)
    else:
      mesh = _mesh()
      with jax.set_mesh(mesh):
        out = _jitted(chunk_size, return_entropy, mesh, packed)(*args)
  return [np.asarray(x, dtype=np.float64) for x in out]


def _assert_bit_identical(got, want):
  for name, g, w in zip(_OUTPUTS, got, want):
    np.testing.assert_array_equal(g, w, err_msg=name)


def _assert_close(got, want):
  """Chunk-size changes only reorder f32 sums (measured <= 1e-6 relative)."""
  for name, g, w in zip(_OUTPUTS, got, want):
    atol = 1e-5 * max(1.0, float(np.max(np.abs(w))))
    np.testing.assert_allclose(g, w, rtol=1e-5, atol=atol, err_msg=name)


def _constraints(lowered_text):
  """(sharding, tensor type) of each sharding constraint, as Shardy prints them.

  `{?}` is an unconstrained dim and `{}` a replicated one.

  Args:
    lowered_text: The StableHLO text of a lowered program.
  """
  return sorted(
      re.findall(
          r"sharding_constraint %\S+ <@\w+, (\[[^\]]*\])>\s*:\s*"
          r"tensor<([^>]+)>",
          lowered_text,
      )
  )


def _lowered(chunk_size, mode, *, use_mesh=True):
  args = _inputs()
  with mock.patch.dict(os.environ, {_ENV: mode}):
    if not use_mesh:
      return _jitted(chunk_size, False, None).lower(*args)
    mesh = _mesh()
    with jax.set_mesh(mesh):
      return _jitted(chunk_size, False, mesh).lower(*args)


def _reference_logps(args):
  """float64 log-probs and entropies without chunking, on the host."""
  kernel, hidden, ids, _ = (np.asarray(a, dtype=np.float64) for a in args)
  y = hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-6)
  logits = y @ kernel
  logits = logits - logits.max(axis=-1, keepdims=True)
  logz = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
  logp_all = logits - logz
  logps = np.take_along_axis(
      logp_all, ids.astype(np.int64)[..., None], axis=-1
  )[..., 0]
  entropy = -(np.exp(logp_all) * logp_all).sum(axis=-1)
  return logps, entropy


class ModeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("auto_512", "auto", 512, False),
      ("auto_at_threshold", "auto", _THRESHOLD, False),
      ("auto_above", "auto", _THRESHOLD + 1, True),
      ("auto_4096", "auto", 4096, True),
      ("empty_is_auto", "", 4096, True),
      ("case_and_space", " AUTO ", 1024, False),
      ("always_512", "always", 512, True),
      ("never_4096", "never", 4096, False),
  )
  def test_mode(self, mode, chunk_size, expected):
    with mock.patch.dict(os.environ, {_ENV: mode}):
      self.assertEqual(_engaged(chunk_size), expected)

  def test_unset_is_auto(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(_ENV, None)
      self.assertFalse(_engaged(1024))
      self.assertTrue(_engaged(2048))

  @parameterized.parameters("1", "true", "on", "yes", "alwayss")
  def test_unknown_mode_raises(self, mode):
    with mock.patch.dict(os.environ, {_ENV: mode}):
      with self.assertRaisesRegex(ValueError, _ENV):
        _engaged(4096)


class ProgramTest(parameterized.TestCase):

  def test_constraint_is_a_noop_without_a_mesh(self):
    x = jnp.zeros((2, 8), jnp.int32)
    self.assertIs(_replicate(x, (1,)), x)

  def test_program_unchanged_without_a_mesh(self):
    self.assertEqual(
        _lowered(4096, "auto", use_mesh=False).as_text(),
        _lowered(4096, "never", use_mesh=False).as_text(),
    )

  @parameterized.parameters(512, _THRESHOLD)
  def test_program_unchanged_at_or_below_threshold(self, chunk_size):
    self.assertEqual(
        _lowered(chunk_size, "auto").as_text(),
        _lowered(chunk_size, "never").as_text(),
    )

  @parameterized.parameters(2 * _THRESHOLD, 4096)
  def test_program_changes_above_threshold(self, chunk_size):
    # Control for the two tests above: the comparison does see the constraints.
    auto = _lowered(chunk_size, "auto").as_text()
    self.assertNotEqual(auto, _lowered(chunk_size, "never").as_text())

  @parameterized.parameters(2 * _THRESHOLD, 4096)
  def test_constrains_exactly_the_32bit_loop_arrays(self, chunk_size):
    # The int32 targets [B, L] going in, and the f32 log-probs [chunks, B,
    # chunk] coming out (forward, and its cotangent in the backward).
    # Sequence dims replicated, the batch dim left to the partitioner.
    nc = _L // chunk_size
    self.assertEqual(
        _constraints(_lowered(chunk_size, "auto").as_text()),
        sorted(
            [("[{?}, {}]", f"{_B}x{_L}xi32")]
            + [("[{}, {?}, {}]", f"{nc}x{_B}x{chunk_size}xf32")] * 2
        ),
    )
    self.assertEqual(_constraints(_lowered(chunk_size, "never").as_text()), [])

  def test_lowered_threshold_changes_the_1024_program(self):
    # Control: had the gate engaged at 1024, the comparison would catch it.
    with mock.patch.object(
        common, "_MAX_CHUNK_WITHOUT_SEQUENCE_REPLICATION", 512
    ):
      self.assertNotEqual(
          _lowered(1024, "auto").as_text(), _lowered(1024, "never").as_text()
      )


class NumericsTest(parameterized.TestCase):

  @parameterized.product(
      chunk_size=(2 * _THRESHOLD, 4096), return_entropy=(False, True)
  )
  def test_bit_identical_to_unreplicated_program(
      self, chunk_size, return_entropy
  ):
    # Only data movement changes, so every value is the same bit for bit.
    _assert_bit_identical(
        _run(chunk_size, "auto", return_entropy=return_entropy),
        _run(chunk_size, "never", return_entropy=return_entropy),
    )

  def test_always_at_1024_is_bit_identical(self):
    _assert_bit_identical(
        _run(1024, "always", return_entropy=True),
        _run(1024, "never", return_entropy=True),
    )

  def test_bit_identical_without_a_mesh(self):
    _assert_bit_identical(
        _run(4096, "auto", return_entropy=True, use_mesh=False),
        _run(4096, "never", return_entropy=True, use_mesh=False),
    )

  @parameterized.parameters(False, True)
  def test_4096_matches_todays_1024(self, return_entropy):
    _assert_close(
        _run(4096, "auto", return_entropy=return_entropy),
        _run(1024, "auto", return_entropy=return_entropy),
    )

  def test_packed_padded_sequence(self):
    got = _run(4096, "auto", return_entropy=True, packed=True)
    _assert_bit_identical(
        got, _run(4096, "never", return_entropy=True, packed=True)
    )
    _assert_close(got, _run(1024, "never", return_entropy=True, packed=True))

  def test_matches_float64_reference(self):
    args = _inputs()
    logps, entropy, _, _ = _run(4096, "auto", return_entropy=True, args=args)
    ref_logps, ref_entropy = _reference_logps(args)
    np.testing.assert_allclose(logps, ref_logps, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(entropy, ref_entropy, rtol=1e-5, atol=1e-5)

  def test_comparison_catches_a_wrong_constraint(self):
    # Control: a constraint that shifts the targets by one position must fail
    # the comparisons the tests above pass.
    def shifted(x, seq_dims):
      return jnp.roll(x, 1, axis=seq_dims[0])

    with mock.patch.object(common, "_replicate_sequence_dims", shifted):
      got = _run(4096, "auto")
    want = _run(4096, "never")
    with self.assertRaises(AssertionError):
      _assert_bit_identical(got, want)
    with self.assertRaises(AssertionError):
      _assert_close(got, want)


if __name__ == "__main__":
  absltest.main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fp32MasterOptimizer with master_storage="residual16" (MaxText fp32_master_storage), on CPU.

residual16 stores a bfloat16 scanned-layer weight's float32 master as the 16 low bits the weight does not hold (uint16)
plus one bit per element, packed along the scan axis, that says whether the cast rounded the magnitude up; each update
rebuilds the master as ((weight bits - up) << 16) | low bits. So masters, weights and Optax state must be bit-identical
to the float32 storage.

  - _residual16_split / _residual16_join: every non-NaN float32 pattern in a large random sample plus the special
    values (zeros, infinities, the largest finite, the smallest normal, denormals, exact bfloat16 rounding ties) is
    rebuilt bit for bit from its bfloat16 cast, by round-to-nearest-even and by truncation; the violation count is
    exactly the number of NaN inputs.
  - _pack_bits / _unpack_bits round-trip along each axis for lengths around multiples of 8.
  - The optimizer: a toy model with scanned `layers` weights (bfloat16 stacked on axis 1, and a float32 one), a
    non-scanned bfloat16 weight and a float32 weight; adamw with clipping at a learning rate large enough to move the
    bfloat16 weights; 8 steps with random gradients: weights, rebuilt masters and Adam moments bit-identical to the
    float32 storage after every step; residual16_violations stays 0; only scanned bfloat16 weights store uint16 low bits
    and a packed uint8 bit array (the stack axis 15 -> 2), every other weight keeps its float32 master.
  - A join that ignores the up bit is caught by the same comparison.
  - The builder refuses an unknown storage, and residual16 without the scan axis.
  - create_maxtext_engine builds the optimizer with MaxText's fp32_master_storage and param_scan_axis.
"""

from unittest import mock

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tunix.utils import maxtext_utils


def _bits(x):
  return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint32))


def _special_values():
  f32 = np.float32
  info = np.finfo(f32)
  vals = [0.0, -0.0, np.inf, -np.inf, info.max, -info.max, info.tiny, -info.tiny]
  vals += [info.tiny / 2, info.tiny / 1024, -info.tiny / 3, f32(1e-45)]  # denormals
  # Exact bfloat16 rounding ties: the low 16 bits are 0x8000 (an even and an odd bfloat16 neighbour).
  ties = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0x7F7F8000, 0x00018000], np.uint32).view(f32)
  return np.concatenate([np.array(vals, f32), ties])


class Residual16BitsTest(absltest.TestCase):

  def _check_rebuild(self, master, weight):
    lo, up, bad = maxtext_utils._residual16_split(jnp.asarray(master), jnp.asarray(weight))
    rebuilt = maxtext_utils._residual16_join(jnp.asarray(weight), lo, up)
    finite = ~np.isnan(master)
    np.testing.assert_array_equal(_bits(rebuilt)[finite], _bits(master)[finite])
    # The violation counter counts exactly the inputs it does not rebuild, and those are all NaN.
    not_rebuilt = _bits(rebuilt) != _bits(master)
    self.assertTrue(np.all(np.isnan(master[not_rebuilt])))
    self.assertEqual(int(bad), int(np.sum(not_rebuilt)))
    self.assertEqual(lo.dtype, jnp.uint16)
    self.assertEqual(up.dtype, jnp.bool_)
    return int(bad)

  def test_rebuilds_every_non_nan_float32_from_its_bfloat16_cast(self):
    rng = np.random.default_rng(0)
    sample = rng.integers(0, 2**32, size=1 << 20, dtype=np.uint64).astype(np.uint32).view(np.float32)
    master = np.concatenate([sample, _special_values()])
    # Round to nearest even (XLA's cast): some NaN patterns cannot be rebuilt, and the counter must see them.
    self.assertGreater(self._check_rebuild(master, jnp.asarray(master).astype(jnp.bfloat16)), 0)
    truncated = jax.lax.bitcast_convert_type(
        (jax.lax.bitcast_convert_type(jnp.asarray(master), jnp.uint32) >> 16).astype(jnp.uint16), jnp.bfloat16
    )
    self._check_rebuild(master, truncated)  # round toward zero: never rounds the magnitude up

  def test_pack_unpack_round_trip(self):
    rng = np.random.default_rng(1)
    for n in (1, 7, 8, 9, 15, 16, 17):
      for axis in (0, 1, 2, -3, -2, -1):
        shape = [3, 4, 5]
        shape[axis] = n
        bits = jnp.asarray(rng.random(shape) < 0.5)
        packed = maxtext_utils._pack_bits(bits, axis)
        self.assertEqual(packed.dtype, jnp.uint8)
        self.assertEqual(packed.shape[axis], -(-n // 8))
        np.testing.assert_array_equal(np.asarray(maxtext_utils._unpack_bits(packed, axis, n)), np.asarray(bits))


def _toy_model():
  from flax import nnx  # pylint: disable=g-import-not-at-top

  class Block(nnx.Module):

    def __init__(self, key):
      k1, k2, k3 = jax.random.split(key, 3)
      self.wi = nnx.Param((jax.random.normal(k1, (16, 15, 8)) * 0.05).astype(jnp.bfloat16))
      self.scale = nnx.Param((1.0 + 0.01 * jax.random.normal(k2, (32, 15))).astype(jnp.bfloat16))
      self.gate = nnx.Param(jax.random.normal(k3, (8, 15, 4)) * 0.05)  # a float32 weight in the stack

  class Toy(nnx.Module):

    def __init__(self, key):
      k1, k2 = jax.random.split(key)
      self.layers = Block(k1)
      self.embed = nnx.Param((jax.random.normal(k2, (24, 8)) * 0.05).astype(jnp.bfloat16))
      self.norm = nnx.Param(jnp.ones((8,), jnp.float32))

  return Toy(jax.random.PRNGKey(0))


def _val(v):
  """The array (or None) held by an optimizer-state leaf."""
  return v.get_value() if hasattr(v, "get_value") else v


def _run(storage, steps, join=None):
  """Per-step weights, rebuilt masters and inner state of the optimizer with `storage`."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  model = _toy_model()
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-2, b1=0.9, b2=0.999, weight_decay=0.0))
  if storage == "f32":
    opt = maxtext_utils._build_fp32_master_optimizer_cls(nnx)(model, tx, wrt=nnx.Param)
  else:
    opt = maxtext_utils._build_fp32_master_optimizer_cls(nnx, master_storage=storage, scan_axis=1)(
        model, tx, wrt=nnx.Param
    )
  rng = np.random.default_rng(2)
  out = []
  for _ in range(steps):
    grads = jax.tree.map(
        lambda x: jnp.asarray(rng.standard_normal(x.shape), jnp.float32), nnx.state(model, nnx.Param)
    )
    with mock.patch.object(maxtext_utils, "_residual16_join", join or maxtext_utils._residual16_join):
      opt.update(model, grads)
    params = {
        "wi": np.asarray(model.layers.wi[...]),
        "scale": np.asarray(model.layers.scale[...]),
        "gate": np.asarray(model.layers.gate[...]),
        "embed": np.asarray(model.embed[...]),
        "norm": np.asarray(model.norm[...]),
    }
    ost = opt.opt_state
    masters = {}
    for name in ("wi", "scale"):
      mp = jnp.asarray(_val(ost["master_params"]["layers"][name]))
      if storage == "residual16":
        up = maxtext_utils._unpack_bits(jnp.asarray(_val(ost["master_up"]["layers"][name])), 1, params[name].shape[1])
        mp = maxtext_utils._residual16_join(jnp.asarray(params[name]), mp, up)
      masters[name] = np.asarray(mp)
    masters["embed"] = np.asarray(_val(ost["master_params"]["embed"]))
    out.append({
        "params": params,
        "masters": masters,
        "inner": [np.asarray(x) for x in jax.tree.leaves(ost["inner"])],
        "violations": int(np.asarray(_val(ost["residual16_violations"]))) if storage == "residual16" else 0,
        "opt_state": ost,
    })
  return out


def _tree_bits_equal(a, b):
  la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
  return len(la) == len(lb) and all(
      x.shape == y.shape and x.dtype == y.dtype
      and np.array_equal(x.reshape(-1).view(np.uint8), y.reshape(-1).view(np.uint8))
      for x, y in zip((np.asarray(v) for v in la), (np.asarray(v) for v in lb))
  )


class Residual16OptimizerTest(absltest.TestCase):

  def test_matches_the_float32_master_bit_for_bit(self):
    ref, got = _run("f32", 8), _run("residual16", 8)
    for step, (r, g) in enumerate(zip(ref, got)):
      self.assertTrue(_tree_bits_equal(g["params"], r["params"]), f"weights, step {step + 1}")
      self.assertTrue(_tree_bits_equal(g["masters"], r["masters"]), f"masters, step {step + 1}")
      self.assertTrue(_tree_bits_equal(g["inner"], r["inner"]), f"adam state, step {step + 1}")
      self.assertEqual(g["violations"], 0)
    # The bfloat16 scanned weights moved (so the comparison covers rounded-up casts, not only exact ones).
    self.assertFalse(np.array_equal(ref[0]["params"]["wi"], ref[-1]["params"]["wi"]))
    ost = got[-1]["opt_state"]
    for name, length in (("wi", 15), ("scale", 15)):
      self.assertEqual(np.asarray(_val(ost["master_params"]["layers"][name])).dtype, np.uint16)
      up = np.asarray(_val(ost["master_up"]["layers"][name]))
      self.assertEqual(up.dtype, np.uint8)
      self.assertEqual(up.shape[1], -(-length // 8))
    self.assertEqual(np.asarray(_val(ost["master_params"]["embed"])).dtype, np.float32)  # not scanned: f32 master
    self.assertIsNone(_val(ost["master_up"]["embed"]))
    self.assertIsNone(_val(ost["master_params"]["norm"]))  # a float32 weight is its own master
    self.assertIsNone(_val(ost["master_params"]["layers"]["gate"]))

  def test_a_join_without_the_up_bit_is_caught(self):
    def wrong_join(p, lo, up):
      del up
      high = jax.lax.bitcast_convert_type(p, jnp.uint16).astype(jnp.uint32)
      return jax.lax.bitcast_convert_type((high << 16) | lo.astype(jnp.uint32), jnp.float32)

    ref, bad = _run("f32", 4), _run("residual16", 4, join=wrong_join)
    self.assertFalse(_tree_bits_equal(bad[-1]["params"], ref[-1]["params"]))

  def test_builder_refuses_unknown_storage_and_missing_scan_axis(self):
    from flax import nnx  # pylint: disable=g-import-not-at-top

    with self.assertRaisesRegex(ValueError, "master_storage"):
      maxtext_utils._build_fp32_master_optimizer_cls(nnx, master_storage="residual8", scan_axis=1)
    with self.assertRaisesRegex(ValueError, "scan"):
      maxtext_utils._build_fp32_master_optimizer_cls(nnx, master_storage="residual16")


class CreateEngineStorageTest(absltest.TestCase):

  def _create(self, storage):
    class FakeEngine:

      def __init__(self, config, mesh=None, wrap_with_tunix_adapter=True, tokenizer_pad_id=0):
        del config, mesh, wrap_with_tunix_adapter, tokenizer_pad_id
        self.model = type("TunixMaxTextAdapter", (), {})()
        self._weight_converter = mock.MagicMock()

    engine_mod = mock.MagicMock()
    engine_mod.MaxTextTrainingEngine = FakeEngine
    cfg = mock.MagicMock()
    cfg.weight_dtype = "bfloat16"
    cfg.fp32_master_storage = storage
    cfg.param_scan_axis = 1
    cfg.float32_gate_logits = False
    with mock.patch.object(
        maxtext_utils, "maxtext_modules", return_value=(mock.MagicMock(), engine_mod, mock.MagicMock())
    ), mock.patch.object(maxtext_utils, "_build_fp32_master_optimizer_cls") as builder:
      maxtext_utils.create_maxtext_engine(cfg, mock.MagicMock(), log_shapes=False)
    return builder

  def test_residual16_reaches_the_optimizer_builder(self):
    builder = self._create("residual16")
    builder.assert_called_once()
    self.assertEqual(builder.call_args.kwargs, {"master_storage": "residual16", "scan_axis": 1})

  def test_f32_builds_the_optimizer_as_before(self):
    builder = self._create("f32")
    builder.assert_called_once()
    self.assertEqual(builder.call_args.kwargs, {})


if __name__ == "__main__":
  absltest.main()

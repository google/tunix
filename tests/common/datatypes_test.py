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

import importlib
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.common import datatypes
from tunix.common import test_utils


class TensorLikeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("ndarray", np.zeros((2, 3), dtype=np.float32)),
      ("numpy_scalar", np.float32(1.0)),
      ("custom_tensor", test_utils.CustomTensor(np.arange(4))),
  )
  def test_isinstance_positive(self, value):
    self.assertIsInstance(value, datatypes.TensorLike)

  @parameterized.named_parameters(
      ("int", 1),
      ("float", 1.5),
      ("bool", True),
      ("str", "str"),
      ("list", [1, 2]),
      (
          "missing_shape",
          types.SimpleNamespace(dtype=np.int32, __array__=lambda: np.zeros(1)),
      ),
      (
          "missing_dtype",
          types.SimpleNamespace(shape=(1,), __array__=lambda: np.zeros(1)),
      ),
      (
          "missing_dunder_array",
          types.SimpleNamespace(shape=(1,), dtype=np.int32),
      ),
  )
  def test_isinstance_negative(self, value):
    self.assertNotIsInstance(value, datatypes.TensorLike)

  def test_custom_tensor_converts_via_array(self):
    custom = test_utils.CustomTensor(np.arange(3, dtype=np.int32))
    self.assertEqual(custom.shape, (3,))
    self.assertEqual(custom.dtype, np.int32)
    tensor: datatypes.ArrayLike = custom
    np.testing.assert_array_equal(np.asarray(tensor), np.arange(3))
    np.testing.assert_array_equal(
        np.asarray(tensor).astype(np.float32),
        np.arange(3, dtype=np.float32),
    )

  def test_module_imports_without_ml_frameworks(self):
    orig = sys.modules["tunix.common.datatypes"]
    roots = (
        "jax",
        "flax",
        "optax",
        "torch",
        "jaxtyping",
        "vllm",
        "pathwaysutils",
        "tpu_sync",
    )
    prefixes = tuple(f"{r}." for r in roots)
    blocked = dict.fromkeys(
        (*roots, *(k for k in sys.modules if k.startswith(prefixes))),
        None,
    )
    try:
      with mock.patch.dict(sys.modules, blocked):
        sys.modules.pop("tunix.common.datatypes", None)
        reloaded = importlib.import_module("tunix.common.datatypes")
        self.assertIsInstance(np.zeros(2), reloaded.TensorLike)
    finally:
      setattr(sys.modules["tunix.common"], "datatypes", orig)


class CacheGeometryTest(parameterized.TestCase):

  def test_valid(self):
    geometry = datatypes.CacheGeometry(
        num_kv_heads=2, head_dim=4, window_size=8
    )
    self.assertEqual(geometry.window_size, 8)

  @parameterized.named_parameters(
      ("num_kv_heads", dict(num_kv_heads=0, head_dim=4), "num_kv_heads"),
      ("head_dim", dict(num_kv_heads=2, head_dim=-1), "head_dim"),
      (
          "window_size",
          dict(num_kv_heads=2, head_dim=4, window_size=0),
          "window_size",
      ),
  )
  def test_non_positive_raises(self, kwargs, field_name):
    with self.assertRaisesRegex(ValueError, field_name):
      datatypes.CacheGeometry(**kwargs)


class RoutedExpertsRefTest(parameterized.TestCase):

  def test_valid(self):
    ref = datatypes.RoutedExpertsRef(worker_id="w", instance_id="i", ref_id=3)
    self.assertEqual(ref.ref_id, 3)

  @parameterized.named_parameters(
      ("empty_worker_id", dict(worker_id="", instance_id="i", ref_id=0)),
      ("empty_instance_id", dict(worker_id="w", instance_id="", ref_id=0)),
      ("bool_ref_id", dict(worker_id="w", instance_id="i", ref_id=True)),
      ("negative_ref_id", dict(worker_id="w", instance_id="i", ref_id=-1)),
      ("overflow_ref_id", dict(worker_id="w", instance_id="i", ref_id=1 << 63)),
  )
  def test_invalid_raises(self, kwargs):
    with self.assertRaises(ValueError):
      datatypes.RoutedExpertsRef(**kwargs)


class RoutedExpertsEncodingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("uint8", "uint8", 8, "uint8"),
      ("uint8_none", "uint8", None, "uint8"),
      ("uint10", "uint10", 300, "uint10"),
      ("int16", "int16", None, "int16"),
      ("auto_uint8", "auto", 255, "uint8"),
      ("auto_uint10", "auto", 1023, "uint10"),
      ("auto_int16", "auto", 1024, "int16"),
  )
  def test_resolve(self, encode_mode, num_experts, expected):
    self.assertEqual(
        datatypes.resolve_routed_experts_wire_encoding(
            encode_mode, num_experts
        ),
        expected,
    )

  @parameterized.named_parameters(
      ("bad_mode", "fp8", 8),
      ("bool_experts", "auto", True),
      ("zero_experts", "auto", 0),
      ("uint8_overflow", "uint8", 256),
      ("uint10_overflow", "uint10", 1024),
      ("auto_none", "auto", None),
  )
  def test_resolve_invalid_raises(self, encode_mode, num_experts):
    with self.assertRaises(ValueError):
      datatypes.resolve_routed_experts_wire_encoding(encode_mode, num_experts)

  @parameterized.named_parameters(
      ("divisible", 8, (2, 5)),
      ("non_divisible", 6, (1, 8)),
  )
  def test_uint10_trailing_shape(self, top_k, expected):
    self.assertEqual(datatypes.uint10_trailing_shape(top_k), expected)

  @parameterized.named_parameters(
      ("uint8", "uint8", 24),
      ("uint10", "uint10", 30),
      ("int16", "int16", 48),
  )
  def test_bytes_per_token(self, encoding, expected):
    self.assertEqual(
        datatypes.routed_experts_bytes_per_token(3, 8, encoding), expected
    )

  @parameterized.named_parameters(
      ("zero_layers", 0, 8, "uint8"),
      ("bool_top_k", 3, True, "uint8"),
      ("bad_encoding", 3, 8, "fp8"),
  )
  def test_bytes_per_token_invalid_raises(self, num_layers, top_k, encoding):
    with self.assertRaises(ValueError):
      datatypes.routed_experts_bytes_per_token(num_layers, top_k, encoding)


class EnumTest(absltest.TestCase):

  def test_mode_str(self):
    self.assertEqual(str(datatypes.Mode.TRAIN), "train")
    self.assertEqual(str(datatypes.Mode.EVAL), "eval")


if __name__ == "__main__":
  absltest.main()

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
"""Shared helpers for the Zero-TIM kernel tests (CPU, Pallas interpret mode).

The kernels' bitwise guarantees are TPU/Mosaic properties.  On CPU the tests
check what interpret mode can prove:

* same-shape invariance (a row computed inside a larger batch is bitwise the
  row computed alone when both runs use the same tiles), which holds on any
  deterministic backend;
* exact known answers on "exact" operands (``exact_bf16``) whose every partial
  sum is representable in f32, so every accumulation order gives the same
  bits and cross-tile/cross-program comparisons are meaningful on CPU;
* closeness to float64 references on random operands.
"""

from __future__ import annotations

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

# The canonical engine bundle runs XLA with excess precision disabled (root
# cause R3): without it XLA may keep bf16 intermediates in f32 across
# fused elementwise chains, differently per program.
EXCESS_PRECISION_FLAG = "--xla_allow_excess_precision=false"
_EXCESS_PRECISION_PREFIX = "--xla_allow_excess_precision="

_UINT_BY_ITEMSIZE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}

# Why configure_cpu() could not apply its device count, if it could not.
_unapplied_reason: str | None = None


def configure_cpu(num_devices: int = 1) -> None:
  """Configures the CPU backend; call from ``setUpModule``.

  Under Bazel every test module is its own process and the settings always
  apply.  A shared test process (the OSS CI runs one pytest process per test
  directory) may already have initialized the backend in an earlier module;
  then the settings are best effort and tests that need more devices than
  exist are skipped by ``cpu_mesh``.  An explicit
  ``--xla_allow_excess_precision=...`` in ``XLA_FLAGS`` is left alone.

  Args:
    num_devices: The number of CPU devices the module's tests need.
  """
  global _unapplied_reason
  flags = os.environ.get("XLA_FLAGS", "")
  if _EXCESS_PRECISION_PREFIX not in flags:
    os.environ["XLA_FLAGS"] = f"{flags} {EXCESS_PRECISION_FLAG}".strip()
  if num_devices > 1:
    try:
      jax.config.update("jax_num_cpu_devices", num_devices)
    except RuntimeError as e:  # The backend is already initialized.
      _unapplied_reason = str(e)


def cpu_mesh(shape: tuple[int, ...], axis_names: tuple[str, ...]):
  """A mesh over the first ``prod(shape)`` CPU devices."""
  count = int(np.prod(shape))
  devices = jax.devices()
  if len(devices) < count:
    message = f"need {count} devices, have {len(devices)}"
    if _unapplied_reason is not None:
      raise unittest.SkipTest(f"{message} ({_unapplied_reason})")
    raise RuntimeError(message)
  return jax.sharding.Mesh(
      np.asarray(devices[:count]).reshape(shape), axis_names
  )


def exact_bf16(rng: np.random.Generator, shape) -> jax.Array:
  """bf16 values ``k / 8`` with ``k`` in [-2, 2].

  Products are multiples of 1/64 and sums of up to 2**12 of them stay exact
  in f32, so any accumulation order produces the same bits.

  Args:
    rng: The numpy generator.
    shape: The array shape.

  Returns:
    The bf16 array.
  """
  return jnp.asarray(rng.integers(-2, 3, size=shape) / 8.0, jnp.bfloat16)


def random_bf16(
    rng: np.random.Generator, shape, scale: float = 1.0
) -> jax.Array:
  return jnp.asarray(rng.standard_normal(shape) * scale, jnp.bfloat16)


def exact_matmul_reference(x, y) -> jax.Array:
  """float64 product rounded once to bf16 (exact for ``exact_bf16`` data)."""
  product = np.asarray(x, np.float64) @ np.asarray(y, np.float64)
  return jnp.asarray(product, jnp.float32).astype(jnp.bfloat16)


def as_f32(x) -> np.ndarray:
  return np.asarray(x, np.float32)


def assert_bitwise_equal(actual, expected) -> None:
  """Asserts identical dtype, shape and bit patterns."""
  actual = np.asarray(actual)
  expected = np.asarray(expected)
  if actual.dtype != expected.dtype:
    raise AssertionError(f"dtype {actual.dtype} != {expected.dtype}")
  if actual.shape != expected.shape:
    raise AssertionError(f"shape {actual.shape} != {expected.shape}")
  uint = _UINT_BY_ITEMSIZE[actual.dtype.itemsize]
  np.testing.assert_array_equal(actual.view(uint), expected.view(uint))

#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/engine_shims/p38_terminal_capture.py"
SPEC = importlib.util.spec_from_file_location("p38_terminal_capture", MODULE)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class TerminalCaptureTest(unittest.TestCase):

  def test_same_input_is_exact_and_one_bit_negative_is_detected(self):
    values = np.linspace(
        -3, 3, capture.P38_TERMINAL_ROW_BUCKET * 513,
        dtype=np.float32).reshape(capture.P38_TERMINAL_ROW_BUCKET, 513)
    observe = jax.jit(capture.fingerprint_terminal_rows)
    baseline = jax.device_get(observe(jnp.asarray(values)))
    repeat = jax.device_get(observe(jnp.asarray(values)))
    for left, right in zip(baseline, repeat):
      self.assertTrue(np.array_equal(left, right))

    mutated = values.copy()
    bits = mutated.view(np.uint32)
    bits[1, 300] ^= np.uint32(1)
    negative = jax.device_get(observe(jnp.asarray(mutated)))
    self.assertFalse(np.array_equal(baseline[0], negative[0]))
    self.assertEqual(baseline[0].shape, (4, 3, 6))
    self.assertEqual(baseline[1].shape, (4, 3))
    self.assertEqual(baseline[2].shape, (4, 3))

  def test_observer_rejects_variable_row_geometry(self):
    with self.assertRaisesRegex(ValueError, "fixed shared observer bucket"):
      capture.fingerprint_terminal_rows(jnp.zeros((1, 513), jnp.float32))

  def test_raw_processed_pair_uses_one_fixed_shape_program(self):
    raw = jnp.arange(4 * 513, dtype=jnp.float32).reshape(4, 513)
    processed = raw * jnp.float32(0.5)
    observe = jax.jit(capture.fingerprint_terminal_pair)
    result = jax.device_get(observe(raw, processed))
    self.assertEqual(len(result), 10)
    self.assertEqual(result[0].shape, (4, 3, 6))
    self.assertEqual(result[5].shape, (4, 3, 6))


if __name__ == "__main__":
  unittest.main()

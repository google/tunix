#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/engine_shims/p38_kv_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("p38_kv_fingerprint", MODULE)
assert SPEC is not None and SPEC.loader is not None
fingerprint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fingerprint)


class KvFingerprintTest(unittest.TestCase):

  def setUp(self):
    self.shape = (2, 4, 1, 2, 4)
    self.extents = fingerprint.validate_kv_fingerprint_contract(
        self.shape, jnp.bfloat16, [4, 2]
    )
    values = jnp.arange(np.prod(self.shape), dtype=jnp.float32).reshape(
        self.shape
    )
    self.pages = values.astype(jnp.bfloat16)
    self.fn = jax.jit(fingerprint.fingerprint_kv_pages)

  def test_repeat_exact_and_one_bit_negative(self):
    first = jax.device_get(self.fn(self.pages, jnp.asarray(self.extents)))
    repeat = jax.device_get(self.fn(self.pages, jnp.asarray(self.extents)))
    for left, right in zip(first, repeat, strict=True):
      self.assertTrue(np.array_equal(left, right))

    bits = jax.lax.bitcast_convert_type(self.pages, jnp.uint16)
    poisoned_bits = bits.at[0, 1, 0, 0, 0].set(bits[0, 1, 0, 0, 0] ^ 1)
    poisoned = jax.lax.bitcast_convert_type(poisoned_bits, jnp.bfloat16)
    changed = jax.device_get(self.fn(poisoned, jnp.asarray(self.extents)))
    self.assertFalse(np.array_equal(first[0][0], changed[0][0]))

  def test_invalid_tail_is_masked(self):
    base = jax.device_get(self.fn(self.pages, jnp.asarray(self.extents)))
    bits = jax.lax.bitcast_convert_type(self.pages, jnp.uint16)
    tail_bits = bits.at[1, 3, 0, 0, 0].set(bits[1, 3, 0, 0, 0] ^ 1)
    tail = jax.lax.bitcast_convert_type(tail_bits, jnp.bfloat16)
    observed = jax.device_get(self.fn(tail, jnp.asarray(self.extents)))
    for left, right in zip(base, observed, strict=True):
      self.assertTrue(np.array_equal(left, right))

  def test_contract_rejects_invalid_geometry(self):
    with self.assertRaisesRegex(ValueError, "rank-5"):
      fingerprint.validate_kv_fingerprint_contract(
          (2, 4, 8), jnp.bfloat16, [4, 4]
      )
    with self.assertRaisesRegex(ValueError, "bfloat16"):
      fingerprint.validate_kv_fingerprint_contract(
          self.shape, jnp.float32, [4, 4]
      )
    with self.assertRaisesRegex(ValueError, "exceed page size"):
      fingerprint.validate_kv_fingerprint_contract(
          self.shape, jnp.bfloat16, [4, 5]
      )

  def test_dp_local_pages_map_to_the_global_cache_axis(self):
    indices = fingerprint.global_page_indices(
        (16, 4, 1, 2, 4), 4, 2, [0, 3]
    )
    self.assertTrue(np.array_equal(indices, np.array([8, 11])))
    with self.assertRaisesRegex(ValueError, "exceeds local capacity"):
      fingerprint.global_page_indices((16, 4, 1, 2, 4), 4, 2, [4])
    with self.assertRaisesRegex(ValueError, "not divisible"):
      fingerprint.global_page_indices((15, 4, 1, 2, 4), 4, 2, [0])

  def test_layer_observer_is_bounded_and_detects_a_valid_bit(self):
    cache_shape = (8, 4, 1, 2, 4)
    values = jnp.arange(np.prod(cache_shape), dtype=jnp.float32).reshape(
        cache_shape
    )
    layer0 = values.astype(jnp.bfloat16)
    layer1 = (values + 1).astype(jnp.bfloat16)
    pages = fingerprint.global_page_indices(cache_shape, 2, 1, [0, 2])
    extents = fingerprint.validate_kv_fingerprint_contract(
        (2, *cache_shape[1:]), jnp.bfloat16, [4, 2]
    )
    observe = jax.jit(fingerprint.fingerprint_kv_cache_layers)
    base = jax.device_get(observe(
        (layer0, layer1), jnp.asarray(pages), jnp.asarray(extents)
    ))
    bits = jax.lax.bitcast_convert_type(layer1, jnp.uint16)
    poisoned_bits = bits.at[4, 1, 0, 0, 0].set(bits[4, 1, 0, 0, 0] ^ 1)
    poisoned = jax.lax.bitcast_convert_type(poisoned_bits, jnp.bfloat16)
    changed = jax.device_get(observe(
        (layer0, poisoned), jnp.asarray(pages), jnp.asarray(extents)
    ))
    self.assertTrue(np.array_equal(base[0][0], changed[0][0]))
    self.assertFalse(np.array_equal(base[0][1], changed[0][1]))
    self.assertEqual(
        fingerprint.estimate_fingerprint_read_bytes(
            [cache_shape, cache_shape], 2
        ),
        2 * 2 * np.prod(cache_shape[1:]) * 2,
    )

  def test_prefix_table_matches_direct_fingerprints(self):
    cache_shape = (8, 4, 1, 2, 4)
    values = jnp.arange(np.prod(cache_shape), dtype=jnp.float32).reshape(
        cache_shape
    )
    caches = (
        values.astype(jnp.bfloat16),
        (values + 3).astype(jnp.bfloat16),
    )
    pages = fingerprint.global_page_indices(cache_shape, 2, 1, [0, 2])
    extents = jnp.asarray([4, 2], dtype=jnp.int32)
    prefix_observer = jax.jit(
        fingerprint.fingerprint_kv_cache_layer_prefixes
    )
    direct_observer = jax.jit(fingerprint.fingerprint_kv_cache_layers)
    table = jax.device_get(prefix_observer(caches, jnp.asarray(pages)))
    direct = jax.device_get(
        direct_observer(caches, jnp.asarray(pages), extents)
    )
    for layer in range(len(caches)):
      for page, extent in enumerate(np.asarray(extents)):
        self.assertTrue(np.array_equal(
            table[0][layer, page, extent - 1], direct[0][layer, page]
        ))
        self.assertTrue(np.array_equal(
            table[1][layer, page, extent - 1], direct[1][layer, page]
        ))


if __name__ == "__main__":
  unittest.main()

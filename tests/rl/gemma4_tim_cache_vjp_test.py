"""Independent logical-cache FP64 controls; not an installed-RPA certificate."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.frozenlake.gemma4_tim import cache_vjp as candidate


def logical_oracle(q, k, v, cache, lengths, tables, bounds, *, write, window):
  """Static small-fixture oracle: per-sequence logical arrays and per-row dot.

  Intentionally does not reuse candidate scatter, mask, softmax or einsum.
  Host metadata is constant in this test oracle, never production routing.
  """
  result = jnp.zeros_like(q)
  new_cache = cache
  page_size = cache.shape[1]
  repeat = q.shape[1] // k.shape[1]
  for seq, length in enumerate(lengths):
    start, end = map(int, bounds[seq:seq + 2])
    if start == end:
      continue
    context = int(length) - (end - start)
    old_k, old_v = [], []
    for pos in range(int(length)):
      pg, off = int(tables[seq, pos // page_size]), pos % page_size
      if write and pos >= context:
        kr, vr = k[start + pos - context], v[start + pos - context]
        new_cache = new_cache.at[pg, off, :, 0].set(kr)
        new_cache = new_cache.at[pg, off, :, 1].set(vr)
      else:
        kr, vr = cache[pg, off, :, 0], cache[pg, off, :, 1]
      old_k.append(kr)
      old_v.append(vr)
    keys, values = jnp.stack(old_k), jnp.stack(old_v)
    for row in range(start, end):
      position = context + row - start
      left = max(0, position - window + 1) if window else 0
      for head in range(q.shape[1]):
        scores = keys[left:position + 1, head // repeat] @ q[row, head]
        exp = jnp.exp(scores - jnp.max(scores))
        probabilities = exp / jnp.sum(exp)
        value = probabilities @ values[left:position + 1, head // repeat]
        result = result.at[row, head].set(value)
  return result, new_cache


class CacheVjpTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Scoped x64 context is entered per test, avoiding changes to other gates.
    cls.lengths = np.array([5, 3, 0], np.int32)
    cls.tables = np.array([[4, 1, 5], [2, 0, -1], [-1, -1, -1]], np.int32)
    cls.bounds = np.array([0, 2, 3, 3], np.int32)

  def setUp(self):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    self.addCleanup(jax.config.update, "jax_enable_x64", previous)
    rng = np.random.default_rng(39)
    self.operands = tuple(jnp.asarray(rng.normal(size=shape) * .2) for shape in
                          ((4, 4, 3), (4, 1, 3), (4, 1, 3), (7, 2, 1, 2, 3)))
    self.cotangents = tuple(jnp.asarray(rng.normal(size=shape)) for shape in
                            (self.operands[0].shape, self.operands[3].shape))
    candidate.validate_metadata(token_capacity=4, cache_shape=self.operands[3].shape,
                                kv_lens=self.lengths, page_tables=self.tables,
                                cu_q_lens=self.bounds)

  def oracle(self, *, write, window):
    return lambda *values: logical_oracle(*values, self.lengths, self.tables,
                                          self.bounds, write=write, window=window)

  def wrapped(self, *, write, window):
    # This is an independent CPU kernel, not Mosaic RPA.
    kernel = lambda *args: logical_oracle(*args[:4], self.lengths, self.tables,
                                          self.bounds, write=write, window=window)
    wrapped = candidate.make_cache_vjp(kernel, update_cache=write, window=window,
                                       compute_dtype=jnp.float64)
    return lambda *values: wrapped(*values, jnp.asarray(self.lengths),
                                    jnp.asarray(self.tables), jnp.asarray(self.bounds))

  def assert_gradients(self, actual, expected):
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
      a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
      self.assertTrue(np.isfinite(a).all())
      norm = np.linalg.norm(b)
      if norm == 0:
        np.testing.assert_array_equal(a, b)
      else:
        relative = np.linalg.norm(a - b) / norm
        self.assertGreater(np.linalg.norm(a), 0, index)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * norm)
        self.assertLessEqual(relative, 1e-5, (index, relative))
        self.assertGreaterEqual(cosine, .999999, (index, cosine))

  def test_four_modes_primal_and_both_output_pullbacks(self):
    for write in (True, False):
      for window in (None, 2):
        with self.subTest(write=write, window=window):
          oracle = self.oracle(write=write, window=window)
          wrapped = self.wrapped(write=write, window=window)
          expected, oracle_pullback = jax.vjp(oracle, *self.operands)
          actual, pullback = jax.vjp(wrapped, *self.operands)
          # Kernel identity compares to the supplied kernel's plain primal,
          # not to an AD-transformed oracle (which can change FP arithmetic).
          for a, b in zip(actual, oracle(*self.operands), strict=True):
            np.testing.assert_array_equal(a, b)
          for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
          dense = candidate.dense_paged_reference(
              *self.operands, self.lengths, self.tables, self.bounds,
              update_cache=write, window=window, compute_dtype=jnp.float64)
          for a, b in zip(dense, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
          gradients = pullback(self.cotangents)
          self.assert_gradients(gradients, oracle_pullback(self.cotangents))
          for leaf in gradients[:3]:
            np.testing.assert_array_equal(leaf[3], 0)  # Dead query row.
          if not write:
            np.testing.assert_array_equal(actual[1], self.operands[3])
            np.testing.assert_array_equal(gradients[1], 0)
            np.testing.assert_array_equal(gradients[2], 0)

  def test_adjacent_finite_difference_directions(self):
    rng = np.random.default_rng(99)
    directions = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in self.operands)
    for write in (True, False):
      wrapped = self.wrapped(write=write, window=2)
      def loss(*xs):
        return sum(jnp.vdot(a, g) for a, g in zip(wrapped(*xs), self.cotangents))
      grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*self.operands)
      directional = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, directions))
      self.assertGreater(abs(directional), 1e-6)
      for eps in (1e-4, 1e-5):
        plus = tuple(x + eps * d for x, d in zip(self.operands, directions))
        minus = tuple(x - eps * d for x, d in zip(self.operands, directions))
        fd = float((loss(*plus) - loss(*minus)) / (2 * eps))
        self.assertLess(abs(fd - directional) / abs(directional), 1e-5)

  def test_missing_cache_cotangent_negative(self):
    _, pullback = jax.vjp(self.wrapped(write=True, window=2), *self.operands)
    healthy = pullback(self.cotangents)
    broken = pullback((self.cotangents[0], jnp.zeros_like(self.cotangents[1])))
    with self.assertRaises(AssertionError):
      self.assert_gradients(broken, healthy)
    for i in (1, 2, 3):
      self.assertGreater(float(jnp.linalg.norm(broken[i] - healthy[i])), .1)

  def test_readonly_detach_and_overwrite_negatives(self):
    wrapped = self.wrapped(write=False, window=2)
    def detached(q, k, v, cache):
      return wrapped(q, k, v, jax.lax.stop_gradient(cache))
    healthy_result, pb = jax.vjp(wrapped, *self.operands)
    broken_result, broken_pb = jax.vjp(detached, *self.operands)
    for a, b in zip(healthy_result, broken_result):
      np.testing.assert_array_equal(a, b)
    with self.assertRaises(AssertionError):
      self.assert_gradients(broken_pb(self.cotangents), pb(self.cotangents))
    overwritten = self.wrapped(write=True, window=2)(*self.operands)
    self.assertGreater(float(jnp.linalg.norm(overwritten[1] - healthy_result[1])), .1)

  def test_missing_window_negative(self):
    for write in (True, False):
      good, good_pb = jax.vjp(self.wrapped(write=write, window=2), *self.operands)
      bad, bad_pb = jax.vjp(self.wrapped(write=write, window=None), *self.operands)
      self.assertGreater(float(jnp.linalg.norm(good[0] - bad[0])), .01)
      with self.assertRaises(AssertionError):
        self.assert_gradients(bad_pb(self.cotangents), good_pb(self.cotangents))

  def test_all_dead_rows_preserve_cache_and_cotangent(self):
    q, k, v, cache = self.operands
    lengths = jnp.array([0], jnp.int32)
    tables = jnp.full((1, 3), -1, jnp.int32)
    bounds = jnp.array([0, 0], jnp.int32)
    for write in (True, False):
      def dense(*values):
        return candidate.dense_paged_reference(*values, lengths, tables, bounds,
                    update_cache=write, window=2, compute_dtype=jnp.float64)
      (out, new_cache), pb = jax.vjp(dense, q, k, v, cache)
      np.testing.assert_array_equal(out, 0)
      np.testing.assert_array_equal(new_cache, cache)
      derivatives = pb(self.cotangents)
      for value in derivatives[:3]:
        np.testing.assert_array_equal(value, 0)
      np.testing.assert_array_equal(derivatives[3], self.cotangents[1])

  def test_metadata_rejects_active_alias_and_bad_lengths(self):
    kwargs = dict(token_capacity=4, cache_shape=self.operands[3].shape,
                  kv_lens=self.lengths, page_tables=self.tables, cu_q_lens=self.bounds)
    for replacement in (
        {"page_tables": np.zeros_like(self.tables)},
        {"page_tables": self.tables.astype(float)},
        {"kv_lens": np.array([7, 3, 0], np.int32)},
        {"cu_q_lens": np.array([0, 3, 2, 3], np.int32)},
        {"cu_q_lens": np.array([0, 2, 3, 5], np.int32)},
    ):
      with self.assertRaises(ValueError):
        candidate.validate_metadata(**(kwargs | replacement))

  def test_checked_four_cpu_head_replication_pullback(self):
    self.assertEqual(len(jax.devices()), 4)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("tp",))
    P = jax.sharding.PartitionSpec
    for write in (True, False):
      def kernel(q, k, v, cache, lengths, tables, bounds):
        return candidate.dense_paged_reference(q, k, v, cache, lengths, tables, bounds,
                    update_cache=write, window=2, compute_dtype=jnp.float64)
      local = candidate.make_cache_vjp(kernel, update_cache=write, window=2,
                                        compute_dtype=jnp.float64)
      mapped = jax.shard_map(local, mesh=mesh,
          in_specs=(P(None, "tp", None), P(None, "tp", None), P(None, "tp", None),
                    P(None, None, "tp", None, None), P(), P(), P()),
          out_specs=(P(None, "tp", None), P(None, None, "tp", None, None)), check_vma=True)
      metadata = tuple(map(jnp.asarray, (self.lengths, self.tables, self.bounds)))
      def loss(q, k, v, cache):
        out, _ = mapped(q, jnp.repeat(k, 4, axis=1), jnp.repeat(v, 4, axis=1),
                         jnp.repeat(cache, 4, axis=2), *metadata)
        return jnp.vdot(out, self.cotangents[0])
      def ordinary(*values):
        return jnp.vdot(self.oracle(write=write, window=2)(*values)[0], self.cotangents[0])
      actual, grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(*self.operands)
      expected, oracle_grad = jax.value_and_grad(ordinary, argnums=(0, 1, 2, 3))(*self.operands)
      np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
      self.assert_gradients(grad, oracle_grad)


if __name__ == "__main__":
  unittest.main()

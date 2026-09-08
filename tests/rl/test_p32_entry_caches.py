"""Entry-cache rebuild rule for the grouped reverse (S2c).

The ragged paged attention forward of chunk ``c`` writes only its own row
positions ``[c*bucket, ...)`` and never reads a cache slot at or beyond
``chunk_start``, so the caches a chunk started from equal the group's final
caches with every later position zeroed.  These tests pin that rule on a
paged layout ``(data_size * blocks_per_req, block_size, ...)`` including a
block size that does not divide the bucket, the compile-once property
(``chunk_start`` is a runtime operand), and dtype/sharding preservation.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter_module


def _paged_caches(data_size, blocks_per_req, block_size, heads, dtype, layers):
  """Caches whose value encodes (rank, position, layer) so zeroing is visible."""
  caches = []
  for layer in range(layers):
    pages = np.arange(data_size * blocks_per_req)
    rank = pages // blocks_per_req
    position = (pages % blocks_per_req)[:, None] * block_size + np.arange(block_size)[None, :]
    value = 1.0 + rank[:, None] * 1000.0 + position + 0.01 * layer
    value = np.broadcast_to(value[:, :, None], (len(pages), block_size, heads))
    caches.append(jnp.asarray(value.astype(dtype)))
  return tuple(caches)


def _reference(caches, chunk_start, data_size):
  out = []
  for cache in caches:
    arr = np.asarray(cache)
    blocks_per_req = arr.shape[0] // data_size
    block_size = arr.shape[1]
    pages = np.arange(arr.shape[0])
    position = (pages % blocks_per_req)[:, None] * block_size + np.arange(block_size)[None, :]
    keep = position < chunk_start
    out.append(np.where(keep[:, :, None], arr, np.zeros((), arr.dtype)))
  return tuple(out)


@pytest.mark.parametrize(
    "block_size,bucket",
    [(4, 8), (8, 8), (3, 8)],
    ids=["block-divides-bucket", "block-equals-bucket", "block-does-not-divide"],
)
def test_entry_caches_match_the_zeroed_final_caches(block_size, bucket):
  data_size, blocks_per_req, heads = 2, 5, 3
  caches = _paged_caches(data_size, blocks_per_req, block_size, heads, np.float32, layers=2)
  max_position = blocks_per_req * block_size
  for chunk_index in range((max_position + bucket - 1) // bucket + 1):
    chunk_start = chunk_index * bucket
    rebuilt = adapter_module._p32_entry_caches(  # pylint: disable=protected-access
        caches, chunk_start, data_size=data_size
    )
    expected = _reference(caches, chunk_start, data_size)
    assert len(rebuilt) == len(expected)
    for got, want in zip(rebuilt, expected):
      assert got.dtype == want.dtype and got.shape == want.shape
      assert np.asarray(got).tobytes() == want.tobytes()
  # chunk 0 starts from the fresh zero caches ...
  first = adapter_module._p32_entry_caches(caches, 0, data_size=data_size)  # pylint: disable=protected-access
  assert all(not np.asarray(c).any() for c in first)
  # ... and a start past every position returns the final caches unchanged.
  whole = adapter_module._p32_entry_caches(caches, max_position, data_size=data_size)  # pylint: disable=protected-access
  for got, want in zip(whole, caches):
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


def test_entry_caches_compile_once_across_chunks_and_keep_dtype_and_sharding():
  data_size = 2
  caches = _paged_caches(data_size, 4, 4, 2, jnp.bfloat16, layers=3)
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")),
      jax.sharding.PartitionSpec("data"),
  )
  caches = tuple(jax.device_put(c, sharding) for c in caches)
  before = len(adapter_module._P32_ENTRY_CACHE_PROGRAMS)  # pylint: disable=protected-access
  outputs = [
      adapter_module._p32_entry_caches(caches, start, data_size=data_size)  # pylint: disable=protected-access
      for start in (0, 4, 8, 12, 16)
  ]
  after = len(adapter_module._P32_ENTRY_CACHE_PROGRAMS)  # pylint: disable=protected-access
  assert after == before + 1, "one program serves every chunk_start"
  key = next(k for k in adapter_module._P32_ENTRY_CACHE_PROGRAMS if k[3] == data_size)  # pylint: disable=protected-access
  program = adapter_module._P32_ENTRY_CACHE_PROGRAMS[key]  # pylint: disable=protected-access
  assert program._cache_size() == 1  # pylint: disable=protected-access
  for rebuilt in outputs:
    for got in rebuilt:
      assert got.dtype == jnp.bfloat16
      assert got.sharding.is_equivalent_to(sharding, got.ndim)


def test_entry_caches_refuse_a_non_paged_cache():
  with pytest.raises(adapter_module.FunctionalMappingError, match="paged cache"):
    adapter_module._p32_entry_caches(  # pylint: disable=protected-access
        (jnp.zeros((5, 4, 2), jnp.float32),), 4, data_size=2
    )


def test_entry_caches_zero_carry_emits_zero_cotangents_with_the_cache_shardings():
  """tasks/v2_dispatch Phase 11: the first reversed chunk's rebuild also
  returns the zero cache cotangents the reverse starts from, shaped,
  typed and sharded like the caches, from one program cached apart from
  the plain rebuild."""
  data_size = 2
  caches = _paged_caches(data_size, 4, 4, 2, jnp.bfloat16, layers=3)
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")),
      jax.sharding.PartitionSpec("data"),
  )
  caches = tuple(jax.device_put(c, sharding) for c in caches)
  plain = adapter_module._p32_entry_caches(caches, 8, data_size=data_size)  # pylint: disable=protected-access
  before = len(adapter_module._P32_ENTRY_CACHE_PROGRAMS)  # pylint: disable=protected-access
  rebuilt, zeros = adapter_module._p32_entry_caches(  # pylint: disable=protected-access
      caches, 8, data_size=data_size, zero_carry=True
  )
  again, zeros_again = adapter_module._p32_entry_caches(  # pylint: disable=protected-access
      caches, 4, data_size=data_size, zero_carry=True
  )
  after = len(adapter_module._P32_ENTRY_CACHE_PROGRAMS)  # pylint: disable=protected-access
  assert after == before + 1, "the zero-carry variant is one more cached program"
  assert len(rebuilt) == len(zeros) == len(caches)
  for got, want in zip(rebuilt, plain):
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
  for zero, cache in zip(zeros + zeros_again, caches + caches):
    assert zero.shape == cache.shape and zero.dtype == cache.dtype
    assert zero.sharding.is_equivalent_to(cache.sharding, zero.ndim)
    assert np.asarray(zero).tobytes() == np.asarray(jnp.zeros_like(cache)).tobytes()
  del again

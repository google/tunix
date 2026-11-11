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

"""Checks the GPU (vLLM Triton) path against the pure-JAX path."""

import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import paged_attention

jax.config.update("jax_num_cpu_devices", 4)

_CACHE = "cache"


def _inputs(q_lens, kv_lens=None, *, num_q_heads=8, num_kv_heads=4,
            head_dim=128, page_size=16, pages_per_seq=8, packing=2,
            max_num_tokens=None, active=None, dtype=jnp.bfloat16, seed=0):
  kv_lens = kv_lens or q_lens
  num_seqs = len(q_lens)
  active = num_seqs if active is None else active
  max_num_tokens = max_num_tokens or sum(q_lens)
  rng = np.random.default_rng(seed)
  pages = num_seqs * pages_per_seq + 3
  table = np.full((num_seqs, pages_per_seq), -1, np.int32)
  free = rng.permutation(pages)
  for s, n in enumerate(kv_lens):
    used = -(-n // page_size)
    table[s, :used], free = free[:used], free[used:]
  rand = lambda *shape: jnp.asarray(rng.normal(size=shape), dtype)
  return dict(
      queries=rand(max_num_tokens, num_q_heads, head_dim),
      keys=rand(max_num_tokens, num_kv_heads, head_dim),
      values=rand(max_num_tokens, num_kv_heads, head_dim),
      kv_cache=rand(pages, page_size, 2 * num_kv_heads // packing, packing,
                    head_dim),
      metadata=paged_attention.PagedAttnMetadata(
          page_indices={_CACHE: jnp.asarray(table)},
          kv_lens=jnp.asarray(kv_lens, jnp.int32),
          query_lens=jnp.asarray(q_lens, jnp.int32),
          distribution=jnp.asarray([[0, 0, active]], jnp.int32),
      ),
  )


def _one_device_mesh(backend=None):
  return jax.sharding.Mesh(np.array(jax.devices(backend)[:1]), ("tp",))


class PagedAttentionGpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("needs a GPU")

  def _check(self, inputs, **kwargs):
    mesh = _one_device_mesh("gpu")
    run = lambda backend: jax.jit(
        lambda x: paged_attention.paged_attention(
            **x, cache_name=_CACHE, mesh=mesh, backend=backend, **kwargs))(
                inputs)
    gpu_out, gpu_cache = run("gpu")
    ref_out, ref_cache = run("cpu")  # Pure-JAX path, still on the GPU.
    np.testing.assert_array_equal(np.asarray(gpu_cache), np.asarray(ref_cache))
    np.testing.assert_allclose(np.asarray(gpu_out, np.float32),
                               np.asarray(ref_out, np.float32),
                               rtol=2e-2, atol=2e-2)

  @parameterized.named_parameters(
      ("prefill", [5, 23, 1, 11], None),
      ("prefill_with_prefix", [8, 8, 8, 8], [40, 33, 8, 17]),
      ("empty_sequence", [12, 0, 9, 4], [12, 5, 9, 4]),
  )
  def test_prefill(self, q_lens, kv_lens):
    self._check(_inputs(q_lens, kv_lens))

  @parameterized.named_parameters(("3d", 4), ("2d", 64))
  def test_decode(self, num_seqs):
    kv_lens = [(7 * i) % 120 + 1 for i in range(num_seqs)]
    self._check(_inputs([1] * num_seqs, kv_lens), decode_only=True)

  def test_padding_and_inactive(self):
    self._check(_inputs([6, 10, 4, 4], max_num_tokens=40, active=3))

  @parameterized.parameters(1, 7, 64)
  def test_sliding_window(self, window):
    self._check(_inputs([5, 23, 1, 11], [40, 33, 8, 17]),
                sliding_window=window)

  def test_evicted_sliding_window_pages_with_negative_indices(self):
    inputs = _inputs([4, 2], [40, 33], page_size=16, pages_per_seq=4)
    # Evict the first page of each sequence (positions 0..15 are outside
    # window=8) by setting its page-table entry to -1.
    table = np.asarray(inputs["metadata"].page_indices[_CACHE]).copy()
    table[:, 0] = -1
    inputs["metadata"] = inputs["metadata"].replace(
        page_indices={_CACHE: jnp.asarray(table)}
    )
    self._check(inputs, sliding_window=8)

  def test_softcap_and_scale(self):
    self._check(_inputs([5, 23, 1, 11]), sm_scale=0.1, soft_cap=5.0)

  def test_shared_kv(self):
    self._check(_inputs([5, 23, 1, 11], [40, 33, 8, 17]),
                update_kv_cache=False)

  @parameterized.parameters((8, 8), (8, 1), (16, 2))
  def test_head_geometry(self, num_q_heads, num_kv_heads):
    self._check(_inputs([5, 23], num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads))

  @parameterized.parameters(256, 512)
  def test_gemma4_head_dims(self, head_dim):
    self._check(_inputs([5, 23], head_dim=head_dim))
    self._check(_inputs([1, 1], [30, 70], head_dim=head_dim),
                decode_only=True)

  @parameterized.product(
      head_dim=(256, 512),
      sliding_window=(None, 512),
      decode_only=(False, True),
      update_kv_cache=(True, False),
  )
  def test_compiles(self, head_dim, sliding_window, decode_only,
                    update_kv_cache):
    """Both Triton kernels compile for Gemma4-like configs; nothing runs."""
    q_lens = [1] * 8 if decode_only else [64, 64]
    inputs = _inputs(q_lens, [128] * len(q_lens), num_q_heads=8,
                     num_kv_heads=2, head_dim=head_dim, pages_per_seq=16)
    mesh = _one_device_mesh("gpu")
    jax.jit(
        lambda x: paged_attention.paged_attention(
            **x, cache_name=_CACHE, mesh=mesh, backend="gpu",
            decode_only=decode_only, sliding_window=sliding_window,
            update_kv_cache=update_kv_cache)
    ).lower(inputs).compile()


class PagedAttentionJaxTest(parameterized.TestCase):

  def test_negative_page_indices_clamp_to_page_zero(self):
    mesh = _one_device_mesh()
    inputs = _inputs([4, 2], [40, 33], page_size=16, pages_per_seq=4)
    # Poison the last physical page with NaNs so that an unclamped -1 index
    # (which wraps to the last page in JAX gather) would produce NaNs via
    # `0.0 * NaN` in the attention value weighted sum.
    run = jax.jit(
        lambda x: paged_attention.paged_attention(
            **x, cache_name=_CACHE, mesh=mesh, backend="cpu", sliding_window=8
        )
    )
    inputs["kv_cache"] = inputs["kv_cache"].at[-1].set(jnp.nan)
    ref_out, _ = run(inputs)

    table = np.asarray(inputs["metadata"].page_indices[_CACHE]).copy()
    table[:, 0] = -1
    evicted_inputs = dict(
        inputs,
        metadata=inputs["metadata"].replace(
            page_indices={_CACHE: jnp.asarray(table)}
        ),
    )
    out, _ = run(evicted_inputs)

    self.assertFalse(bool(jnp.any(jnp.isnan(out))))
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(ref_out, np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.parameters(
      # Full TP sharding across Q, KV, and KV cache.
      (8, 4, 2, 4),
      # MQA on tp=2: KV head is padded to tp_size=2 and sharded across TP.
      (4, 1, 1, 2),
      (4, 1, 2, 2),
      # GQA with 1 < num_kv_heads < tp: KV heads are padded to tp_size=4.
      (8, 2, 1, 4),
      (8, 2, 2, 4),
  )
  def test_mesh_sharding_matches_unsharded(
      self, num_q_heads, num_kv_heads, packing, tp_size
  ):
    def _pad_cache_for_tp(cache):
      if num_kv_heads >= tp_size:
        return cache
      pages, page_size, _, pack, head_dim = cache.shape
      reps = tp_size // num_kv_heads
      unpacked = cache.reshape(pages, page_size, num_kv_heads, 2, head_dim)
      padded = jnp.repeat(unpacked, reps, axis=2)
      return padded.reshape(
          pages, page_size, 2 * tp_size // pack, pack, head_dim
      )

    cpu_devices = jax.devices("cpu")[:tp_size]
    self.assertLen(cpu_devices, tp_size)
    inputs = jax.device_put(
        _inputs(
            [5, 7],
            [12, 15],
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            packing=packing,
            head_dim=32,
            dtype=jnp.float32,
        ),
        cpu_devices[0],
    )
    ref_mesh = jax.sharding.Mesh(np.array(cpu_devices[:1]), ("tp",))
    ref_out, ref_cache = jax.jit(
        lambda x: paged_attention.paged_attention(
            **x, cache_name=_CACHE, mesh=ref_mesh, backend="cpu"
        )
    )(inputs)

    mesh = jax.sharding.Mesh(np.array(cpu_devices), ("tp",))
    sharded_inputs = dict(
        inputs, kv_cache=_pad_cache_for_tp(inputs["kv_cache"])
    )
    sharded_inputs = jax.device_put(
        sharded_inputs,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    sharded_out, sharded_cache = jax.jit(
        lambda x: paged_attention.paged_attention(
            **x,
            cache_name=_CACHE,
            mesh=mesh,
            backend="cpu",
        )
    )(sharded_inputs)

    np.testing.assert_allclose(sharded_out, ref_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        sharded_cache, _pad_cache_for_tp(ref_cache), rtol=1e-5, atol=1e-5
    )

  def test_indivisible_kv_heads_for_tp_raises(self):
    cpu_devices = jax.devices("cpu")[:4]
    mesh = jax.sharding.Mesh(np.array(cpu_devices), ("tp",))
    inputs = _inputs(
        [5, 7],
        [12, 15],
        num_q_heads=12,
        num_kv_heads=3,
        packing=1,
        head_dim=32,
        dtype=jnp.float32,
    )
    with self.assertRaisesRegex(ValueError, r"tp_size \(4\) must be divisible"):
      paged_attention.paged_attention(
          **inputs, cache_name=_CACHE, mesh=mesh, backend="cpu"
      )

  def test_metadata_decode_only_is_static_and_forwarded_to_gpu(self):
    inputs = _inputs([1, 1], [12, 15], head_dim=32, dtype=jnp.float32)
    self.assertIs(inputs["metadata"].decode_only, False)
    decode_metadata = jax.device_put(
        inputs["metadata"].replace(decode_only=True), jax.devices("cpu")[0]
    )
    self.assertIs(decode_metadata.decode_only, True)

    mesh = _one_device_mesh("cpu")
    with mock.patch(
        "tunix.models.paged_attention_vllm_gpu", create=True
    ) as mock_gpu:
      mock_gpu.paged_attention.side_effect = (
          lambda q, k, v, cache, *args, **kwargs: (q, cache)
      )
      run = jax.jit(
          lambda x: paged_attention.paged_attention(
              **x, cache_name=_CACHE, mesh=mesh, backend="gpu"
          )
      )
      run(inputs)
      self.assertIs(
          mock_gpu.paged_attention.call_args.kwargs["decode_only"], False
      )

      run(dict(inputs, metadata=decode_metadata))
      self.assertIs(
          mock_gpu.paged_attention.call_args.kwargs["decode_only"], True
      )

      paged_attention.paged_attention(
          **inputs,
          cache_name=_CACHE,
          mesh=mesh,
          backend="gpu",
          decode_only=True,
      )
      self.assertIs(
          mock_gpu.paged_attention.call_args.kwargs["decode_only"], True
      )


_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 8
_PAGE_SIZE = 8
_PAGES_PER_SEQ = 4
_PAGES_PER_GROUP = 16
# Padding rows and tokens past the groups' own.
_NUM_PADDING_ROWS = 2
_NUM_PADDING_TOKENS = 3


def _group_inputs(rng, rows):
  """Returns a DP group's inputs, as if it ran alone against its own pages.

  Args:
    rng: the random generator.
    rows: `(query_len, kv_len)` of each row, decodes first.

  Returns:
    The group's inputs to `paged_attention`.
  """
  table = np.full((len(rows), _PAGES_PER_SEQ), -1, np.int32)
  free = rng.permutation(_PAGES_PER_GROUP)
  for row, (_, kv_len) in enumerate(rows):
    used = -(-kv_len // _PAGE_SIZE)
    table[row, :used], free = free[:used], free[used:]
  rand = lambda *shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
  num_tokens = sum(q_len for q_len, _ in rows)
  num_decodes = sum(q_len == 1 for q_len, _ in rows)
  return dict(
      queries=rand(num_tokens, _NUM_Q_HEADS, _HEAD_DIM),
      keys=rand(num_tokens, _NUM_KV_HEADS, _HEAD_DIM),
      values=rand(num_tokens, _NUM_KV_HEADS, _HEAD_DIM),
      # float32 packs one element per word.
      kv_cache=rand(
          _PAGES_PER_GROUP, _PAGE_SIZE, 2 * _NUM_KV_HEADS, 1, _HEAD_DIM
      ),
      metadata=paged_attention.PagedAttnMetadata(
          page_indices={_CACHE: jnp.asarray(table)},
          kv_lens=jnp.asarray([kv for _, kv in rows], jnp.int32).reshape(-1),
          query_lens=jnp.asarray([q for q, _ in rows], jnp.int32).reshape(-1),
          distribution=jnp.asarray(
              [[num_decodes, num_decodes, len(rows)]], jnp.int32
          ),
      ),
  )


def _pad(x, size):
  return jnp.pad(x, [(0, size - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def _batch(groups):
  """Packs the groups' inputs into equal-sized slices with padding."""
  rows_per_group = (
      max(g["metadata"].kv_lens.shape[0] for g in groups) + _NUM_PADDING_ROWS
  )
  tokens_per_group = (
      max(g["queries"].shape[0] for g in groups) + _NUM_PADDING_TOKENS
  )
  cat = lambda xs, size: jnp.concatenate([_pad(x, size) for x in xs])
  tables = [
      jnp.pad(
          g["metadata"].page_indices[_CACHE],
          ((0, rows_per_group - g["metadata"].kv_lens.shape[0]), (0, 0)),
          constant_values=-1,
      ).astype(jnp.int32)
      for g in groups
  ]
  return dict(
      queries=cat([g["queries"] for g in groups], tokens_per_group),
      keys=cat([g["keys"] for g in groups], tokens_per_group),
      values=cat([g["values"] for g in groups], tokens_per_group),
      kv_cache=jnp.concatenate([g["kv_cache"] for g in groups]),
      metadata=paged_attention.PagedAttnMetadata(
          page_indices={_CACHE: jnp.concatenate(tables)},
          kv_lens=cat([g["metadata"].kv_lens for g in groups], rows_per_group),
          query_lens=cat(
              [g["metadata"].query_lens for g in groups], rows_per_group
          ),
          distribution=jnp.concatenate(
              [g["metadata"].distribution for g in groups]
          ),
      ),
  )


def _cache_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
  dp_axes = tuple(axis for axis in mesh.axis_names if axis in ("dp", "fsdp"))
  tp_axis = "tp" if "tp" in mesh.shape else None
  return jax.sharding.NamedSharding(
      mesh,
      jax.sharding.PartitionSpec(dp_axes, None, tp_axis, None, None),
  )


def _run(inputs, mesh, update_kv_cache):
  return jax.jit(
      lambda x: paged_attention.paged_attention(
          **x,
          cache_name=_CACHE,
          mesh=mesh,
          backend="cpu",
          update_kv_cache=update_kv_cache,
      )
  )(inputs)


class PagedAttentionShardingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < 4:
      self.skipTest("needs 4 CPU devices")

  def _mesh(self, mesh_shape):
    n = math.prod(mesh_shape.values())
    return jax.sharding.Mesh(
        np.array(jax.devices("cpu")[:n]).reshape(tuple(mesh_shape.values())),
        tuple(mesh_shape),
    )

  @parameterized.product(
      setup=(
          # 2 groups over "dp", each with its KV heads over "tp".
          dict(mesh_shape=dict(dp=2, tp=2), group_rows=[
              [(1, 9), (5, 21)],
              [(1, 4), (13, 13)],
          ]),
          # 4 groups, one of them empty.
          dict(mesh_shape=dict(dp=4), group_rows=[
              [(1, 9), (5, 21)],
              [],
              [(13, 13)],
              [(1, 5), (1, 30), (7, 7)],
          ]),
          # 4 groups over two DP axes.
          dict(mesh_shape=dict(dp=2, fsdp=2), group_rows=[
              [(3, 3)],
              [(1, 17), (2, 10)],
              [(1, 1)],
              [(6, 25)],
          ]),
      ),
      update_kv_cache=(True, False),
  )
  def test_each_group_matches_running_alone(self, setup, update_kv_cache):
    mesh = self._mesh(setup["mesh_shape"])
    ref_mesh = self._mesh(dict(dp=1))
    rng = np.random.default_rng(0)
    groups = [_group_inputs(rng, rows) for rows in setup["group_rows"]]
    num_groups = len(groups)
    batch = _batch(groups)
    cache_sharding = _cache_sharding(mesh)
    batch["kv_cache"] = jax.device_put(batch["kv_cache"], cache_sharding)

    out, cache = _run(batch, mesh, update_kv_cache)

    tokens_per_group = batch["queries"].shape[0] // num_groups
    for g, group in enumerate(groups):
      token_start = g * tokens_per_group
      num_tokens = group["queries"].shape[0]
      if num_tokens:
        ref_out, ref_cache = _run(group, ref_mesh, update_kv_cache)
        np.testing.assert_allclose(
            np.asarray(out[token_start : token_start + num_tokens]),
            np.asarray(ref_out),
            rtol=1e-5,
            atol=1e-5,
        )
      else:
        ref_cache = group["kv_cache"]
      np.testing.assert_array_equal(
          np.asarray(cache[g * _PAGES_PER_GROUP : (g + 1) * _PAGES_PER_GROUP]),
          np.asarray(ref_cache),
      )
      # Padding tokens within the group's slice are zeroed.
      np.testing.assert_array_equal(
          np.asarray(
              out[token_start + num_tokens : (g + 1) * tokens_per_group]
          ),
          0,
      )
    # The updated cache keeps the sharding of the cache that was passed in.
    self.assertTrue(cache.sharding.is_equivalent_to(cache_sharding, cache.ndim))

  def test_one_group_on_a_mesh_matches_unsharded(self):
    mesh = self._mesh(dict(dp=1, tp=2))
    ref_mesh = self._mesh(dict(dp=1))
    group = _group_inputs(np.random.default_rng(0), [(1, 9), (5, 21)])
    ref_out, ref_cache = _run(group, ref_mesh, True)
    sharded = dict(
        group,
        kv_cache=jax.device_put(group["kv_cache"], _cache_sharding(mesh)),
    )

    out, cache = _run(sharded, mesh, True)

    np.testing.assert_allclose(
        np.asarray(out), np.asarray(ref_out), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(ref_cache))

  def test_rejects_group_count_that_does_not_match_the_mesh(self):
    mesh = self._mesh(dict(dp=4))
    rng = np.random.default_rng(0)
    batch = _batch([_group_inputs(rng, [(3, 3)]) for _ in range(2)])
    with self.assertRaisesRegex(ValueError, "2 DP groups"):
      _run(batch, mesh, True)


if __name__ == "__main__":
  absltest.main()

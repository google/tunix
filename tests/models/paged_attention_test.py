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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from tunix.models import paged_attention


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
      metadata=paged_attention.RPAMetadata(
          page_indices={_CACHE: jnp.asarray(table)},
          kv_lens=jnp.asarray(kv_lens, jnp.int32),
          query_lens=jnp.asarray(q_lens, jnp.int32),
          distribution=jnp.asarray([0, 0, active], jnp.int32),
      ),
  )


class PagedAttentionGpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("needs a GPU")

  def _check(self, inputs, **kwargs):
    run = lambda backend: jax.jit(
        lambda x: paged_attention.ragged_paged_attention(
            **x, cache_name=_CACHE, backend=backend, **kwargs))(inputs)
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
    jax.jit(
        lambda x: paged_attention.ragged_paged_attention(
            **x, cache_name=_CACHE, backend="gpu", decode_only=decode_only,
            sliding_window=sliding_window, update_kv_cache=update_kv_cache)
    ).lower(inputs).compile()


if __name__ == "__main__":
  absltest.main()

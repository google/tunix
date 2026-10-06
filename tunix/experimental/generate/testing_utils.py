# Copyright 2026 The Tunix Authors.
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

"""Test helpers that build an `LLMEngine` around a toy transformer."""

from collections.abc import Mapping
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.generate import tokenizer_adapter

VOCAB_SIZE = 32
PAGE_SIZE = 4
GEOMETRIES = {
    'layer_0': kv_cache_manager_lib.CacheGeometry(num_kv_heads=1, head_dim=1)
}
# One page of one cache: `page_size` slots of a packed K and V float32 value.
BYTES_PER_PAGE = PAGE_SIZE * 2 * 4
GIB_PER_PAGE = BYTES_PER_PAGE / (1 << 30)


def mesh() -> jax.sharding.Mesh:
  return jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))


@dataclasses.dataclass(frozen=True)
class PagedSumConfig:
  dtype: jax.typing.DTypeLike = jnp.float32


class PagedSumTransformer(nnx.Module):
  """A stand-in transformer that reads and writes the paged KV cache.

  Each token writes `token + offset` into its KV slot, and its most likely
  successor is the sum of the values its row holds up to and including its
  own position, modulo the vocab size. The prediction only reads the cache,
  so a wrong page table, a wrong KV length or KV left over from other weights
  all change what the model generates.
  """

  def __init__(
      self,
      offset: jax.typing.ArrayLike = 0.0,
      dtype: jax.typing.DTypeLike = jnp.float32,
      geometries: Mapping[str, kv_cache_manager_lib.CacheGeometry] = GEOMETRIES,
  ):
    self.config = PagedSumConfig(dtype=dtype)
    self.offset = nnx.Param(jnp.asarray(offset, dtype=dtype))
    self.geometries = dict(geometries)

  def init_kv_cache(
      self, cache_config: kv_cache_manager_lib.CacheConfig
  ) -> kv_cache_manager_lib.KVCacheManager:
    """Builds one cache per entry of `geometries`."""
    return kv_cache_manager_lib.KVCacheManager(cache_config, self.geometries)

  def __call__(self, tokens, positions, cache, metadata, mesh):
    del mesh
    num_tokens = tokens.shape[0]
    ragged = model_runner_lib.RaggedArray(
        data=tokens, lens=jnp.asarray(metadata.query_lens)
    )
    rows = ragged.row_idxs
    is_real = jnp.arange(num_tokens) < jnp.sum(metadata.query_lens)

    new_cache = {}
    next_tokens = jnp.zeros((num_tokens,), dtype=jnp.int32)
    for name, pages in cache.items():
      page_table = jnp.asarray(metadata.page_indices[name])
      num_pages, page_size = pages.shape[:2]

      # Padding tokens point past the last page, so their writes are dropped.
      page_ids = jnp.where(
          is_real, page_table[rows, positions // page_size], num_pages
      )
      values = tokens.astype(pages.dtype) + self.offset.value
      pages = pages.at[page_ids, positions % page_size, 0, 0, 0].set(
          values, mode='drop'
      )
      new_cache[name] = pages

      # Unallocated page table entries only cover positions past the token,
      # which the mask below drops.
      row_values = pages[jnp.maximum(page_table[rows], 0)][..., 0, 0, 0]
      row_values = row_values.reshape(num_tokens, -1)
      kv_positions = jnp.arange(row_values.shape[1])
      total = jnp.sum(
          jnp.where(kv_positions[None, :] <= positions[:, None], row_values, 0),
          axis=1,
      )
      next_tokens = jnp.mod(jnp.round(total), VOCAB_SIZE).astype(jnp.int32)

    logits = 10.0 * jax.nn.one_hot(next_tokens, VOCAB_SIZE)
    return logits, new_cache


class WhitespaceTokenizer:
  """A tokenizer whose text is token ids separated by spaces, e.g. '1 2 3'."""

  def encode(self, text: str) -> list[int]:
    return [int(token) for token in text.split()]

  def decode(self, ids: list[int]) -> str:
    return ' '.join(str(token) for token in ids)

  def bos_id(self) -> int:
    return 0

  def eos_id(self) -> int:
    return 1

  def pad_id(self) -> int:
    return 0


def make_engine(
    transformer: nnx.Module | None = None,
    *,
    max_model_len: int = 24,
    max_device_size_gib: float = 64 * GIB_PER_PAGE,
    max_num_batched_tokens: int = 32,
    chunked_prefill_length: int = 32,
    num_scheduler_steps: int = 1,
    max_top_k: int = 4,
    return_logprobs: bool = False,
    return_logits: bool = False,
    eos_token_ids: frozenset[int] = frozenset(),
) -> engine_lib.LLMEngine:
  """Returns an engine running `transformer`, a `PagedSumTransformer` if None."""
  return engine_lib.LLMEngine(
      transformer if transformer is not None else PagedSumTransformer(),
      tokenizer=tokenizer_adapter.TokenizerAdapter(WhitespaceTokenizer()),
      cache_config=kv_cache_manager_lib.CacheConfig(
          max_device_size_gib=max_device_size_gib,
          page_size=PAGE_SIZE,
          dtype=jnp.float32,
      ),
      scheduler_config=scheduler_lib.SchedulerConfig(
          max_num_batched_tokens=max_num_batched_tokens,
          max_num_seqs=4,
          max_chunked_prefill_length=chunked_prefill_length,
          num_scheduler_steps=num_scheduler_steps,
      ),
      model_runner_config=model_runner_lib.ModelRunnerConfig(
          max_top_k=max_top_k,
          mesh=mesh(),
          return_logprobs=return_logprobs,
          return_logits=return_logits,
          num_scheduler_steps=num_scheduler_steps,
      ),
      max_model_len=max_model_len,
      eos_token_ids=eos_token_ids,
  )


def make_request(
    request_id: str, prompt: list[int], **sampling_kwargs
) -> sampler_lib.SamplingRequest:
  """Returns a greedy request, generating up to 8 tokens by default."""
  sampling_kwargs.setdefault('max_tokens', 8)
  sampling_kwargs.setdefault('temperature', 0.0)
  return sampler_lib.SamplingRequest(
      request_id=request_id,
      prompt=np.asarray(prompt, dtype=np.int32),
      sampling_params=sampler_lib.SamplingParams(**sampling_kwargs),
  )

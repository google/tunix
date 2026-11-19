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
from unittest import mock

from flax import nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.common import configs as common_configs
from tunix.common import datatypes
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils

VOCAB_SIZE = 32
PAGE_SIZE = 4
GEOMETRIES = {
    'layer_0': datatypes.CacheGeometry(num_kv_heads=1, head_dim=1)
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
      geometries: Mapping[str, datatypes.CacheGeometry] = GEOMETRIES,
  ):
    self.config = PagedSumConfig(dtype=dtype)
    self.offset = nnx.Param(jnp.asarray(offset, dtype=dtype))
    self.geometries = dict(geometries)

  def cache_geometries(self) -> dict[str, datatypes.CacheGeometry]:
    """Returns the geometry of each configured KV cache."""
    return dict(self.geometries)

  def __call__(
      self, tokens, positions, cache, metadata, mesh, logits_idxs=None
  ):
    logits, cache = self._forward(tokens, positions, cache, metadata, mesh)
    if logits_idxs is not None:
      logits = jnp.take_along_axis(logits, logits_idxs[..., None], axis=1)
    return logits.reshape(-1, VOCAB_SIZE), cache

  def _forward(self, tokens, positions, cache, metadata, model_mesh):
    """Returns the `[num_ranks, tokens_per_rank, vocab]` logits and cache."""
    # As in paged attention, each device runs its DP rank's slice of rows and
    # tokens against its own pages.
    dp_axes = utils.kv_cache_dp_axes(model_mesh)
    cache_specs = {name: P(dp_axes) for name in cache}
    return shard_map.shard_map(
        self._run,
        mesh=model_mesh,
        in_specs=(
            P(dp_axes),
            P(dp_axes),
            cache_specs,
            {name: P(dp_axes) for name in metadata.page_indices},
            P(dp_axes),
            P(),
        ),
        out_specs=(P(dp_axes), cache_specs),
        check_rep=False,
    )(
        tokens,
        positions,
        cache,
        {
            name: jnp.asarray(table)
            for name, table in metadata.page_indices.items()
        },
        jnp.asarray(metadata.query_lens),
        self.offset.value,
    )

  def _run(self, tokens, positions, cache, page_indices, query_lens, offset):
    """Runs one DP rank's rows, which come first, against its pages."""
    tokens = jnp.squeeze(tokens, axis=0)
    positions = jnp.squeeze(positions, axis=0)
    num_tokens = tokens.shape[0]
    rows = jnp.repeat(
        jnp.arange(len(query_lens)),
        query_lens,
        total_repeat_length=num_tokens,
    )
    is_real = jnp.arange(num_tokens) < jnp.sum(query_lens)

    new_cache = {}
    next_tokens = jnp.zeros((num_tokens,), dtype=jnp.int32)
    for name, pages in cache.items():
      page_table = jnp.asarray(page_indices[name])
      num_pages, page_size = pages.shape[:2]

      # Padding tokens point past the last page, so their writes are dropped.
      page_ids = jnp.where(
          is_real, page_table[rows, positions // page_size], num_pages
      )
      values = tokens.astype(pages.dtype) + offset
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

    logits = 10.0 * jax.nn.one_hot(next_tokens[None], VOCAB_SIZE)
    return logits, new_cache


class UniformTransformer(PagedSumTransformer):
  """A `PagedSumTransformer` that gives every token the same logit.

  At a positive temperature it samples uniformly from the vocabulary, so
  independent samples of the same prompt almost surely differ.
  """

  def __call__(
      self, tokens, positions, cache, metadata, mesh, logits_idxs=None
  ):
    logits, new_cache = super().__call__(
        tokens, positions, cache, metadata, mesh, logits_idxs=logits_idxs
    )
    return jnp.zeros_like(logits), new_cache


class WhitespaceTokenizer:
  """A tokenizer whose text is token ids separated by spaces, e.g. '1 2 3'."""

  def encode(self, text: str) -> list[int]:
    return [int(token) for token in text.split()]

  def decode(self, ids: list[int]) -> str:
    return ' '.join(str(token) for token in ids)

  def bos_id(self) -> int | None:
    return None

  def eos_id(self) -> int:
    return 1

  def pad_id(self) -> int:
    return 0


def make_engine(
    transformer: nnx.Module | None = None,
    *,
    max_model_len: int = 24,
    max_device_size_gib: float = 64 * GIB_PER_PAGE,
    max_host_size_gib: float = 0.0,
    max_num_batched_tokens: int = 32,
    chunked_prefill_length: int = 32,
    num_scheduler_steps: int = 1,
    max_top_k: int = 4,
    return_logprobs: bool = False,
    return_logits: bool = False,
    eos_token_ids: frozenset[int] = frozenset(),
    max_num_seqs: int = 4,
    dp_mesh: jax.sharding.Mesh | None = None,
) -> engine_lib.LLMEngine:
  """Returns an engine running `transformer`, a `PagedSumTransformer` if None.

  Args:
    transformer: The model to run.
    max_model_len: The engine's maximum sequence length.
    max_device_size_gib: The KV cache budget of each device.
    max_host_size_gib: The KV cache budget of the host.
    max_num_batched_tokens: The token budget of each DP rank.
    chunked_prefill_length: The longest prefill chunk.
    num_scheduler_steps: The forward passes of each engine step.
    max_top_k: The largest `top_k` requests may use.
    return_logprobs: Whether requests return logprobs.
    return_logits: Whether requests return logits.
    eos_token_ids: The tokens that finish a request.
    max_num_seqs: The sequence budget of each DP rank.
    dp_mesh: A mesh to split the KV caches into DP ranks over, one per device
      along its axes. None runs one rank on a single device.
  """
  device_mesh = dp_mesh if dp_mesh is not None else mesh()
  rollout_config = common_configs.RolloutConfig(
      top_k=max_top_k,
      eos_tokens=list(eos_token_ids),
      return_logprobs=return_logprobs,
      return_logits=return_logits,
      rollout_vanilla_v2_kv_cache_page_size=PAGE_SIZE,
      rollout_vanilla_v2_host_size_gib=max_host_size_gib,
      rollout_vanilla_v2_max_num_seqs=max_num_seqs,
      rollout_vanilla_v2_max_num_batched_tokens=max_num_batched_tokens,
      rollout_vanilla_v2_chunked_prefill_length=chunked_prefill_length,
      rollout_vanilla_v2_num_scheduler_steps=num_scheduler_steps,
  )
  with mock.patch.object(
      utils, 'resolve_max_device_size_gib', return_value=max_device_size_gib
  ):
    return engine_lib.LLMEngine(
        transformer if transformer is not None else PagedSumTransformer(),
        rollout_config=rollout_config,
        mesh=device_mesh,
        max_model_len=max_model_len,
    )


def make_request(
    request_id: str,
    prompt: list[int],
    *,
    route_key: int | str | None = None,
    **sampling_kwargs,
) -> request_lib.EngineRequest:
  """Returns a greedy request, generating up to 8 tokens by default."""
  sampling_kwargs.setdefault('max_tokens', 8)
  sampling_kwargs.setdefault('temperature', 0.0)
  return request_lib.EngineRequest(
      request_id=request_id,
      prompt_token_ids=list(prompt),
      sampling_params=request_lib.SamplingParams(**sampling_kwargs),
      route_key=route_key,
  )

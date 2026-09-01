"""Core orchestration engine for continuous batching."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import scheduler
from tunix.experimental.generate import tiered_page_pool as tiered_page_lib
from tunix.experimental.generate import utils


def create_kv_page_manager(
    cache_config,
    model_config,
    kv_dtype: jnp.dtype,
    dp_axis: str | None = None,
    tp_axis: str | None = None,
    dp_size: int = 1,
    tp_size: int = 1,
) -> tiered_page_lib.TieredPagePoolManager:
  """Initializes a TieredPageManager for the KV Cache."""

  num_layers = model_config.num_layers
  num_kv_heads = model_config.num_kv_heads
  head_dim = model_config.head_dim
  kv_packing = utils.get_dtype_packing(kv_dtype)

  assert (2 * num_kv_heads) % kv_packing == 0
  packed_kv_dim = 2 * num_kv_heads // kv_packing

  # TODO: Check if a model defines an init cache function.
  # If so, use it to initalize the cache.

  # TODO: Utilizing this function should throw a deprecated warning.
  partition_keys = tuple(f'layer_{i}' for i in range(num_layers))
  page_subshape = (packed_kv_dim, kv_packing, head_dim)
  logical_subsharding = ('tp', None, None)
  logical_page_sharding = 'dp'

  # TODO: We should not have to explicitly construct the logical sharding here.
  logical_tpu_sharding = (logical_page_sharding, None) + logical_subsharding
  num_tpu_pages = utils.calculate_pages_for_capacity(
      cache_config.max_tpu_bytes,
      logical_tpu_sharding,
      cache_config.page_size,
      page_subshape,
      dp_size,
      kv_dtype,
      partition_keys,
  )

  logical_cpu_sharding = (None,) * len(logical_tpu_sharding)
  num_cpu_pages = utils.calculate_pages_for_capacity(
      cache_config.max_cpu_bytes,
      logical_cpu_sharding,
      cache_config.page_size,
      page_subshape,
      dp_size,
      kv_dtype,
      partition_keys,
  )

  config = tiered_page_lib.TieredPagePoolConfig(
      page_size=cache_config.page_size,
      num_tpu_pages=num_tpu_pages,
      num_cpu_pages=num_cpu_pages,
      page_subshape=page_subshape,
      dtype=kv_dtype,
      partition_keys=partition_keys,
      logical_page_sharding=logical_page_sharding,
      logical_subsharding=logical_subsharding,
      dp_axis=dp_axis,
      tp_axis=tp_axis,
      dp_size=dp_size,
      tp_size=tp_size,
  )

  tpu_pool, cpu_pool = config.init()
  return tiered_page_lib.TieredPagePoolManager(
      tiered_config=config,
      tpu_pool=tpu_pool,
      cpu_pool=cpu_pool,
  )


class LLMEngine:
  """Core Continuous Batching Engine orchestration layer."""

  def __init__(
      self,
      transformer: 'nnx.Module',
      tokenizer: Any,
      cache_config: sampler_lib.CacheConfig,
  ):
    self.transformer = transformer
    self.tokenizer = tokenizer
    self.cache_config = cache_config
    self.max_num_batch_tokens = cache_config.max_num_batch_tokens

    self._new_requests = []
    self.max_seq_len = (
        cache_config.max_prompt_length + cache_config.max_tokens_to_generate
    )

    self.sampler = sampler_lib.VanillaSampler(
        transformer=transformer,
        cache_config=cache_config,
    )

    model_config = self.transformer.config  # pytype: disable=attribute-error
    shd_config = getattr(model_config, 'shd_config', None)

    kv_dtype = self.sampler.dtype

    dp_size = 1
    tp_size = 1
    dp_axis = None
    tp_axis = None

    if shd_config is not None:
      dp_axis = shd_config.act_btd[0]
      tp_axis = shd_config.act_btnh[2]

    try:
      _, flat_transformer_state = self.sampler.model_def_and_state()
      param_0 = jax.tree.leaves(flat_transformer_state)[0]
      if (
          hasattr(param_0, 'sharding')
          and hasattr(param_0.sharding, 'mesh')
          and param_0.sharding.mesh is not None
      ):
        mesh = param_0.sharding.mesh
        dp_size = mesh.shape.get(dp_axis, 1) if dp_axis else 1
        tp_size = mesh.shape.get(tp_axis, 1) if tp_axis else 1
    except Exception:
      pass

    self.cache_manager = create_kv_page_manager(
        cache_config=cache_config,
        model_config=model_config,
        kv_dtype=kv_dtype,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )

    eos_ids = [
        tokenizer.eos_id()
        if hasattr(tokenizer, 'eos_id')
        else tokenizer.GetPieceSize()
    ]
    self.scheduler = scheduler.Scheduler(
        kv_cache_manager=self.cache_manager,
        max_num_batch_tokens=self.max_num_batch_tokens,
        max_seqs_per_batch=self.cache_config.max_num_seqs,
        max_tokens_to_generate=self.cache_config.max_tokens_to_generate,
        chunked_prefill_length=self.cache_config.chunked_prefill_size,
        eos_token_ids=eos_ids,
    )

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)

    if hasattr(self.tokenizer, 'bos_id') and callable(self.tokenizer.bos_id):
      bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
      if hasattr(self.tokenizer, 'dedup_bos_ids'):
        input_ids = np.array(
            self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
        )
      else:
        input_ids = np.array(bos_tok + input_ids, dtype=np.int32)
    else:
      input_ids = np.array(input_ids, dtype=np.int32)
    return input_ids

  def add_request(self, req_id: str, prompt: str, **kwargs):
    token_ids = self.tokenize(prompt)
    req = scheduler.Request(req_id, list(token_ids))
    for k, v in kwargs.items():
      setattr(req, k, v)
    self._new_requests.append(req)

    return req

  def has_unfinished_requests(self) -> bool:
    return (
        len(self._new_requests) > 0
        or self.scheduler.num_active_requests > 0
    )

  def step(
      self,
      sampling_config: sampler_lib.SamplingConfig | None = None,
      return_logits: bool = False,
  ):
    """One physical iteration of the continuous batch engine."""
    if sampling_config is None:
      sampling_config = sampler_lib.SamplingConfig()

    ordered_reqs, distribution_list = self.scheduler.schedule_step(
        self._new_requests
    )
    self._new_requests.clear()
    if not ordered_reqs:
      return []

    # --- Form ragged inputs for the attention kernel ---
    max_n_batch_tokens = (
        self.max_num_batch_tokens
    )  # TODO (AGT): Move to and get from sampling config
    max_n_seqs = self.cache_config.max_num_seqs
    max_n_pages_per_seq = utils.cdiv(
        self.max_seq_len, self.cache_config.page_size
    )

    # Ensure that we have enough space in the input buffers.
    # Otherwise, deadlock can occur if a sequence is offloaded
    # with length greater than the max_n_batch_tokens. This
    # constraint can be relaxed once chunked prefill is supported.
    assert max_n_batch_tokens >= self.max_seq_len

    tokens = np.zeros(max_n_batch_tokens, dtype=np.int32)
    query_lens = np.zeros(max_n_seqs, dtype=np.int32)
    kv_lens = np.zeros(max_n_seqs, dtype=np.int32)
    page_indices = np.zeros((max_n_seqs, max_n_pages_per_seq), dtype=np.int32)
    distribution = np.array(distribution_list, dtype=np.int32)

    total_n_batch_tokens = 0
    for i, req in enumerate(ordered_reqs):
      n_completed = req.num_completed_tokens
      n_in_flight = req.num_in_flight_tokens

      in_flight = req.token_ids[n_completed : n_completed + n_in_flight]
      start_idx = total_n_batch_tokens
      end_idx = total_n_batch_tokens + n_in_flight
      tokens[start_idx:end_idx] = in_flight

      kv_lens[i] = n_completed + n_in_flight
      query_lens[i] = n_in_flight

      phys_idxs = [self.cache_manager.get_page_idx(pid) for pid in req.page_ids]
      page_indices[i, : len(phys_idxs)] = phys_idxs

      total_n_batch_tokens += n_in_flight

    metadata = sampler_lib.RPAMetadata(
        page_indices=jnp.array(page_indices),
        query_lens=jnp.array(query_lens),
        kv_lens=jnp.array(kv_lens),
        distribution=jnp.array(distribution),
    )
    tokens = jnp.array(tokens)

    # --- Sample input tokens ---
    gen_tokens, logits, logp, next_cache = self.sampler.sample_step(
        cache=self.cache_manager.tpu_pool.partition_pages,
        tokens=tokens,
        metadata=metadata,
        sampling_config=sampling_config,
    )

    # Since JAX expects no-side effects,
    # we must update pages outside of the sampler
    self.cache_manager.update_tpu_pool(next_cache)

    completed_reqs = self.scheduler.update_from_output(
        gen_tokens, logits, logp,
        eos_token_ids=sampling_config.eos_token_ids if sampling_config else None
    )

    return completed_reqs


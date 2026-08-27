import jax
from typing import List, Any
from flax import nnx

import numpy as np
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from tunix.generate import scheduler
from tunix.generate import tiered_page_manager as tiered_page_lib
from tunix.generate import sampler_v2 as sampler_lib
from tunix.tests import test_common as tc
"""Core orchestration engine for continuous batching."""

def create_kv_page_manager(
    num_tpu_pages: int,
    num_cpu_pages: int,
    cache_config,
    model_config,
    kv_dtype: jnp.dtype,
    dp_axis: str | None = None,
    tp_axis: str | None = None,
    dp_size: int = 1,
    tp_size: int = 1,
) -> tiered_page_lib.TieredPageManager:
    """
    Initializes a TieredPageManager for the KV Cache.
    """

    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim
    kv_packing = utils.get_dtype_packing(kv_dtype)

    assert((2 * num_kv_heads) % kv_packing == 0)
    packed_kv_dim = 2 * num_kv_heads // kv_packing
    
    # TODO: Check if a model defines an init cache function.
    # If so, use it to initalize the cache.
    # TODO: Utilizing this function should throw a deprecated warning.
    partition_keys = tuple(f"layer_{i}" for i in range(num_layers))
    page_subshape = (packed_kv_dim, kv_packing, head_dim)
    logical_subsharding = ('tp_axis', None, None)
    logical_page_sharding = 'dp_axis'
    
    # TODO: This should be cleaner.
    logical_tpu_sharding = (logical_page_sharding, None) + logical_subsharding
    num_tpu_pages = utils.calculate_pages_for_capacity(
      max_tpu_bytes,
      logical_tpu_sharding,
      config.page_size,
      page_subshape,
      dp_size,
      kv_dtype,
      partition_keys
    )
    
    logical_cpu_sharding = (None,) * len(logical_tpu_sharding)
    num_cpu_pages = utils.calculate_pages_for_capacity(
      max_cpu_bytes,
      logical_cpu_sharding,
      config.page_size,
      page_subshape,
      dp_size,
      kv_dtype,
      partition_keys
    )
      
    config = tiered_page_lib.TieredPagePoolConfig(
        page_size=cache_config.page_size,
        page_subshape=page_subshape,
        dtype=kv_dtype,
        partition_keys=partition_keys,
        num_tpu_pages=num_tpu_pages,
        num_cpu_pages=num_cpu_pages,
        logical_page_sharding=logical_page_sharding,
        logical_subsharding=logical_subsharding,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )

    tpu_pool, cpu_pool = config.init()
    return TieredPageManager(
        tiered_config=tiered_memory_config,
        tpu_pool=tpu_pool,
        cpu_pool=cpu_pool,
        max_num_seqs=cache_config.max_num_seqs,
    )

class LLMEngine:
    """Core Continuous Batching Engine orchestration layer."""
    def __init__(
      self, 
      transformer: "nnx.Module", 
      tokenizer: Any, 
      cache_config: Any,
      image_processor: Any | None = None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.max_num_batch_tokens = cache_config.max_num_batch_tokens
        
        self.eos_ids = [tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else tokenizer.GetPieceSize()]
        self.generated_logprobs = {}
        self.generated_logits = {}
        
        # TODO (AGT): Properly wire this (maybe pull it out into a seperate func, and add a sampling config) 
        self.sampler = sampler_lib.VanillaSampler(
            transformer=transformer,
            tokenizer=tokenizer,
            cache_config=cache_config,
            image_processor=image_processor,
            static_token_capacity=self.max_num_batch_tokens, 
            temperature=temperature,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k if top_k is not None else -1,
            return_logprobs=return_logprobs,
            forbidden_token_ids=list(forbidden_tokens) if forbidden_tokens else None,
        )
        
        # Initialize scheduling and physical memory allocators
        model_config = self.transformer.config
        shd_config = getattr(model_config, 'shd_config', None)

        kv_dtype = self.sampler.dtype
        
        # TODO (AGT): Move kv_cache initalization into a seperate function.
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
        
        self.kv_cache_manager = create_kv_cache_manager(
            cache_config=cache_config,
            model_config=model_config,
            kv_dtype=kv_dtype,
            dp_axis=dp_axis,
            tp_axis=tp_axis,
            dp_size=dp_size,
            tp_size=tp_size,
            num_tpu_pages=num_tpu_pages,
            num_cpu_pages=num_cpu_pages,
        )
        
        self.scheduler = scheduler.Scheduler(
            kv_cache_manager=self.cache_manager,
            max_num_batch_tokens=self.max_num_batch_tokens,
        )
        
    def add_request(self, req_id: str, prompt_tokens: List[int], **kwargs):
        req = scheduler.Request(req_id, prompt_tokens)
        for k, v in kwargs.items():
            setattr(req, k, v)
        self.scheduler._queue_new_requests([req])
        self.generated_tokens[req_id] = []
        self.generated_logprobs[req_id] = []
        self.generated_logits[req_id] = []
        return req
        
    def has_unfinished_requests(self) -> bool:
        return len(self.scheduler.pending_requests) > 0 or len(self.scheduler.running_requests) > 0
        
    def step(
        self,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        return_logits: bool = False,
        return_logprobs: bool = False,
    ):
        """One physical iteration of the continuous batch engine."""
        
        ordered_reqs, distribution_list = self.scheduler.schedule_step()
        if not ordered_reqs:
            return
            
        j = distribution_list[1] # num_decode + num full prefill.
        k = distribution_list[2] # j + num partial (chunked) prefill.
        
        # --- Form ragged inputs for the attention kernel --- 
        max_n_batch_tokens = self.max_num_batch_tokens # TODO (AGT): Move to and get from sampling config
        max_n_seqs = self.cache_config.max_num_seqs
        max_seq_len = self.max_seq_len # TODO (AGT): Move to and get from sampling config
        max_n_pages_per_seq = utils.cdiv(max_seq_len, self.cache_config.page_size)

        tokens = np.zeros(max_n_batch_tokens, dtype=np.int32) 
        query_lens = np.zeros(max_n_seqs, dtype=np.int32) 
        kv_lens = np.zeros(max_n_seqs, dtype=np.int32) 
        page_indices = np.zeros((max_n_seqs, max_n_pages_per_seq), dtype=np.int32) # (max_n_seqs, max_n_pages)
        distribution = np.array(distribution_list, dtype=np.int32)

        total_n_batch_tokens = 0
        for i,req in enumerate(ordered_reqs):
          n_completed = req.num_completed_tokens
          n_in_flight = req.num_in_flight_tokens
          
          in_flight = req.token_ids[n_completed : n_completed + n_in_flight]
          start_idx = total_n_batch_tokens 
          end_idx = total_n_batch_tokens + n_in_flight
          tokens[start_idx: end_idx] = in_flight

          kv_lens[i] = n_completed
          query_lens[i] = n_in_flight
                          
          phys_idxs = [self.kv_cache_manager.get_page_idx(pid) for pid in req.page_ids]
          page_indices[i, :len(phys_idxs)] = phys_idxs

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
            cache=self.kv_cache_manager.tpu_pool.partition_pages,
            tokens=tokens,
            metadata=metadata,
            
            
        )
        
        # Since JAX expects no-side effects, we must update pages outside of the sampler
        self.kv_manager.update_tpu_pool(next_cache)
        
        # TODO: handle echo 
        # --- Update requests with generated tokens ---
        # Update decode and full prefill requests 
        for idx in range(j):
          r = ordered_reqs[idx]
          new_token = r.token_ids[-1]
                      
          r.num_completed_tokens += r.num_in_flight_tokens
          r.num_in_flight_tokens = 0
          
          terminated = new_token in self.eos_ids or (eos_tokens and new_token in eos_tokens)
          truncated = (len(self.generated_tokens[r.request_id]) >= r.max_tokens_to_generate
          if terminated or truncated:
              for pid in reversed(r.page_ids):
                  self.scheduler._release_page(pid)
              self.scheduler.running_requests.remove(r)

          self.generated_tokens[r.request_id].append(tok)
          if logp is not None:
              self.generated_logprobs[r.request_id].append(float(logp[idx]))
          if logits is not None:
              self.generated_logits[r.request_id].append(list(logits[idx]))
          r.token_ids.append(tok)

                
        # Update chunked prefill requests
        for idx in range(j, k):
            r = ordered_reqs[idx]
            r.num_completed_tokens += r.num_in_flight_tokens
            r.num_in_flight_tokens = 0

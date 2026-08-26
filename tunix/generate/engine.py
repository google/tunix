import jax
from typing import List, Any
from flax import nnx

import numpy as np
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from tunix.generate import scheduler
from tunix.generate import cache_manager as cache_manager_lib
from tunix.generate import sampler_v2 as sampler_lib
from tunix.tests import test_common as tc


class LLMEngine:
    """Core Continuous Batching Engine orchestration layer."""
    def __init__(
        self, 
        transformer: "nnx.Module", 
        tokenizer: Any, 
        cache_config: Any,
        image_processor: Any | None = None,
        max_seq_len: int = 1000,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.max_seq_len = max_seq_len
        self.max_num_batch_tokens = getattr(cache_config, 'max_num_batch_tokens', 1024)
        self.max_num_batch_tokens = getattr(cache_config, "max_num_batch_tokens", 1024)
        
        self.eos_ids = [tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else tokenizer.GetPieceSize()]
        self.generated_tokens = {} # request_id -> list of ints
        self.generated_logprobs = {}
        self.generated_logits = {}
        
        #  
        self.sampler = sampler_lib.VanillaSampler(
            transformer=transformer,
            tokenizer=tokenizer,
            cache_config=cache_config,
            image_processor=image_processor,
            max_seq_len=max_seq_len,
        )
        
        # Initialize scheduling and physical memory allocators
        model_config = self.transformer.config
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
        
        try:
          import tunix.generate.utils as utils
          num_tpu_pages, num_cpu_pages = utils._calculate_pages_for_capacity(
              cache_config, model_config, kv_dtype, tp_size
          )
        except Exception:
          num_tpu_pages, num_cpu_pages = 100, 100

        self.cache_manager = cache_manager_lib.init_cache_manager(
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
            cache_manager=self.cache_manager,
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
        eos_tokens: tuple[int, ...] | None = None,
        forbidden_tokens: tuple[int, ...] | None = None,
    ):
        """One physical iteration of the continuous batch engine."""
        
        ordered_reqs, distribution_list = self.scheduler.schedule_step([])
        ordered_reqs = list(ordered_reqs)
        if not ordered_reqs:
            return
            
        distribution = np.array(distribution_list, dtype=np.int32)
        j = distribution_list[1]
        k = distribution_list[2]
        
        # Build 1D arrays
        # Build 1D arrays
        tokens = []
        active_seq_lens = []
        seq_lens = []
        page_indices = []
        
        max_pages = max([len(r.page_ids) for r in ordered_reqs] + [1])
        
        for r in ordered_reqs:
            completed = getattr(r, 'num_completed_tokens', 0)
            in_flight = getattr(r, 'num_in_flight_tokens', 0)
            
            # Sub-slice the token_ids block. If it's decoding, we append the generated token.
            if completed >= len(r.token_ids):
                # Decoding: append last generated token
                last_tok = self.generated_tokens[r.request_id][-1] if self.generated_tokens[r.request_id] else r.token_ids[-1]
                toks = [last_tok]
            else:
                # Prefill: append in-flight prompt block
                toks = r.token_ids[completed : completed + in_flight]
                
            tokens.extend(toks)
            active_seq_lens.append(len(toks))
            seq_lens.append(completed + len(toks))
            
            # Map logical page IDs to physical hardware indices
            phys_idxs = [self.cache_manager._page_id_to_idx[pid] for pid in r.page_ids]
            # Pad indices for batch uniformity
            phys_idxs.extend([0] * (max_pages - len(phys_idxs)))
            page_indices.append(phys_idxs)
            
        max_batch_tokens = getattr(self, 'max_num_batch_tokens', 1024)
        max_seqs = getattr(self.cache_config, 'max_num_seqs', 256)
        max_pages = max([len(p) for p in page_indices] + [1])
        
        seq_lens = seq_lens + [0] * (max_seqs - len(seq_lens))
        active_seq_lens = active_seq_lens + [0] * (max_seqs - len(active_seq_lens))
        
        pad_amount = max_batch_tokens - len(tokens)
        if pad_amount > 0:
            tokens.extend([0] * pad_amount)
            
        for _ in range(max_seqs - len(page_indices)):
            page_indices.append([0] * max_pages)
            
        for p in page_indices:
            p.extend([0] * (max_pages - len(p)))
            
        metadata = sampler_lib.RPAMetadata(
            page_indices=jnp.array(page_indices, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            active_seq_lens=jnp.array(active_seq_lens, dtype=jnp.int32),
            distribution=jnp.array(distribution, dtype=jnp.int32),
        )
        tokens = jnp.array(tokens, dtype=jnp.int32)
        total_tokens = max_batch_tokens
        
        gen_tokens, logits, logp, next_cache = self.sampler.sample_step(
            cache=self.cache_manager.tpu_block.partition_pages,
            tokens=tokens,
            metadata=metadata,
            static_token_capacity=total_tokens, 
            temperature=temperature,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k if top_k is not None else -1,
            return_logprobs=return_logprobs,
            forbidden_token_ids=list(forbidden_tokens) if forbidden_tokens else None,
        )
        
        self.cache_manager.tpu_block.partition_pages = next_cache
        
        # We only sampled for indices < j (decodes and prefills_completing)
        for idx in range(j):
            r = ordered_reqs[idx]
            tok = int(gen_tokens[idx])
            self.generated_tokens[r.request_id].append(tok)
            if logp is not None:
                try:
                    self.generated_logprobs[r.request_id].append(float(logp[idx]))
                except Exception as e:
                    print(f"Logprobs Exception: {e}")
            if logits is not None:
                try:
                    self.generated_logits[r.request_id].append(list(logits[idx]))
                except Exception:
                    pass
            r.token_ids.append(tok)
            
            if not hasattr(r, 'num_completed_tokens'):
                r.num_completed_tokens = 0
            if not hasattr(r, 'num_in_flight_tokens'):
                r.num_in_flight_tokens = 0
                
            r.num_completed_tokens += r.num_in_flight_tokens
            # A decode token counts as 1 completed token on the next tick
            # But wait, num_completed_tokens tracks prompt tokens... wait!
            # The scheduler `n_new_pages` uses `r.num_completed_tokens`. It must track total context tokens!
            if completed >= len(r.token_ids):
                r.num_completed_tokens += 1
            
            if tok in self.eos_ids or (eos_tokens and tok in eos_tokens) or (len(r.token_ids) + len(self.generated_tokens[r.request_id])) >= self.max_seq_len:
                for pid in reversed(r.page_ids):
                    self.scheduler._release_page(pid)
                self.scheduler.running_requests.remove(r)
                
        # for chunked prefills, update their completed tokens
        for idx in range(j, k):
            r = ordered_reqs[idx]
            if not hasattr(r, 'num_completed_tokens'):
                 r.num_completed_tokens = 0
            r.num_completed_tokens += getattr(r, 'num_in_flight_tokens', 0)

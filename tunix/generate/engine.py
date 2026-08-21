import jax
from typing import List, Any
from flax import nnx
from tunix.generate import scheduler
from tunix.generate import cache_manager as cache_manager_lib
from tunix.generate import continuous_sampler as sampler_lib
from tunix.generate import batch_page_manager as batch_page_manager_lib
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
        
        self.eos_ids = [tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else tokenizer.GetPieceSize()]
        self.generated_tokens = {} # request_id -> list of ints
        
        # 1. Initialize PHYSICAL Page Managers here instead of the sampler
        if hasattr(transformer, 'config'):
            dtype = transformer.config.dtype
            num_kv_heads = transformer.config.num_kv_heads
            head_dim = transformer.config.head_dim
            num_layers = transformer.config.num_layers
        else:
            dtype = jnp.float32
            num_kv_heads = 1
            head_dim = 1
            num_layers = 1
            
        import jax.numpy as jnp
        block_spec = batch_page_manager_lib.BlockSpec(name="kv_cache", dtype=dtype, subshape=(num_kv_heads, head_dim))
        
        # Assume roughly some logic to get num pages from max_bytes, since we removed PageManagerConfig
        import jax.numpy as jnp
        import numpy as np
        item_size = jnp.dtype(dtype).itemsize
        page_bytes = item_size * self.cache_config.page_size * num_kv_heads * head_dim
        num_hbm_pages = getattr(self.cache_config, "hbm_cache_max_bytes", 1) // page_bytes
        hbm_block = block_spec.init(
            num_pages=num_hbm_pages,
            page_size=self.cache_config.page_size,
            device=None
        )
        max_num_pages_per_seq = (self.max_seq_len + self.cache_config.page_size - 1) // self.cache_config.page_size
        self.hbm_pm = batch_page_manager_lib.BatchPageManager(
            block=hbm_block,
            page_indices=jnp.full((self.cache_config.max_num_seqs, max_num_pages_per_seq), -1, dtype=jnp.int32),
            seq_lens=jnp.zeros((self.cache_config.max_num_seqs,), dtype=jnp.int32),
            max_seq_len=self.max_seq_len,
            window_size=None
        )
        
        if getattr(self.cache_config, "cpu_offload_bytes", 0) > 0:
            num_cpu_pages = self.cache_config.cpu_offload_bytes // page_bytes
            shape = (num_cpu_pages, self.cache_config.page_size) + block_spec.subshape
            self.cpu_block = jax.device_put(jnp.zeros(shape, dtype=dtype), jax.devices("cpu")[0])
        else:
            self.cpu_block = None
        
        # 2. Own and Initialize the Sampler!
        self.sampler = sampler_lib.ContinuousSampler(
            transformer=transformer,
            tokenizer=tokenizer,
            cache_config=cache_config,
            image_processor=image_processor,
            max_seq_len=max_seq_len,
        )
        
        # Initialize scheduling and physical memory allocators
        self.cache_manager = cache_manager_lib.CacheManager(
            hbm_page_manager=self.hbm_pm,
            max_num_seqs=cache_config.max_num_seqs,
            max_num_pages_per_seq=utils.cdiv(self.max_seq_len, cache_config.page_size),
            page_size=cache_config.page_size,
            cpu_block=self.cpu_block
        )
        
        self.scheduler = scheduler.Scheduler(
            cache_manager=self.cache_manager,
            page_size=cache_config.page_size,
            max_num_seqs=cache_config.max_num_seqs,
            max_num_batch_tokens=getattr(cache_config, "max_num_batch_tokens", 1024),
        )
        
    def add_request(self, req_id: str, prompt_tokens: List[int]):
        req = scheduler.Request(req_id, prompt_tokens)
        self.scheduler._queue_new_requests([req])
        self.generated_tokens[req_id] = []
        return req
        
    def has_unfinished_requests(self) -> bool:
        return len(self.scheduler.pending_requests) > 0 or len(self.scheduler.running_requests) > 0
        
    def step(self):
        """One physical iteration of the continuous batch engine."""
        
        decodes, prefills = self.scheduler.schedule_step([])
        all_active = decodes + prefills
        
        if not all_active:
            return
            
        cache = self.cache_manager
        next_tokens_cpu, next_cache = self.sampler.sample(all_active, prefills, cache)
        self.cache_manager = next_cache 
        
        for i, r in enumerate(all_active):
            tok = int(next_tokens_cpu[i])
            self.generated_tokens[r.req_id].append(tok)
            r.is_prefill_done = True 
            
            if tok in self.eos_ids or (len(r.prompt_tokens) + len(self.generated_tokens[r.req_id])) >= self.max_seq_len:
                for pid in reversed(r.page_ids):
                    self.scheduler.release_page(pid)
                self.scheduler.running_requests.remove(r)
            else:
                r.prompt_tokens.append(tok)

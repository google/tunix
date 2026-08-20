import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from tunix.generate import base_sampler
from tunix.generate.scheduler import Scheduler, Request
from tunix.generate.cache_manager import CacheManager
from tunix.generate import page_manager as page_manager_lib
from tunix.generate import utils

class ContinuousSampler(base_sampler.BaseSampler):
    def __init__(
        self,
        transformer: nnx.Module,
        tokenizer: any,
        cache_config: any,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.cache_config = cache_config
        
        self._transformer_graphdef = nnx.graphdef(transformer)
        self._transformer_state = nnx.variables(transformer)
        self._flattened_transformer_state = jax.tree.leaves(
            self._transformer_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
        
        # We will cleanly JIT the individual forward execution functions which will be 
        # orchestrated purely by Python.
        self._compiled_prefill_fn = jax.jit(self._prefill_fn)
        self._compiled_decode_fn = jax.jit(self._decode_fn)

    def _init_components(self, max_seq_len: int, dtype: jnp.dtype):
        """Build the physical PageManagers and inject them into our Python Scheduler."""
        num_kv_heads = self.transformer.config.num_kv_heads
        head_dim = self.transformer.config.head_dim
        num_layers = self.transformer.config.num_layers
        
        # TODO: Configure dp/tp shard bounds natively.
        
        hbm_pm_config = page_manager_lib.PageManagerConfig(
            page_size=self.cache_config.page_size,
            max_seq_len=max_seq_len,
            max_bytes=self.cache_config.hbm_cache_max_bytes,
            num_kv_heads=num_kv_heads,
            max_num_seqs=self.cache_config.max_num_seqs,
            head_dim=head_dim,
            dtype=dtype,
            num_layers=num_layers,
        )
        
        # Initialize bare JAX physical components
        self.hbm_pm = hbm_pm_config.init()
        # TODO: CPU initialization
        self.cpu_pm = None

        self.cache_manager = CacheManager(
            hbm_page_manager=self.hbm_pm,
            offload_page_manager=self.cpu_pm
        )
        
        self.scheduler = Scheduler(
            cache_manager=self.cache_manager,
            page_size=self.cache_config.page_size,
            max_num_seqs=self.cache_config.max_num_seqs
        )

    def _prefill_fn(self, params, tokens, positions, hbm_cache):
        # Implementation mirrors original JAX kernel boundary but relies purely on Scheduler's mapping
        pass

    def _decode_fn(self, params, tokens, positions, hbm_cache):
        # Implementation mirrors original JAX kernel boundary but relies purely on Scheduler's mapping
        pass
        
    def __call__(
        self,
        input_strings,
        max_generation_steps: int,
        # ... other config kwargs ...
    ):
        """Python-driven execution loop replacing the lax.while_loop structure."""
        
        # 1. Init state & limits
        tokens = [self.tokenizer.encode(x) for x in input_strings]
        
        # We setup static maximal limits just for JAX tensor pad bounds
        max_prompt_length = max(len(t) for t in tokens)
        max_prompt_length = utils.next_power_of_2(max_prompt_length)
        max_seq_len = max_prompt_length + max_generation_steps
        
        # Dummy dtype extract
        dtype = jnp.bfloat16 # Extract properly from params 
        
        self._init_components(max_seq_len, dtype)
        
        # 2. Ingest sequences dynamically into scheduler
        requests = []
        for i, prompt_tokens in enumerate(tokens):
            requests.append(Request(str(i), prompt_tokens))
            
        # 3. Native Python continuously batched while loop
        # We need a native mechanism to track 'done' condition across all requests.
        active_requests = len(requests)
        
        # Queue up everything!
        self.scheduler._queue_new_requests(requests)
        
        while active_requests > 0:
            # Tell the scheduler to step logic bounds, which physically manipulates JAX PageManager mappings
            # directly against memory constraints, prefix matched chunks, and LRU eviction bounds.
            decodes, prefills = self.scheduler.schedule_step([])
            
            # Extract physically bounded hbm_pm
            hbm_cache = self.scheduler.cache_manager.hbm_page_manager
            
            # Execute dynamically grouped forward passes!
            if prefills:
                # build ragged inputs, execute self._compiled_prefill_fn
                pass
                
            if decodes:
                # build ragged inputs, execute self._compiled_decode_fn
                pass
                
            # Simulate completion tracking correctly...
            active_requests = 0 # Temp break for dev

        return base_sampler.SamplerOutput(
            None, None, None, None, None, None, None, None, None, None, None 
        )

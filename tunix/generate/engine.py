import jax
from typing import List, Any
from flax import nnx
from tunix.generate import scheduler
from tunix.generate import cache_manager as cache_manager_lib
from tunix.generate import continuous_sampler as sampler_lib
from tunix.generate import cache_manager as batch_cache_manager_lib
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
            transformer=transformer,
            cache_config=cache_config,
            max_seq_len=max_seq_len
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

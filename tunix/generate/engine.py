import jax
from typing import List, Any
from tunix.generate import scheduler
from tunix.generate import cache_manager as cache_manager_lib

class LLMEngine:
    """Core Continuous Batching Engine orchestration layer."""
    def __init__(self, sampler: Any, cache_config: Any):
        self.sampler = sampler
        self.cache_config = cache_config
        self.eos_ids = [sampler.tokenizer.eos_id()]
        self.generated_tokens = {} # request_id -> list of ints
        
        # Initialize scheduling and physical memory allocators
        self.cache_manager = cache_manager_lib.CacheManager(
            hbm_page_manager=sampler.hbm_pm,
            offload_page_manager=sampler.cpu_pm
        )
        
        self.scheduler = scheduler.Scheduler(
            cache_manager=self.cache_manager,
            page_size=cache_config.page_size,
            max_num_seqs=cache_config.max_num_seqs,
        )
        
    def add_request(self, req_id: str, prompt_tokens: List[int]):
        req = scheduler.Request(req_id, prompt_tokens)
        # Note: We must queue list of new requests, then step pulls from pending.
        self.scheduler._queue_new_requests([req])
        self.generated_tokens[req_id] = []
        return req
        
    def has_unfinished_requests(self) -> bool:
        return len(self.scheduler.pending_requests) > 0 or len(self.scheduler.running_requests) > 0
        
    def step(self):
        """One physical iteration of the continuous batch engine."""
        
        # 1. schedule
        decodes, prefills = self.scheduler.schedule_step([])
        all_active = decodes + prefills
        
        if not all_active:
            return
            
        hbm_pm = self.cache_manager.hbm_page_manager
        
        # 2. sample
        next_tokens_cpu, next_hbm_pm = self.sampler.sample(all_active, prefills, hbm_pm)
        
        # Write updated physical blocks back to python state tracker
        self.cache_manager.hbm_page_manager = next_hbm_pm 
        
        # 3. update requests / outputs
        for i, r in enumerate(all_active):
            tok = int(next_tokens_cpu[i])
            self.generated_tokens[r.req_id].append(tok)
            r.is_prefill_done = True 
            
            if tok in self.eos_ids or (len(r.prompt_tokens) + len(self.generated_tokens[r.req_id])) >= self.sampler.max_seq_len:
                # Sequence reached EOS or max seq len!
                for pid in reversed(r.page_ids):
                    self.scheduler.release_page(pid)
                self.scheduler.running_requests.remove(r)
            else:
                # Append the newly generated token back to prompt_tokens so next loop tracks length + evaluates it natively!
                r.prompt_tokens.append(tok)

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
        
        model_config = transformer.config if hasattr(transformer, "config") else getattr(transformer, "model_config", None)
        kv_dtype = getattr(model_config, 'dtype', jax.numpy.float32) if model_config else jax.numpy.float32
        
        self.cache_manager = batch_cache_manager_lib.init_cache_manager(
            cache_config=cache_config,
            model_config=model_config,
            kv_dtype=kv_dtype,
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
        
        running_requests = self.scheduler.schedule_step([])
        if not running_requests:
            return
            
        import numpy as np
        
        # Categorize running requests by execution mode to form [i, j, k] distribution
        # i: purely decoding (1 token)
        # i -> j: prefill completing (multi-token, will sample)
        # j -> k: chunk prefill (multi-token, won't sample yet)
        decodes = []
        prefills_completing = []
        chunked_prefills = []
        
        for r in running_requests:
            total_prompt_tokens = len(r.token_ids)
            # Find if this step will hit the end of the prompt
            if getattr(r, 'num_completed_tokens', 0) >= total_prompt_tokens:
                # Already completed prompt, so it's a decode
                decodes.append(r)
            else:
                completed = getattr(r, 'num_completed_tokens', 0)
                in_flight = getattr(r, 'num_in_flight_tokens', 0)
                if completed + in_flight >= total_prompt_tokens:
                    prefills_completing.append(r)
                else:
                    chunked_prefills.append(r)
                    
        # Re-order requests for the batch
        ordered_reqs = decodes + prefills_completing + chunked_prefills
        
        # Build distribution
        i = len(decodes)
        j = i + len(prefills_completing)
        k = j + len(chunked_prefills)
        distribution = np.array([i, j, k], dtype=np.int32)
        
        # Build 1D arrays
        tokens = []
        active_seq_lens = []
        seq_lens = []
        
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
            
        tokens = np.array(tokens, dtype=np.int32)
        active_seq_lens = np.array(active_seq_lens, dtype=np.int32)
        seq_lens = np.array(seq_lens, dtype=np.int32)
        
        total_tokens = len(tokens)
        
        gen_tokens, logits, logp, next_cache = self.sampler.sample_step(
            cache=self.cache_manager.page_manager,
            seq_lens=seq_lens,
            tokens=tokens,
            active_seq_lens=active_seq_lens,
            distribution=distribution,
            static_token_capacity=total_tokens, 
            temperature=0.0
        )
        
        self.cache_manager.page_manager = next_cache
        
        # We only sampled for indices < j (decodes and prefills_completing)
        for idx in range(j):
            r = ordered_reqs[idx]
            tok = int(gen_tokens[idx])
            self.generated_tokens[r.request_id].append(tok)
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
            
            if tok in self.eos_ids or (len(r.token_ids) + len(self.generated_tokens[r.request_id])) >= self.max_seq_len:
                for pid in reversed(r.page_ids):
                    self.scheduler._release_page(pid)
                self.scheduler.running_requests.remove(r)
                
        # for chunked prefills, update their completed tokens
        for idx in range(j, k):
            r = ordered_reqs[idx]
            if not hasattr(r, 'num_completed_tokens'):
                 r.num_completed_tokens = 0
            r.num_completed_tokens += getattr(r, 'num_in_flight_tokens', 0)


    def generate(self, prompt_tokens: List[List[int]]) -> List[List[int]]:
        """Vanilla synchronous rollout loop."""
        req_ids = [f"req_{i}_{id(self)}" for i in range(len(prompt_tokens))]
        
        for req_id, p_tokens in zip(req_ids, prompt_tokens):
            self.add_request(req_id, p_tokens)
            
        while self.has_unfinished_requests():
            self.step()
            
        return [self.generated_tokens[req_id] for req_id in req_ids]

import dataclasses

@dataclasses.dataclass
class CacheConfig:
  """Serving & execution config (decoupled from ModelConfig)."""
  page_size: int = 16
  max_num_seqs: int = 256
  max_prompt_length: int = 1000
  max_tokens_to_generate: int = 1000
  max_tpu_bytes: int = 5 * 1024 ** 3

from typing import Any, Sequence, Tuple, Optional, Iterable, Dict, List
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tunix.generate import page_manager as page_manager_lib
from tunix.generate import cache_manager as cache_manager_lib
from tunix.generate import utils


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Sample a token using top-p sampling."""
    if temperature == 0.0:
        return sample_best(logits, return_logprobs)

    next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

    _no_topk = top_k is None or top_k <= 0
    if top_p >= 1.0 and _no_topk:
        next_token = jax.random.categorical(key, logits=next_token_logits)
        if not return_logprobs:
            return next_token, None
        logp = jax.nn.log_softmax(next_token_logits, axis=-1)
        logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
        logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
        return next_token, logp_sampled

    k = next_token_logits.shape[-1] if _no_topk else top_k
    logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)

    probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
    cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum_probs - probs_sorted > top_p
    logits_sorted = jnp.where(mask, -jnp.inf, logits_sorted)

    next_token_idx = jax.random.categorical(key, logits=logits_sorted)
    next_token = jnp.take_along_axis(indices, next_token_idx[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)

    if return_logprobs:
        logp = jax.nn.log_softmax(next_token_logits, axis=-1)
        logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
        logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    else:
        logp_sampled = None

    return next_token, logp_sampled


def sample_best(
    logits, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
    next_token = next_token[:, 0]
    if not return_logprobs:
        return next_token, None
    logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[:, None], axis=-1)
    return next_token, logp_sampled


class VanillaSampler:
    """
    Stateless Continuous Batching JAX Sampler.
    It takes physical CacheManager objects and arrays, evaluates the 
    Transformer natively, samples tokens using JAX PRNGs, and returns 
    the sampled tokens + logits natively back to the central Engine 
    Python event loop without interleaving logical scheduling or state here.
    """
    
    def __init__(
        self,
        transformer: "nnx.Module",
        tokenizer: Any,
        cache_config: Any,
        image_processor: Any | None = None,
        max_seq_len: int = 1000,
        seed: int = 0,
    ):
        self._transformer_graphdef = nnx.graphdef(transformer)
        self._transformer_state = nnx.variables(transformer)
        self._flattened_transformer_state = jax.tree.leaves(
            self._transformer_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

        self.tokenizer = tokenizer

        config = transformer.config if hasattr(transformer, "config") else getattr(transformer, "model_config", None)
        self.dtype = getattr(config, 'dtype', jnp.float32) if config else jnp.float32

        self.cache_config = cache_config
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        
        self._compiled_sample_step = jax.jit(
            self._sample_step_fn,
            static_argnames=["batch_size", "static_token_capacity"]
        )

    def model_def_and_state(self) -> tuple[Any, Any]:
        return self._transformer_graphdef, self._flattened_transformer_state

    @property
    def transformer(self) -> nnx.Module:
        return nnx.merge(
            self._transformer_graphdef, self._flattened_transformer_state
        )

    @property
    def transformer_state(self) -> Any:
        return self._transformer_state

    @transformer_state.setter
    def transformer_state(self, state: Any) -> None:
        def check_tree_structure(tree1, tree2):
            if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(tree2):
                raise ValueError('New state must have the same structure as the old state.')

            def check_shape_dtype_sharding(x, y):
                def equivalent_sharding(x, y):
                    if isinstance(x.sharding, jax.sharding.SingleDeviceSharding) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
                        return x.sharding.device_set == y.sharding.device_set
                    if not (isinstance(x.sharding, jax.sharding.NamedSharding) and isinstance(y.sharding, jax.sharding.NamedSharding)):
                        return False
                    if x.sharding.mesh != y.sharding.mesh:
                        return False
                    mesh = x.sharding.mesh
                    diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
                    for spec in diff_spec:
                        if spec and mesh.shape[spec] != 1:
                            return False
                    return True

                return jnp.shape(x) == jnp.shape(y) and x.dtype == y.dtype and equivalent_sharding(x, y)

            if not all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2))):
                raise ValueError('New state must have the same shape, dtype and sharding as the old state.')

        check_tree_structure(self._transformer_state, state)
        self._transformer_state = state
        self._flattened_transformer_state = jax.tree.leaves(
            self._transformer_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

    def sample_step(
        self,
        cache: Any,
        seq_lens: np.ndarray,
        kv_lens: np.ndarray,
        tokens: np.ndarray,
        active_seq_lens: np.ndarray,
        distribution: np.ndarray,
        static_token_capacity: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        forbidden_token_ids: list[int] | None = None,
        return_logprobs: bool = False,
        step: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        
        batch_size = len(seq_lens)
        
        logits, updated_cache = self._compiled_sample_step(
            self._flattened_transformer_state, 
            cache,
            jnp.array(seq_lens, dtype=jnp.int32),
            jnp.array(tokens, dtype=jnp.int32),
            jnp.array(active_seq_lens, dtype=jnp.int32),
            jnp.array(distribution, dtype=jnp.int32),
            batch_size=batch_size,
            static_token_capacity=static_token_capacity
        )
        
        if forbidden_token_ids:
            logits = logits.at[:, forbidden_token_ids].set(-jnp.inf)
            
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), step)
        
        tokens, logp = jax.jit(sample_top_p, static_argnames=["temperature", "top_p", "top_k", "return_logprobs"])(
            logits[:, None, :], 
            key, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k,
            return_logprobs=return_logprobs,
        )
        
        tokens_cpu = jax.device_get(tokens)
        logits_cpu = jax.device_get(logits)
        logp_cpu = jax.device_get(logp) if return_logprobs else None
        
        return tokens_cpu, logits_cpu, logp_cpu, updated_cache

    def _sample_step_fn(
        self,
        params: Any,
        cache: Any,
        seq_lens: jnp.ndarray,
        tokens_ragged: jnp.ndarray,
        active_seq_lens: jnp.ndarray,
        distribution: jnp.ndarray,
        batch_size: int,
        static_token_capacity: int,
    ) -> Tuple[jnp.ndarray, Any]:
        
        transformer = nnx.merge(self._transformer_graphdef, params)
        
        ragged = page_manager_lib.RaggedArray(
            data=tokens_ragged,
            lens=active_seq_lens,
        )
        seq_idxs = ragged.row_idxs
        positions = ragged.intra_offsets

        global_positions = positions + (seq_lens[seq_idxs] - active_seq_lens[seq_idxs])
        tokens = tokens_ragged
        
        logits = transformer(
            tokens,
            global_positions, 
            cache=cache,
            distribution=distribution, 
            seq_lens=seq_lens,
            soft_cap=None, 
        )
        
        last_token_idxs = jnp.cumsum(active_seq_lens) - 1
        valid_idxs = jnp.maximum(0, last_token_idxs)
        last_token_logits = logits[valid_idxs]
        
        # Zero out invalid logits from inactive sequences or chunked sequences.
        # distribution = [i, j, k] mapping:
        # sequences[0:i] are decode.
        # sequences[i:j] are prefill only.
        # sequences[j:k] are chunked prefill.
        # We only want to sample from [0:j] (i.e. decoding seqs and completed prefill seqs).
        # Sequences [j:k] are chunk-prefilling and should not emit tokens yet.
        num_sampleable = distribution[1] 
        seq_indices = jnp.arange(batch_size)
        is_sampleable = seq_indices < num_sampleable
        
        last_token_logits = jnp.where(is_sampleable[:, None], last_token_logits, 0.0)
        
        return last_token_logits, cache




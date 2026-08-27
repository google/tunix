from flax import nnx
import dataclasses
from typing import Any, Sequence, Tuple, Optional, Iterable, Dict, List
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

@struct.dataclass
class RPAMetadata:
    """Encapsulates execution metadata arrays for the Ragged Page Attention kernel."""
    page_indices: np.ndarray | jnp.ndarray
    kv_lens: np.ndarray | jnp.ndarray
    query_lens: np.ndarray | jnp.ndarray
    distribution: np.ndarray | jnp.ndarray

@struct.dataclass
class RaggedArray:
    data: jnp.ndarray
    lens: jnp.ndarray
    
    @property
    def row_idxs(self) -> jnp.ndarray:
        return jnp.repeat(jnp.arange(len(self.lens)), self.lens, total_repeat_length=self.data.shape[0])
        
    @property
    def intra_offsets(self) -> jnp.ndarray:
        num_tokens = self.data.shape[0]
        positions = jnp.arange(num_tokens)
        starts = jnp.zeros_like(self.lens)
        if len(self.lens) > 0:
            starts = starts.at[1:].set(jnp.cumsum(self.lens)[:-1])
        return positions - starts[self.row_idxs]

import dataclasses

@dataclasses.dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    forbidden_token_ids: tuple[int, ...] | None = None
    eos_token_ids: tuple[int, ...] | None = None
    return_logprobs: bool = False


@dataclasses.dataclass
class CacheConfig:
  """Serving & execution config (decoupled from ModelConfig)."""
  page_size: int = 16
  max_num_seqs: int = 256
  max_prompt_length: int = 1000
  max_tokens_to_generate: int = 1000
  max_tpu_bytes: int = 5 * 1024 ** 3
  max_cpu_bytes: int = 0
  max_num_batch_tokens: int = 2048


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
    return next_token, logp_sampled if logp_sampled is None else jnp.squeeze(logp_sampled, axis=-1)


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
        
        default_dtype = jnp.float32
        try:
            leaves = jax.tree.leaves(self._transformer_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
            for leaf in leaves:
                if isinstance(leaf, nnx.Variable):
                    val = leaf.get_value() if hasattr(leaf, 'get_value') else getattr(leaf, 'value', None)
                    if val is not None and hasattr(val, 'dtype'):
                        default_dtype = val.dtype
                        if default_dtype != jnp.float32:
                            break
        except Exception:
            pass

        # We inherently trust the physical transformer types (if assigned) over the abstract config fallback
        self.dtype = default_dtype

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
        def get_all_param_types(tree):
            param_types = set()
            jax.tree_util.tree_map(
                lambda x: param_types.add(type(x)),
                tree,
                is_leaf=lambda x: isinstance(x, nnx.Variable),
            )
            return param_types

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

        from flax.nnx import statelib
        from flax.nnx import filterlib

        param_types = get_all_param_types(state)
        
        if nnx.Param in param_types:
            # Full state replacement.
            check_tree_structure(self._transformer_state, state)
            self._transformer_state = state
        else:
            # LoRA state replacement.
            if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
                raise ValueError(f'Only LoRAParam is supported. Received invalid `param_types`: {param_types}')
                
            original_lora_params = statelib.filter_state(self._transformer_state, nnx.LoRAParam)
            check_tree_structure(original_lora_params, state)
            base_state = statelib.filter_state(self._transformer_state, filterlib.Not(nnx.LoRAParam))
            self._transformer_state = statelib.merge_state(base_state, state)

        self._flattened_transformer_state = jax.tree.leaves(
            self._transformer_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

    def sample_step(
        self,
        cache: Any,
        tokens: np.ndarray,
        metadata: RPAMetadata,
        sampling_config: SamplingConfig,
        static_token_capacity: int = 1000,
        step: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        
        batch_size = len(metadata.kv_lens)
        
        logits, updated_cache = self._compiled_sample_step(
            self._flattened_transformer_state, 
            cache,
            jnp.array(tokens, dtype=jnp.int32),
            metadata,
            batch_size=batch_size,
            static_token_capacity=static_token_capacity
        )
        
        if sampling_config.forbidden_token_ids:
            logits = logits.at[:, sampling_config.forbidden_token_ids].set(-jnp.inf)
            
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), step)
        
        tokens, logp = jax.jit(sample_top_p, static_argnames=["temperature", "top_p", "top_k", "return_logprobs"])(
            logits[:, None, :], 
            key, 
            temperature=sampling_config.temperature, 
            top_p=sampling_config.top_p, 
            top_k=sampling_config.top_k,
            return_logprobs=sampling_config.return_logprobs,
        )
        
        tokens_cpu = jax.device_get(tokens)
        logits_cpu = jax.device_get(logits)
        logp_cpu = jax.device_get(logp) if sampling_config.return_logprobs else None
        
        return tokens_cpu, logits_cpu, logp_cpu, updated_cache

    def _sample_step_fn(
        self,
        params: Any,
        cache: Any,
        tokens_ragged: jnp.ndarray,
        metadata: RPAMetadata,
        batch_size: int = 1,
        static_token_capacity: int = 1000,
    ) -> Tuple[jnp.ndarray, Any]:
        
        transformer = nnx.merge(self._transformer_graphdef, params)
        
        ragged = RaggedArray(
            data=tokens_ragged,
            lens=metadata.query_lens,
        )
        seq_idxs = ragged.row_idxs
        positions = ragged.intra_offsets
        global_positions = positions + (metadata.kv_lens[seq_idxs] - metadata.query_lens[seq_idxs])
        tokens = tokens_ragged
        
        # The transformer natively supports receiving metadata (or None for non-ragged cases).
        logits, updated_cache = transformer(
            tokens,
            global_positions, 
            cache=cache,
            metadata=metadata,
            soft_cap=None, 
        )
        
        last_token_idxs = jnp.cumsum(metadata.query_lens) - 1
        valid_idxs = jnp.maximum(0, last_token_idxs)
        last_token_logits = logits[valid_idxs]
        
        num_sampleable = metadata.distribution[1] 
        seq_indices = jnp.arange(batch_size)
        is_sampleable = seq_indices < num_sampleable
        
        last_token_logits = jnp.where(is_sampleable[:, None], last_token_logits, 0.0)
            
        return last_token_logits, updated_cache




import dataclasses
from typing import Optional, List, Any
import jax

@dataclasses.dataclass
class SamplerOutput:
    """Output of the sampler."""
    text: List[str]
    logits: Optional[List[Any]]
    tokens: List[Any]
    padded_prompt_tokens: Optional[List[List[int]]] = None
    logprobs: Optional[List[Any]] = None

"""Sampler for tunix."""

import dataclasses
import inspect
from flax import nnx
from flax import struct
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import numpy as np

Cache = dict[str, jax.Array]

@struct.dataclass
class RPAMetadata:
  """Encapsulates execution metadata arrays for the Ragged Page Attention kernel."""

  page_indices: np.ndarray | jnp.ndarray
  kv_lens: np.ndarray | jnp.ndarray
  query_lens: np.ndarray | jnp.ndarray
  distribution: np.ndarray | jnp.ndarray


@struct.dataclass
class RaggedArray:
  """Encapsulates a ragged array."""

  data: jnp.ndarray
  lens: jnp.ndarray

  @property
  def row_idxs(self) -> jnp.ndarray:
    return jnp.repeat(
        jnp.arange(len(self.lens)),
        self.lens,
        total_repeat_length=self.data.shape[0],
    )

  @property
  def intra_offsets(self) -> jnp.ndarray:
    num_tokens = self.data.shape[0]
    positions = jnp.arange(num_tokens)
    starts = jnp.zeros_like(self.lens)
    if len(self.lens) > 0:
      starts = starts.at[1:].set(jnp.cumsum(self.lens)[:-1])
    return positions - starts[self.row_idxs]


@struct.dataclass
class SamplingConfig:
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = -1
  forbidden_token_ids: tuple[int, ...] | None = None
  eos_token_ids: tuple[int, ...] | None = None
  return_logprobs: bool = False


@dataclasses.dataclass
class CacheConfig:
  """Serving & execution config."""

  page_size: int = 16
  max_num_seqs: int = 32
  max_prompt_length: int = 128
  max_tokens_to_generate: int = 896
  max_tpu_bytes: int = 1 * 1024**3
  max_cpu_bytes: int = 0
  max_num_batch_tokens: int = 4096
  chunked_prefill_size: int = 1024


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
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
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)  # pytype: disable=bad-argument-type

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
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


class VanillaSampler:
  """A sampler for tunix."""

  def __init__(
      self,
      transformer: nnx.Module,
      cache_config: CacheConfig,
      seed: int = 0,
  ):
    self._transformer_graphdef = nnx.graphdef(transformer)
    self._transformer_state = nnx.variables(transformer)
    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

    self.cache_config = cache_config
    self._seed = seed

    self._step_count = jnp.array(0)
    self._rng = jax.random.PRNGKey(seed)

    self._compiled_sample_step = jax.jit(
        self._sample_step,
        static_argnames=['sampling_config'],
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, statelib.State]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state  # pytype: disable=bad-return

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:
    def get_all_param_types(tree):
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):
        def equivalent_sharding(x, y):
          if isinstance(
              x.sharding, jax.sharding.SingleDeviceSharding
          ) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
            return x.sharding.device_set == y.sharding.device_set
          if not (
              isinstance(x.sharding, jax.sharding.NamedSharding)
              and isinstance(y.sharding, jax.sharding.NamedSharding)
          ):
            return False
          if x.sharding.mesh != y.sharding.mesh:
            return False
          mesh = x.sharding.mesh
          diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
          for spec in diff_spec:
            if spec and mesh.shape[spec] != 1:
              return False
          return True

        return (
            jnp.shape(x) == jnp.shape(y)
            and x.dtype == y.dtype
            and equivalent_sharding(x, y)
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            ' state.'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`:'
            f' {param_types}'
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and hasattr(
        self.transformer.config, 'dtype'
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def _sample_step(
      self,
      params: statelib.State,
      step_count: int,
      cache: Cache,
      tokens: jnp.ndarray,
      metadata: RPAMetadata,
      sampling_config: SamplingConfig,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, Cache, int]:

    logits, updated_cache = self._model_step_fn(
        params,
        cache,
        tokens,
        metadata,
    )

    if sampling_config.forbidden_token_ids:
      logits = logits.at[
          :, :, sampling_config.forbidden_token_ids
      ].set(-jnp.inf)

    key = jax.random.fold_in(jax.random.PRNGKey(self._seed), step_count)
    if sampling_config.temperature == 0.0:
      tokens, logp = sample_best(
          logits,
          return_logprobs=sampling_config.return_logprobs,
      )
    else:
      tokens, logp = sample_top_p(  # pytype: disable=bad-assignment
          logits,
          key,
          temperature=sampling_config.temperature,
          top_p=sampling_config.top_p,
          top_k=sampling_config.top_k,
          return_logprobs=sampling_config.return_logprobs,
      )  # pytype: disable=bad-assignment

    return tokens, logits, logp, updated_cache, step_count + 1  # pytype: disable=bad-return

  def sample_step(
      self,
      cache: Cache,
      tokens: jnp.ndarray,
      metadata: RPAMetadata,
      sampling_config: SamplingConfig,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Cache]:
    """Performs a single sampling step."""

    tokens, logits, logp, updated_cache, self._step_count = (
        self._compiled_sample_step(
            self._flattened_transformer_state,
            self._step_count,
            cache,
            tokens,
            metadata,
            sampling_config,
        )
    )

    tokens_cpu = jax.device_get(tokens)
    logits_cpu = jax.device_get(logits)
    logp_cpu = jax.device_get(logp) if sampling_config.return_logprobs else None

    return tokens_cpu, logits_cpu, logp_cpu, updated_cache

  def _model_step_fn(
      self,
      params: statelib.State,
      cache: Cache,
      packed_tokens: jnp.ndarray,
      metadata: RPAMetadata,
  ) -> tuple[jnp.ndarray, Cache]:
    """Performs a single model step."""

    transformer = nnx.merge(self._transformer_graphdef, params)
    kwargs = {}
    decode_only_last_token = self._supports_decode_only_last_token
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True

    ragged = RaggedArray(
        data=packed_tokens,
        lens=metadata.query_lens,  # pytype: disable=bad-argument-type
    )
    seq_idxs = ragged.row_idxs
    local_positions = ragged.intra_offsets

    global_positions = local_positions + metadata.kv_lens[seq_idxs]

    logits, updated_cache = transformer(
        packed_tokens,
        global_positions,
        cache=cache,
        metadata=metadata,
        soft_cap=None,
        **kwargs,
    )
    last_token_idxs = jnp.cumsum(metadata.query_lens) - 1
    last_token_logits = logits[last_token_idxs]
    last_token_logits = jnp.expand_dims(last_token_logits, axis=1)

    return last_token_logits, updated_cache

@struct.dataclass
class SamplerOutput:
  """Output of the sampler."""

  text: list[str]
  logits: list[np.ndarray] | None
  tokens: list[np.ndarray]
  logprobs: list[np.ndarray] | None = None


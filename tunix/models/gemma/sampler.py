"""Sampler for Gemma transformer."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Optional

import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from tunix.models.gemma import model as gemma_lib
from tunix.sft import utils

# Keep the import below for google internal lint.
import sentencepiece as spm  # isort:skip  # pylint: disable=line-too-long


def _sample_top_p(
    probs: jnp.ndarray, p: float, key: jax.Array, k: Optional[int] = None
) -> jnp.ndarray:
  """Sample a token using top-p sampling."""
  k = probs.shape[-1] if k is None else k
  probs_sorted, indices = jax.lax.top_k(probs, k=k)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

  next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)
  return next_token


def sample_top_p(logits, key, temperature: float, top_p: float, top_k: int):
  probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
  next_token = _sample_top_p(probs, top_p, key, top_k)
  return next_token


def sample_best(logits):
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  return next_token


def _compute_attention_masks(
    time_step: jax.Array, seq_len: int, input_mask: jax.Array
) -> jax.Array:
  """Computes causal attention mask."""
  batch_size = input_mask.shape[0]
  batch_time_step = jnp.full((batch_size, 1), time_step, dtype=jnp.uint32)
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
  )
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (batch_size, max_seq_len),
  )
  input_mask = (
      jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  )

  causal_padding = jnp.logical_or(causal_padding, input_mask)
  attention_mask = causal_padding[:, jnp.newaxis, :].astype(jnp.bool_)

  return ~attention_mask


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, gemma_lib.LayerCache]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: Sequence[int] | None

  # Random seed for sampling.
  seed: jax.Array

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Top-p sampling threshold.
  top_p: float = flax.struct.field(pytree_node=False)

  # Top-k sampling threshold.
  top_k: int | None = flax.struct.field(default=None, pytree_node=False)


@dataclasses.dataclass
class SamplerOutput:
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: Optional[list[jax.Array]]

  # Tokens corresponding to the generated samples.
  tokens: list[jax.Array]

  # Left padded prompt tokens.
  padded_prompt_tokens: jax.Array

  # Per-step log probs for output tokens
  logprobs: Optional[list[float]]


class Sampler:
  """Sampler for gemma transformer."""

  def __init__(
      self,
      transformer: gemma_lib.Gemma,
      vocab: spm.SentencePieceProcessor,
      cache_size: int = 1024,
  ):
    """Initializes a sampler for a Gemma model.

    Args:
      transformer: an instance of the Gemma transformer.
      vocab: vocabulary of the given model.
      cache_size: size of the cache for the transformer.
    """
    self.vocab = vocab
    self.cache_size = cache_size
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    # we separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time
    self._compiled_decode_fn = jax.jit(self._decode_fn)
    self._compiled_prefill_fn = jax.jit(self._prefill_fn)

  @property
  def transformer(self) -> gemma_lib.Gemma:
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
        return (
            jnp.shape(x) == jnp.shape(y)
            and jnp.dtype(x) == jnp.dtype(y)
            and x.sharding == y.sharding
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`: '
            f'{param_types}'
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
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: Sequence[int] | None,
      temperature: float,
      top_p: float,
      top_k: int,
      seed: jax.Array,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size = all_input_ids.shape[0]
    num_input_tokens = all_input_ids.shape[1]
    buffer_size = total_sampling_steps + 1

    token_buffer = jnp.full(
        (
            batch_size,
            buffer_size,
        ),
        self.vocab.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    input_mask = input_mask.at[:, :num_input_tokens].set(
        all_input_ids != self.vocab.pad_id()
    )
    positions = utils.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, buffer_size, self.transformer.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=self.transformer.init_cache(
            cache_size=self.cache_size,
            batch_size=batch_size,
            dtype=self.dtype,
        ),
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )

  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = jnp.array([self.vocab.bos_id()] + input_ids, dtype=jnp.int32)
    return input_ids

  def mask_tokens_after_eos_ids(self, token_buffer):
    """Mask token IDs after the EOS token with the padding ID."""
    eos_id = self.vocab.eos_id()
    eos_exists = jnp.any(jnp.equal(token_buffer, eos_id), axis=-1)
    eos_indices = jnp.where(
        eos_exists,
        jnp.argmax(jnp.equal(token_buffer, eos_id), axis=-1),
        token_buffer.shape[-1],
    )
    mask = jnp.less_equal(
        jnp.arange(token_buffer.shape[-1]), eos_indices[:, None]
    )
    masked_token_buffer = token_buffer * mask + self.vocab.pad_id() * (1 - mask)

    return masked_token_buffer

  def _prefill_fn(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs prefill."""
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = sampler_state.decoding_step

    tokens = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )
    step_positions = jax.lax.dynamic_slice(
        sampler_state.positions,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    input_mask = tokens != self.vocab.pad_id()
    attention_mask = make_causal_attn_mask(input_mask, self.cache_size)

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )

    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.temperature > 0:
      key = jax.random.fold_in(sampler_state.seed, decoding_step)
      next_token_candidate = sample_top_p(
          logits,
          key,
          sampler_state.temperature,
          sampler_state.top_p,
          sampler_state.top_k,
      )
    else:
      next_token_candidate = sample_best(logits)

    token_buffer = sampler_state.token_buffer.at[:, decoding_step + 1].set(
        next_token_candidate
    )

    if sampler_state.logits_buffer is not None:
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, 1, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    done = sampler_state.done | jnp.equal(
        token_buffer[:, decoding_step + 1], self.vocab.eos_id()
    )

    return _SamplingState(
        decoding_step=decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        top_p=sampler_state.top_p,
        top_k=sampler_state.top_k,
        seed=sampler_state.seed,
    )

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = jnp.asarray(sampler_state.decoding_step, dtype=jnp.int32)

    last_token = sampler_state.token_buffer[:, decoding_step]
    last_token = last_token.reshape((batch_size, 1))
    step_positions = jnp.expand_dims(
        sampler_state.positions[:, decoding_step], -1
    )

    input_mask = sampler_state.token_buffer == self.vocab.pad_id()
    attention_mask = _compute_attention_masks(
        decoding_step, self.cache_size, input_mask
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        last_token,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.temperature > 0:
      key = jax.random.fold_in(sampler_state.seed, decoding_step)
      next_token_candidate = sample_top_p(
          logits,
          key,
          sampler_state.temperature,
          sampler_state.top_p,
          sampler_state.top_k,
      )
    else:
      next_token_candidate = sample_best(logits)

    next_token_candidate = jnp.where(
        decoding_step < sampler_state.num_input_tokens - 1,
        sampler_state.token_buffer[:, decoding_step + 1],
        next_token_candidate,
    )

    token_buffer = sampler_state.token_buffer.at[:, decoding_step + 1].set(
        next_token_candidate
    )

    if sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = sampler_state.logits_buffer.at[:, decoding_step + 1].set(
          next_logits
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    done = sampler_state.done | jnp.equal(
        token_buffer[:, decoding_step + 1], self.vocab.eos_id()
    )

    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        top_p=sampler_state.top_p,
        top_k=sampler_state.top_k,
        seed=sampler_state.seed,
    )

  def __call__(
      self,
      input_strings: Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      forbidden_tokens: Sequence[str] | None = None,
      temperature: float = 0.0,
      top_p: float = 0.95,
      top_k: int | None = None,
      seed: int | None = None,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      max_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whgether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      seed: random seed for sampling.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    forbidden_token_ids = None
    if forbidden_tokens is not None:
      forbidden_token_ids = []
      for token in forbidden_tokens:
        token_id = self.vocab.EncodeAsIds(token)
        if len(token_id) != 1:
          raise ValueError(
              'Forbidden tokens must map to single token ids in the vocab.'
          )
        forbidden_token_ids.extend(token_id)
      forbidden_token_ids = tuple(forbidden_token_ids)

    tokens = [self.tokenize(x) for x in input_strings]
    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = max_tokens_length
    all_input_ids = jnp.array([
        pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.vocab.pad_id(),
            left=True,
        )
        for x in tokens
    ])
    total_sampling_steps = max_prompt_length + max_generation_steps

    if seed is None:
      seed = jax.random.PRNGKey(0)
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)
    sampling_state = self.init_sample_state(
        all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state, sampling_state
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )

    masked_token_buffer = self.mask_tokens_after_eos_ids(
        sampling_state.token_buffer
    )

    out_tokens = []
    out_logits = []
    for i, token_buffer in enumerate(masked_token_buffer):
      start_idx = (
          find_first_non_pad_idx(token_buffer, self.vocab.pad_id())
          if echo
          else max_prompt_length
      )
      out_tokens.append(token_buffer[start_idx:total_sampling_steps])
      if return_logits:
        logits_buffer = sampling_state.logits_buffer[i]
        out_logits.append(logits_buffer[start_idx:total_sampling_steps])

    decoded_outputs = [
        self.vocab.DecodeIds(tokens.tolist()) for tokens in out_tokens
    ]

    result = SamplerOutput(
        text=decoded_outputs,
        logits=out_logits,
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
    return result


def pad_to_length(
    x: jax.Array,
    target_length: int,
    pad_value: int = 0,
    left=False,
    axis: int = 0,
) -> jax.Array:
  """Pads a JAX array to a specified target length along a given axis.

  Args:
      x: The JAX array to pad.
      target_length: The desired length of the padded array.
      pad_value: The value to use for padding (default: 0).
      left: If True, add padding tokens to the left of the array.
      axis: The axis along which to pad (default: 0).

  Returns:
      A new JAX array that is padded to the target length along the specified
      axis. Return original array if it is already longer than the target
      length.
  """
  length = x.shape[axis]
  if length >= target_length:
    return x

  padding_shape = list(x.shape)
  padding_shape[axis] = target_length - length
  padding = jnp.full(padding_shape, pad_value, dtype=x.dtype)

  if left:
    return jnp.concatenate([padding, x], axis=axis)
  else:
    return jnp.concatenate([x, padding], axis=axis)


def find_first_non_pad_idx(ids, pad_id):
  """Finds the index of the first non-pad token."""
  mask = ids != pad_id
  if jnp.any(mask):
    return jnp.argmax(mask)
  else:
    return 0


def make_causal_attn_mask(input_mask: jax.Array, cache_size: int) -> jax.Array:
  """Create causal attention mask for prefill.

  The causal attention mask during prefill phase is having shape
  (B, T, CACHE_SIZE).

  Args:
    input_mask: Mask for the input
    cache_size: KV cache size

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attn_mask *= causal_mask[None, ...]
  padding = cache_size - seq_len
  assert padding >= 0
  attn_mask = jnp.pad(
      attn_mask, (*((0, 0) for _ in range(attn_mask.ndim - 1)), (0, padding))
  )
  return attn_mask

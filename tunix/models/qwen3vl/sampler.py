# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL sampler with image support.

Supports greedy and top-p sampling for both text-only and vision-language
inputs using the Qwen3-VL JAX model.

Usage::

    # Text-only (greedy):
    python -m tunix.models.qwen3vl.sampler \\
        --model_id_or_dir Qwen/Qwen3-VL-4B-Instruct \\
        --prompt "<|im_start|>user\\nWhat is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n"

    # With an image (top-p):
    python -m tunix.models.qwen3vl.sampler \\
        --model_id_or_dir Qwen/Qwen3-VL-4B-Instruct \\
        --prompt "<|im_start|>user\\nWhat is in the image?<|im_end|>\\n<|im_start|>assistant\\n" \\
        --image_url https://fastly.picsum.photos/id/100/300/200.jpg \\
        --top_p 0.9 --temperature 0.7
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import io
import os
from typing import Optional

import flax
from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor
from tunix.models.qwen3vl import model as model_lib
from tunix.models.qwen3vl import params as params_lib
from tunix.models.qwen3vl.consistency_test import resolve_model_dir
from tunix.models.qwen3vl.model import get_rope_index
from tunix.models.qwen3vl.vision import compute_grid_data
from tunix.models.qwen3vl.vision import VisionGridData

# Special token IDs (constant for all Qwen3-VL checkpoints).
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652

# ---------------------------------------------------------------------------
# Sampling state
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class _SamplingState:
  """Internal state carried through the decode loop."""

  # Current decode step (counts up from 0 after prefill).
  decoding_step: jnp.int32

  # Token buffer: filled left-to-right, prompt first then generated tokens.
  token_buffer: jnp.ndarray  # [B, total_steps]

  # 1-D position of the *next* token to generate (all three M-RoPE axes
  # receive the same value for a pure-text token).
  next_position: jnp.ndarray  # [B]

  # Per-layer KV cache.
  cache: dict

  # Whether each sequence has finished (hit EOS or step limit).
  done: jnp.ndarray  # [B]

  # Accumulated logits (None when not requested).
  logits_buffer: jnp.ndarray | None  # [B, total_steps, V]

  # Random key threaded through top-p sampling.
  seed: jax.Array

  # --- static (not traced by JAX) ---
  total_sampling_steps: int = flax.struct.field(pytree_node=False)
  num_input_tokens: int = flax.struct.field(pytree_node=False)
  sampling_mode: str = flax.struct.field(pytree_node=False)
  temperature: float = flax.struct.field(pytree_node=False)
  top_p: float = flax.struct.field(pytree_node=False)
  top_k: int | None = flax.struct.field(pytree_node=False)
  forbidden_token_ids: tuple[int, ...] | None = flax.struct.field(
      pytree_node=False
  )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class Qwen3VLSampler:
  """Autoregressive sampler for Qwen3-VL with optional image support.

  Differences from the generic Tunix ``Sampler``:

  * Model call takes ``(input_tokens, positions_3d, pixel_values,
    vision_precomputed, cache, attention_mask)`` — the 3D M-RoPE position
    tensor is computed here using ``get_rope_index``.
  * Vision tokens are injected during prefill; decode steps are text-only.
  """

  def __init__(
      self,
      model: model_lib.Qwen3VL,
      processor: AutoProcessor,
      cache_size: int,
  ):
    self._model_graphdef: graph.NodeDef = nnx.graphdef(model)
    self._model_state: statelib.State = nnx.variables(model)
    self._flattened_model_state = jax.tree.leaves(
        self._model_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )
    self._cache_size = cache_size
    self._config: model_lib.ModelConfig = model.config
    self._processor = processor
    self._tokenizer = processor.tokenizer

    self._compiled_decode_fn = jax.jit(self._decode_fn)

  @property
  def _model(self) -> model_lib.Qwen3VL:
    return nnx.merge(self._model_graphdef, self._flattened_model_state)

  @property
  def _dtype(self) -> jnp.dtype:
    return self._flattened_model_state[0].dtype

  # ------------------------------------------------------------------
  # Input preparation
  # ------------------------------------------------------------------

  def _prepare_text_inputs(
      self,
      prompts: Sequence[str],
      pad_to_multiple: int = 1,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise prompts and return left-padded input_ids and attention_mask."""
    tok = self._tokenizer
    pad_id = (
        tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    )

    encoded = [tok.encode(p, add_special_tokens=True) for p in prompts]
    max_len = max(len(e) for e in encoded)
    if pad_to_multiple > 1:
      max_len = (
          (max_len + pad_to_multiple - 1) // pad_to_multiple
      ) * pad_to_multiple

    input_ids = np.full((len(prompts), max_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((len(prompts), max_len), dtype=np.int32)
    for i, enc in enumerate(encoded):
      start = max_len - len(enc)
      input_ids[i, start:] = enc
      attention_mask[i, start:] = 1

    return input_ids, attention_mask

  # ------------------------------------------------------------------
  # Prefill
  # ------------------------------------------------------------------

  def _prefill(
      self,
      input_ids: np.ndarray,  # [B, L]
      attention_mask: np.ndarray,  # [B, L]
      positions_3d: jax.Array,  # [3, B, L]
      total_sampling_steps: int,
      vision_grid: VisionGridData | None,
      pixel_values: np.ndarray | None,
      sampling_mode: str,
      temperature: float,
      top_p: float,
      top_k: int | None,
      forbidden_token_ids: tuple[int, ...] | None,
      seed: jax.Array,
      include_logits: bool,
  ) -> _SamplingState:
    """Run the prefill pass and initialise the sampling state."""
    batch_size, seq_len = input_ids.shape

    input_ids_jax = jnp.array(input_ids)
    attn_mask_jax = jnp.array(attention_mask)

    pixel_values_jax = None
    if pixel_values is not None and self._config.vision_config is not None:
      pixel_values_jax = jnp.array(pixel_values).astype(self._dtype)

    pad_id = (
        self._tokenizer.pad_token_id
        if self._tokenizer.pad_token_id is not None
        else self._tokenizer.eos_token_id
    )
    token_buffer = np.full(
        (batch_size, total_sampling_steps), pad_id, dtype=np.int32
    )
    token_buffer[:, :seq_len] = input_ids
    token_buffer_jax = jnp.array(token_buffer)

    cache = self._model.init_cache(batch_size, self._cache_size, self._dtype)

    model = nnx.merge(self._model_graphdef, self._flattened_model_state)
    logits, cache = model(
        input_ids_jax,
        positions_3d,
        pixel_values_jax,
        vision_grid,
        cache,
        attn_mask_jax.astype(jnp.bool_),
    )

    first_token, seed = self._sample_token(
        logits[:, -1:, :],
        sampling_mode,
        temperature,
        top_p,
        top_k,
        forbidden_token_ids,
        seed,
        step=0,
    )
    token_buffer_jax = token_buffer_jax.at[:, seq_len].set(first_token)

    next_position = jnp.max(positions_3d[0], axis=-1) + 1  # [B]

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, total_sampling_steps, self._config.vocab_size),
          dtype=jnp.float32,
      )
      logits_buffer = logits_buffer.at[:, seq_len - 1].set(
          logits[:, -1, :].astype(jnp.float32)
      )
    else:
      logits_buffer = None

    eos_id = self._tokenizer.eos_token_id
    done = jnp.isin(first_token, jnp.array([eos_id]))

    return _SamplingState(
        decoding_step=jnp.int32(seq_len),
        token_buffer=token_buffer_jax,
        next_position=next_position,
        cache=cache,
        done=done,
        logits_buffer=logits_buffer,
        seed=seed,
        total_sampling_steps=total_sampling_steps,
        num_input_tokens=seq_len,
        sampling_mode=sampling_mode,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        forbidden_token_ids=forbidden_token_ids,
    )

  # ------------------------------------------------------------------
  # Decode loop
  # ------------------------------------------------------------------

  def _decode_fn(
      self,
      params: statelib.State,
      state: _SamplingState,
      eos_ids: jax.Array,
  ) -> _SamplingState:
    """JIT-compiled decode loop (``jax.lax.while_loop``)."""

    def cond(s: _SamplingState) -> jnp.ndarray:
      return (s.decoding_step < s.total_sampling_steps - 1) & jnp.any(~s.done)

    def step(s: _SamplingState) -> _SamplingState:
      return self._decode_step(params, s, eos_ids)

    return jax.lax.while_loop(cond, step, state)

  def _decode_step(
      self,
      params: statelib.State,
      state: _SamplingState,
      eos_ids: jax.Array,
  ) -> _SamplingState:
    """Single autoregressive decode step."""
    batch_size = state.token_buffer.shape[0]
    step = state.decoding_step

    last_token = state.token_buffer[:, step].reshape(batch_size, 1)  # [B, 1]
    pos = state.next_position[:, None]  # [B, 1]
    positions_3d = jnp.stack([pos, pos, pos], axis=0)  # [3, B, 1]

    model = nnx.merge(self._model_graphdef, params)
    logits, new_cache = model(
        last_token,
        positions_3d,
        None,  # pixel_values
        None,  # vision_precomputed
        state.cache,
        None,  # attention_mask (single real token, no padding)
    )

    next_token, new_seed = self._sample_token(
        logits,
        state.sampling_mode,
        state.temperature,
        state.top_p,
        state.top_k,
        state.forbidden_token_ids,
        state.seed,
        step=step,
    )

    new_token_buffer = state.token_buffer.at[:, step + 1].set(next_token)
    new_done = state.done | jnp.isin(next_token, eos_ids)

    new_logits_buffer = state.logits_buffer
    if state.logits_buffer is not None:
      new_logits_buffer = state.logits_buffer.at[:, step].set(
          logits[:, 0, :].astype(jnp.float32)
      )

    return _SamplingState(
        decoding_step=step + 1,
        token_buffer=new_token_buffer,
        next_position=state.next_position + 1,
        cache=new_cache,
        done=new_done,
        logits_buffer=new_logits_buffer,
        seed=new_seed,
        total_sampling_steps=state.total_sampling_steps,
        num_input_tokens=state.num_input_tokens,
        sampling_mode=state.sampling_mode,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        forbidden_token_ids=state.forbidden_token_ids,
    )

  # ------------------------------------------------------------------
  # Token sampling helpers
  # ------------------------------------------------------------------

  def _sample_token(
      self,
      logits: jnp.ndarray,  # [B, 1, V]
      mode: str,
      temperature: float,
      top_p: float,
      top_k: int | None,
      forbidden_token_ids: tuple[int, ...] | None,
      seed: jax.Array,
      step: int | jnp.ndarray,
  ) -> tuple[jnp.ndarray, jax.Array]:
    """Returns ``(next_token [B], new_seed)``."""
    logits_1d = logits[:, -1, :]  # [B, V]
    if forbidden_token_ids:
      logits_1d = logits_1d.at[:, list(forbidden_token_ids)].set(-jnp.inf)

    if mode == 'greedy':
      return jnp.argmax(logits_1d, axis=-1), seed

    key = jax.random.fold_in(seed, step)
    new_seed = jax.random.fold_in(seed, step + 1)
    probs = jax.nn.softmax(logits_1d.astype(jnp.float32) / temperature, axis=-1)
    k = probs.shape[-1] if top_k is None else top_k
    probs_sorted, indices = jax.lax.top_k(probs, k=k)
    cumsum = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum - probs_sorted > top_p
    probs_sorted = jnp.where(mask, 0.0, probs_sorted)
    probs_sorted = probs_sorted / jnp.sum(probs_sorted, axis=-1, keepdims=True)
    sampled = jax.random.categorical(key, jnp.log(probs_sorted + 1e-10))
    next_token = jnp.take_along_axis(indices, sampled[:, None], axis=-1)[:, 0]
    return next_token, new_seed

  # ------------------------------------------------------------------
  # Public interface
  # ------------------------------------------------------------------

  def __call__(
      self,
      prompts: str | Sequence[str],
      max_new_tokens: int = 100,
      images=None,  # PIL Image, list of Images, or None
      temperature: float = 1.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      eos_tokens: Sequence[int] | None = None,
      forbidden_tokens: Sequence[int] | None = None,
      seed: int = 0,
      echo: bool = False,
  ) -> list[str]:
    """Generate completions.

    Args:
      prompts: A single prompt string or a list of prompt strings.
      max_new_tokens: Maximum number of tokens to generate per prompt.
      images: Optional PIL Image(s) for vision-language prompts.
      temperature: Sampling temperature (used when top_p is set).
      top_p: Nucleus sampling threshold.  If None, greedy decoding is used.
      top_k: Limits the top-k candidates for top-p sampling.
      eos_tokens: Token IDs that signal end-of-sequence.  Defaults to the
        tokenizer's ``eos_token_id``.
      forbidden_tokens: Token IDs that may never be sampled.
      seed: Integer seed for reproducible top-p sampling.
      echo: If True, include the prompt tokens in the returned strings.

    Returns:
      A list of generated strings, one per prompt.
    """
    if isinstance(prompts, str):
      prompts = [prompts]

    sampling_mode = 'top_p' if top_p is not None else 'greedy'
    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None
    rng = jax.random.PRNGKey(seed)

    eos_id = self._tokenizer.eos_token_id
    eos_ids = jnp.array(eos_tokens if eos_tokens else [eos_id])

    # --- Prepare inputs ---
    if images is not None:
      if not isinstance(images, (list, tuple)):
        images = [images] * len(prompts)
      messages_batch = [
          [{
              'role': 'user',
              'content': [
                  {'type': 'image', 'image': ''},
                  {'type': 'text', 'text': p},
              ],
          }]
          for p in prompts
      ]
      texts = [
          self._processor.apply_chat_template(
              m, tokenize=False, add_generation_prompt=True
          )
          for m in messages_batch
      ]
      # Do NOT truncate: truncation strips image tokens from the text while
      # the image processor still generates the full patch grid, causing a
      # token-count mismatch that the processor now validates (5.3.0+).
      # If the encoded length exceeds cache_size we raise a clear error below.
      inputs = self._processor(
          text=texts,
          images=images,
          padding=True,
          return_tensors=None,
      )
      input_ids = np.array(inputs['input_ids'], dtype=np.int32)  # [B, L]
      seq_len_check = input_ids.shape[1]
      if seq_len_check + max_new_tokens > self._cache_size:
        raise ValueError(
            f'Encoded sequence length ({seq_len_check}) + max_new_tokens'
            f' ({max_new_tokens}) = {seq_len_check + max_new_tokens} exceeds'
            f' cache_size {self._cache_size}. Pass a larger cache_size to'
            ' Qwen3VLSampler or reduce the image resolution.'
        )
      attention_mask = np.array(
          inputs['attention_mask'], dtype=np.int32
      )  # [B, L]
      pixel_values = np.array(inputs['pixel_values'], dtype=np.float32)
      image_grid_thw = np.array(
          inputs['image_grid_thw'], dtype=np.int32
      )  # [N, 3]

      vcfg = self._config.vision_config
      vision_grid = compute_grid_data(image_grid_thw, vcfg)
      positions_3d, _ = get_rope_index(
          input_ids=jnp.array(input_ids),
          image_grid_thw=jnp.array(image_grid_thw),
          video_grid_thw=None,
          attention_mask=jnp.array(attention_mask),
          spatial_merge_size=vcfg.spatial_merge_size,
          image_token_id=vcfg.image_pad_id,
          video_token_id=_VIDEO_TOKEN_ID,
          vision_start_token_id=_VISION_START_TOKEN_ID,
      )  # [3, B, L]
    else:
      input_ids, attention_mask = self._prepare_text_inputs(prompts)
      pixel_values = None
      vision_grid = None
      attn = jnp.array(attention_mask)
      positions_1d = jnp.cumsum(attn, axis=-1) - 1
      positions_1d = jnp.where(attn, positions_1d, 0)
      positions_3d = jnp.stack([positions_1d] * 3, axis=0)  # [3, B, L]

    seq_len = input_ids.shape[1]
    total_sampling_steps = seq_len + max_new_tokens
    if total_sampling_steps > self._cache_size:
      raise ValueError(
          f'seq_len ({seq_len}) + max_new_tokens ({max_new_tokens}) = '
          f'{total_sampling_steps} exceeds cache_size {self._cache_size}.'
      )

    # --- Prefill ---
    state = self._prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        positions_3d=positions_3d,
        total_sampling_steps=total_sampling_steps,
        vision_grid=vision_grid,
        pixel_values=pixel_values,
        sampling_mode=sampling_mode,
        temperature=temperature,
        top_p=top_p if top_p is not None else 1.0,
        top_k=top_k,
        forbidden_token_ids=forbidden_token_ids,
        seed=rng,
        include_logits=False,
    )

    # --- Decode ---
    state = self._compiled_decode_fn(
        self._flattened_model_state, state, eos_ids
    )

    # --- Decode tokens to strings ---
    pad_id = (
        self._tokenizer.pad_token_id
        if self._tokenizer.pad_token_id is not None
        else eos_id
    )
    outputs = []
    for i, token_buffer in enumerate(np.array(state.token_buffer)):
      start = 0 if echo else seq_len
      gen_tokens = token_buffer[seq_len:]
      end = seq_len
      for j, tok in enumerate(gen_tokens):
        if tok in np.array(eos_ids).tolist():
          end = seq_len + j
          break
      else:
        end = seq_len + len(gen_tokens)

      out_tokens = token_buffer[start:end]
      out_tokens = out_tokens[out_tokens != pad_id]
      outputs.append(
          self._tokenizer.decode(out_tokens.tolist(), skip_special_tokens=True)
      )

    return outputs


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_sampler(
    model_id_or_dir: str,
    cache_size: int = 512,
    dtype: str = 'bfloat16',
) -> Qwen3VLSampler:
  """Load a Qwen3-VL model and return a ready-to-use ``Qwen3VLSampler``.

  Args:
    model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
    cache_size: KV-cache capacity in tokens.
    dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.

  Returns:
    A ``Qwen3VLSampler`` instance.
  """
  jax_dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
  model_dir = resolve_model_dir(model_id_or_dir)
  config = model_lib.ModelConfig.qwen3vl_4b()

  with jax.default_device(jax.devices()[0]):
    model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=None, dtype=jax_dtype
    )

  processor = AutoProcessor.from_pretrained(model_dir)
  return Qwen3VLSampler(model, processor, cache_size=cache_size)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    model_id_or_dir: str = 'Qwen/Qwen3-VL-4B-Instruct',
    prompt: str = '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n',
    image_url: str | None = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float | None = None,
    cache_size: int = 512,
    dtype: str = 'bfloat16',
) -> str:
  """Run the sampler and return the generated text.

  Can be called directly from a Python console::

      from tunix.models.qwen3vl.sampler import main, load_sampler

      # One-shot (reloads model each call):
      print(main('Qwen/Qwen3-VL-4B-Instruct', 'Hello, my name is'))

      # Efficient (reuse loaded sampler):
      sampler = load_sampler('Qwen/Qwen3-VL-4B-Instruct')
      print(sampler(['Hello!', 'Tell me a joke.'], max_new_tokens=80))
  """
  image = None
  if image_url is not None:
    from PIL import Image as PILImage
    import requests

    if os.path.isfile(image_url):
      image = PILImage.open(image_url).convert('RGB')
    else:
      image = PILImage.open(
          io.BytesIO(requests.get(image_url, timeout=30).content)
      ).convert('RGB')

  sampler = load_sampler(model_id_or_dir, cache_size=cache_size, dtype=dtype)
  results = sampler(
      [prompt],
      max_new_tokens=max_new_tokens,
      images=[image] if image is not None else None,
      temperature=temperature,
      top_p=top_p,
  )
  print(f'Prompt : {prompt}')
  print(f'Output : {results[0]}')
  return results[0]


if __name__ == '__main__' and '__file__' in globals():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--model_id_or_dir',
      default='Qwen/Qwen3-VL-4B-Instruct',
      help='HuggingFace repo ID or local checkpoint directory.',
  )
  parser.add_argument(
      '--prompt',
      default=(
          '<|im_start|>user\nWhat is the capital of'
          ' France?<|im_end|>\n<|im_start|>assistant\n'
      ),
      help='Text prompt.',
  )
  parser.add_argument(
      '--image_url',
      default=None,
      help='Path or URL to an image (enables vision-language mode).',
  )
  parser.add_argument(
      '--max_new_tokens',
      type=int,
      default=50,
      help='Maximum tokens to generate.',
  )
  parser.add_argument(
      '--temperature',
      type=float,
      default=1.0,
      help='Sampling temperature (only used with --top_p).',
  )
  parser.add_argument(
      '--top_p',
      type=float,
      default=None,
      help='Top-p threshold.  Omit for greedy decoding.',
  )
  parser.add_argument(
      '--cache_size',
      type=int,
      default=512,
      help='KV-cache size in tokens.',
  )
  parser.add_argument(
      '--dtype',
      choices=['bfloat16', 'float32'],
      default='bfloat16',
      help='Compute dtype.',
  )
  _args = parser.parse_args()
  main(
      model_id_or_dir=_args.model_id_or_dir,
      prompt=_args.prompt,
      image_url=_args.image_url,
      max_new_tokens=_args.max_new_tokens,
      temperature=_args.temperature,
      top_p=_args.top_p,
      cache_size=_args.cache_size,
      dtype=_args.dtype,
  )

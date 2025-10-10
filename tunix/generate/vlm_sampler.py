"""VLM-aware sampler for PaLI-Gemma (SigLIP vision + Gemma text)."""

from __future__ import annotations

from typing import List, Optional

from flax import nnx
from flax.nnx import statelib
import jax
from jax import random
import jax.numpy as jnp
from tunix.generate import tokenizer_adapter as tok_adapt
from tunix.generate.base_sampler import BaseSampler
from tunix.generate.base_sampler import SamplerOutput
from tunix.generate.utils import preprocess_image


class VLMSampler(BaseSampler):
  """Minimal image-aware sampler.

  Notes:
    - Greedy / temperature / top-p / top-k supported.
    - No KV cache for now (keeps it simple). Add later if needed.
  """

  def __init__(
      self,
      *,
      transformer: nnx.Module,
      tokenizer: tok_adapt.TokenizerAdapter,
      image_size: int,
  ):
    self._transformer = transformer
    self._tokenizer = tokenizer
    self._image_size = image_size

  # ---- BaseSampler API ----
  @property
  def transformer(self) -> nnx.Module:
    return self._transformer

  @property
  def transformer_state(self) -> statelib.State:
    return nnx.state(self._transformer)

  def pad_id(self) -> int:
    return int(self._tokenizer.pad_id())

  def eos_id(self) -> int:
    return int(self._tokenizer.eos_id())

  def tokenize(self, input_string: str) -> jax.Array:
    ids = self._tokenizer.encode(input_string)
    return jnp.asarray(ids, dtype=jnp.int32)

  # ---- helpers ----
  def _prep_batch(
      self,
      input_strings: List[str],
      images: jax.Array,
      max_prompt_length: Optional[int],
  ):
    # tokenize and (optionally) left-pad/truncate prompts to uniform length
    token_lists = [self._tokenizer.encode(s) for s in input_strings]
    if max_prompt_length is not None:
      token_lists = [
          tl[-max_prompt_length:]
          if len(tl) > max_prompt_length
          else [self.pad_id()] * (max_prompt_length - len(tl)) + tl
          for tl in token_lists
      ]
    max_len = max(1, max(len(tl) for tl in token_lists))
    pad = self.pad_id()
    toks = jnp.asarray(
        [([pad] * (max_len - len(tl)) + tl) for tl in token_lists],
        dtype=jnp.int32,
    )
    return toks, token_lists  # ragged reference

  def _sample_logits(self, logits, temperature, top_k, top_p, rng):
    # logits: [B, V]
    if temperature is None or temperature <= 0.0:
      return jnp.argmax(logits, axis=-1)  # greedy

    lc = logits / jnp.asarray(temperature, logits.dtype)

    if top_k is not None and top_k > 0:
      topk_vals, _ = jax.lax.top_k(lc, k=top_k)
      kth = jnp.min(topk_vals, axis=-1, keepdims=True)
      mask = lc < kth
      lc = jnp.where(mask, -1e9, lc)

    if top_p is not None and 0.0 < top_p < 1.0:
      sort_idx = jnp.argsort(lc, axis=-1)[:, ::-1]
      lc_sorted = jnp.take_along_axis(lc, sort_idx, axis=-1)
      probs = jax.nn.softmax(lc_sorted, axis=-1)
      cdf = jnp.cumsum(probs, axis=-1)
      keep = cdf <= top_p
      keep = keep.at[:, 0].set(True)  # ensure at least one
      mask_sorted = jnp.logical_not(keep)
      mask = jnp.take_along_axis(
          mask_sorted, jnp.argsort(sort_idx, axis=-1), axis=-1
      )
      lc = jnp.where(mask, -1e9, lc)

    probs = jax.nn.softmax(lc, axis=-1)
    return random.categorical(rng, jnp.log(probs), axis=-1)

  def _make_pos_and_mask(self, tokens: jnp.ndarray):
    """tokens: [B, L] int32 -> (positions [B,L] int32, mask [B,L] int32)."""
    pad_id = self.pad_id()
    mask = (tokens != pad_id).astype(jnp.int32)  # 1 for real tokens
    positions = (
        jnp.cumsum(mask, axis=1) - 1
    ) * mask  # 0.. for non-pads, 0 for pads
    return positions.astype(jnp.int32), mask

  def __call__(
      self,
      input_strings: List[str],
      max_generation_steps,
      max_prompt_length=None,
      temperature=0.0,
      top_p=None,
      top_k=None,
      beam_size=None,
      seed=None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
      *,
      images: jax.Array,
  ) -> SamplerOutput:
    """Generate text conditioned on images.

    Args:
      input_strings: list of prompts (strings).
      images: float32/uint8 [B,H,W,3] raw images, preprocessed here.
    """
    assert beam_size in (
        None,
        1,
    ), "Beam search not implemented in VLMSampler v1"
    assert (
        multi_sampling == 1
    ), "multi_sampling not implemented in VLMSampler v1"

    if images.dtype != jnp.uint8:
      images = images.astype(jnp.uint8)
    imgs = preprocess_image(images, self._image_size)  # [B,S,S,3] float32
    imgs = imgs.astype(jnp.float32)

    # Tokenize batch
    prompt_tokens, _ = self._prep_batch(input_strings, imgs, max_prompt_length)
    B = prompt_tokens.shape[0]

    # Generation loop (no KV cache; feed 1..T+step each time)
    rng = random.PRNGKey(0) if seed is None else seed
    seq = prompt_tokens.astype(jnp.int32)
    collected_logits = []  # list of [B, V]
    out_tokens = []

    for _ in range(int(max_generation_steps)):
      positions, attn_mask = self._make_pos_and_mask(seq)
      # Call transformer with its exact signature:
      # (last_tokens, positions, cache, attention_mask, *, pixel_values=None, output_hidden_states=False)
      logits, _ = self._transformer(
          last_tokens=seq,  # [B, L]
          positions=positions,  # [B, L]
          cache=None,
          attention_mask=attn_mask,  # [B, L]
          pixel_values=imgs,  # [B, S, S, 3]
          output_hidden_states=False,
      )  # -> [B, L, V]

      next_logits = logits[:, -1, :]      # [B, V]
      if return_logits:
        collected_logits.append(next_logits)

      rng, sub = random.split(rng)
      next_ids = self._sample_logits(
          next_logits, temperature, top_k, top_p, sub
      )  # [B]
      out_tokens.append(next_ids)

      # Append to sequence
      seq = jnp.concatenate([seq, next_ids[:, None]], axis=1).astype(jnp.int32)

      # Early stop if all EOS
      if jnp.all(next_ids == self.eos_id()):
        break

    # Build outputs
    gen_tokens = (
        jnp.stack(out_tokens, axis=1)
        if out_tokens
        else jnp.zeros((B, 0), dtype=jnp.int32)
    )
    tokens_full = (
        jnp.concatenate([prompt_tokens, gen_tokens], axis=1)
        if echo
        else gen_tokens
    )

    # Decode
    texts = []
    for b in range(B):
      toks = list(tokens_full[b].tolist())
      if not echo and self.eos_id() in toks:
        toks = toks[: toks.index(self.eos_id()) + 1]
      texts.append(self._tokenizer.decode(toks))

    logits_out = (
        jnp.stack(collected_logits, axis=1)
        if (return_logits and collected_logits)
        else None
    )

    return SamplerOutput(
        text=texts,
        logits=logits_out,
        tokens=tokens_full,
        padded_prompt_tokens=prompt_tokens,
        logprobs=None,
    )
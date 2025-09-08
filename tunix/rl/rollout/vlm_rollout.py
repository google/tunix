"""VLM rollout worker (vanilla engine) using VLMSampler and PaLI-Gemma."""

from __future__ import annotations
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from tunix.rl.rollout.base_rollout import BaseRollout, RolloutConfig, RolloutOutput
from tunix.generate.vlm_sampler import VLMSampler
from tunix.generate import tokenizer_adapter as tok_adapt
from tunix.models.paligemma import model as pali_model


class VLMRollout(BaseRollout):
  def __init__(
      self,
      *,
      model: pali_model.PaLIGemma,
      tokenizer: tok_adapt.TokenizerAdapter,
      image_size: int,
  ):
    self._model = model
    self._tokenizer = tokenizer
    self._sampler = VLMSampler(
        transformer=model,
        tokenizer=tokenizer,
        image_size=image_size,
    )

  # ---- BaseRollout API ----
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """kwargs must include: images: jnp.ndarray [B,H,W,3] (uint8/float32)"""
    if "images" not in kwargs:
      raise ValueError("VLMRollout.generate requires kwargs['images']")
    images = kwargs["images"]

    out = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        return_logits=True,
        echo=False,
        images=images,
    )

    # SamplerOutput -> RolloutOutput field mapping
    return RolloutOutput(
        text=out.text,
        logits=out.logits if out.logits is not None else jnp.zeros((0,), jnp.float32),
        tokens=out.tokens,
        left_padded_prompt_tokens=out.padded_prompt_tokens,
        logprobs=out.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,       # [B, T_prompt]
      completion_tokens: jax.Array,   # [B, T_gen]
      *,
      images: jax.Array,
  ) -> jax.Array:
    """Teacher-forced log-probs for completions given prompts + images.

    Returns:
      logps: [B, T_gen] float32
    """
    # Build full input (prompt + completion[:-1]) to next-token predict completion
    B, T_prompt = prompt_tokens.shape
    B2, T_gen = completion_tokens.shape
    assert B == B2

    # preprocess images once
    from tunix.models.siglip import preprocess as siglip_pp
    imgs = images
    if imgs.dtype == jnp.uint8:
      imgs = imgs.astype(jnp.float32)
    imgs = siglip_pp.preprocess(imgs, self._sampler._image_size)

    # Feed prompt + completion except last token
    inp = jnp.concatenate([prompt_tokens, completion_tokens[:, :-1]], axis=1) if T_gen > 0 else prompt_tokens
    logits, _ = self._model(
        images=imgs,
        input_tokens=inp,
        positions=None,
        cache=None,
        attention_mask=None,
    )  # [B, T_total, V]
    # Slice logits aligned to completion positions
    logits_comp = logits[:, -T_gen:, :] if T_gen > 0 else logits[:, -0:, :]
    # Gather log probs of the actual completion tokens
    probs = jax.nn.softmax(logits_comp, axis=-1)
    gather = jnp.take_along_axis(probs, completion_tokens[..., None], axis=-1)[..., 0]
    logps = jnp.log(gather + 1e-9)
    return logps

  def update_params(
      self,
      params: Any,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Update PaLI-Gemma parameters (e.g., after GRPO step)."""
    if filter_types is None:
      nnx.update(self._model, params)
    else:
      # Only update selected types (e.g., nnx.LoRAParam)
      current = nnx.state(self._model, *filter_types)
      merged = jax.tree.map(lambda a, b: b if b is not None else a, current, params)
      nnx.update(self._model, merged)

  def pad_id(self) -> int:
    return int(self._tokenizer.pad_id)

  def eos_id(self) -> int:
    return int(self._tokenizer.eos_id)

  def model(self) -> Any:
    return self._model

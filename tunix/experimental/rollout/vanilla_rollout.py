# Copyright 2026 The Tunix Authors.
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

"""Rollout worker backed by the continuous batching engine.

Selected with `rollout_engine='vanilla_v2'`, it is the successor to
`tunix.rl.rollout.vanilla_rollout`.
"""

import dataclasses
from typing import Any

from flax import nnx
import jax
import jaxtyping
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter
from tunix.generate import utils
from tunix.rl import common
from tunix.rl.rollout import base_rollout


def _build_engine(
    model: nnx.Module,
    tokenizer: tokenizer_adapter.TokenizerAdapter,
    rollout_config: base_rollout.RolloutConfig,
    mesh: jax.sharding.Mesh,
    max_model_len: int,
) -> engine_lib.LLMEngine:
  """Returns an engine running `model` as `rollout_config` describes.

  Args:
    model: The model to sample from. Its `cache_geometries` returns the
      geometry of each of its KV caches.
    tokenizer: The tokenizer for text prompts.
    rollout_config: The engine's settings.
    mesh: The mesh the model runs on.
    max_model_len: The maximum number of tokens a sequence may hold.

  Raises:
    ValueError: If `rollout_config.rollout_vanilla_v2_hbm_utilization` is
      invalid or device HBM is insufficient.
  """
  eos_token_ids: set[int] = set()
  if tokenizer.eos_id() is not None:
    eos_token_ids.add(tokenizer.eos_id())
  if rollout_config.eos_tokens:
    eos_token_ids.update(rollout_config.eos_tokens)
  rollout_config = dataclasses.replace(
      rollout_config, eos_tokens=list(eos_token_ids)
  )
  return engine_lib.LLMEngine(
      model,
      rollout_config=rollout_config,
      mesh=mesh,
      max_model_len=max_model_len,
  )


class VanillaRollout(base_rollout.BaseRollout):
  """Rollout worker backed by the continuous batching `Sampler`."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      rollout_config: base_rollout.RolloutConfig,
      mesh: jax.sharding.Mesh,
      max_model_len: int,
  ):
    """Initializes the rollout worker.

    Args:
      model: The rollout actor. Must have a `config` with a `dtype`, and
        implement `cache_geometries()`.
      tokenizer: A `TokenizerAdapter`, or a raw tokenizer to wrap in one.
      rollout_config: The engine's settings. The engine is built once from this
        config; later `generate` calls only vary the per-request sampling
        parameters, except `return_logprobs`, `return_logits`, and `top_k`,
        which are baked into the compiled step and cannot exceed the config it
        is given. Its `seed` seeds the engine's sampling keys.
      mesh: The mesh the model runs on.
      max_model_len: The maximum number of tokens a sequence may hold, prompt
        and generated.
    """
    if not isinstance(tokenizer, tokenizer_adapter.TokenizerAdapter):
      tokenizer = tokenizer_adapter.TokenizerAdapter(tokenizer)
    self._tokenizer = tokenizer
    self._graphdef = nnx.graphdef(model)
    self._sampler = sampler_lib.Sampler(
        _build_engine(model, tokenizer, rollout_config, mesh, max_model_len),
        tokenizer=tokenizer,
        server_mode=rollout_config.rollout_vanilla_v2_server_mode,
        server_submission_threshold=rollout_config.rollout_vanilla_v2_server_mode_submission_threshold,
        server_submission_timeout_s=rollout_config.rollout_vanilla_v2_server_mode_submission_timeout_s,
    )

  def generate(
      self,
      prompts: list[str] | None,
      rollout_config: base_rollout.RolloutConfig,
      *,
      prompt_token_ids: list[Any] | None = None,
      route_key: int | str | None = None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates a completion for each prompt.

    Args:
      prompts: The text prompts, or None if `prompt_token_ids` is given.
      rollout_config: The sampling settings for this call. Its `seed` is not
        used: the engine was seeded once, when it was built.
      prompt_token_ids: The prompts as token ids, used instead of `prompts`.
      route_key: Optional key of the prompts' data parallel rank. Prompts with
        the same key run in the same rank, and so can share its prefix cache.
      **kwargs: Ignored.

    Returns:
      The completions, in the order the prompts were given.
    """
    del kwargs
    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        echo=False,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        pad_output=False,
        return_logprobs=rollout_config.return_logprobs,
        return_logits=rollout_config.return_logits,
        prompt_token_ids=prompt_token_ids,
        route_key=route_key,
    )
    return base_rollout.RolloutOutput(
        text=output.text,
        logits=output.logits,  # pyrefly: ignore[bad-argument-type]
        tokens=output.tokens,  # pyrefly: ignore[bad-argument-type]
        left_padded_prompt_tokens=output.padded_prompt_tokens,
        logprobs=output.logprobs,  # pyrefly: ignore[bad-argument-type]
        prompt_lengths=output.prompt_lengths,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    return common.compute_per_token_logps(  # pytype: disable=bad-return-type
        self._graphdef,
        self._sampler.transformer_state,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        stop_gradient=True,
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    self._sampler.update_params(params, filter_types)

  def reinitialize_cache(self) -> None:
    """Discards cached KV after the shared model weights change in place."""
    self._sampler.reinitialize_cache()

  def release_for_training(
      self,
      free_weights: bool = True,
      src_state: jaxtyping.PyTree | None = None,
  ) -> None:
    """Releases the KV cache and optionally model weights before training."""
    self._sampler.delete_cache()
    if free_weights:
      jax.effects_barrier()
      utils.delete_state_buffers(
          self._sampler.transformer_state, src_state=src_state
      )

  def restore_after_training(self) -> None:
    """Reallocates the KV cache if it was deleted before training."""
    self._sampler.reinitialize_cache()

  def pad_id(self) -> int:
    return self._tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return nnx.merge(self._graphdef, self._sampler.transformer_state)

  def close(self) -> None:
    """Stops the engine loop, in server mode."""
    self._sampler.stop()

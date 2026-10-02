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

Selected with `rollout_engine='vanillav2'`, it is the successor to
`tunix.rl.rollout.vanilla_rollout`.
"""

import itertools
from typing import Any

from flax import nnx
import jax
import jaxtyping
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.rollout import sampler as rollout_sampler_lib
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
    model: The model to sample from. Its `init_kv_cache` builds its KV caches.
    tokenizer: The tokenizer for text prompts.
    rollout_config: The engine's settings.
    mesh: The mesh the model runs on.
    max_model_len: The maximum number of tokens a sequence may hold.

  Raises:
    ValueError: If `rollout_config.kv_cache_max_device_bytes` is not set.
  """
  if rollout_config.kv_cache_max_device_bytes is None:
    raise ValueError(
        "rollout_engine='vanillav2' requires"
        ' rollout_config.kv_cache_max_device_bytes.'
    )
  if rollout_config.eos_tokens:
    eos_token_ids = frozenset(rollout_config.eos_tokens)
  else:
    eos_token_ids = frozenset([tokenizer.eos_id()])
  return engine_lib.LLMEngine(
      model,
      tokenizer=tokenizer,
      cache_config=kv_cache_manager_lib.CacheConfig(
          max_device_bytes=rollout_config.kv_cache_max_device_bytes,
          page_size=rollout_config.kv_cache_page_size,
          enable_prefix_caching=rollout_config.enable_prefix_caching,
          dtype=model.config.dtype,
      ),
      scheduler_config=scheduler_lib.SchedulerConfig(
          max_num_batched_tokens=rollout_config.max_num_batched_tokens,
          max_num_seqs=rollout_config.max_num_seqs,
          chunked_prefill_length=rollout_config.chunked_prefill_length,
          num_scheduler_steps=rollout_config.num_scheduler_steps,
          eos_token_ids=eos_token_ids,
      ),
      model_runner_config=model_runner_lib.ModelRunnerConfig(
          max_top_k=(
              -1 if rollout_config.top_k is None else rollout_config.top_k
          ),
          mesh=mesh,
          return_logprobs=rollout_config.return_logprobs,
          num_scheduler_steps=rollout_config.num_scheduler_steps,
          seed=0 if rollout_config.seed is None else int(rollout_config.seed),
      ),
      max_model_len=max_model_len,
  )


class VanillaRollout(base_rollout.BaseRollout):
  """Rollout worker backed by the continuous batching engine."""

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
      model: The model to sample from.
      tokenizer: The tokenizer prompts are encoded with.
      rollout_config: Configures the engine, which is built once. `generate`
        only reads its per-request settings from the config it is given.
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
        server_mode=rollout_config.server_mode,
        submission_threshold=rollout_config.server_mode_submission_threshold,
        submission_timeout_s=rollout_config.server_mode_submission_timeout_s,
    )
    self._request_ids = itertools.count()

  def generate(
      self,
      prompts: list[str] | None,
      rollout_config: base_rollout.RolloutConfig,
      *,
      prompt_token_ids: list[Any] | None = None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates a completion for each prompt.

    Args:
      prompts: The text prompts, or None if `prompt_token_ids` is given.
      rollout_config: The sampling settings for this call.
      prompt_token_ids: The prompts as token ids, used instead of `prompts`.
      **kwargs: Ignored.

    Returns:
      The completions, in the order the prompts were given.
    """
    del kwargs
    if prompt_token_ids is not None:
      inputs = [np.asarray(ids, dtype=np.int32) for ids in prompt_token_ids]
    else:
      inputs = list(prompts or [])
    sampling_params = rollout_sampler_lib.SamplingParams(
        max_tokens=rollout_config.max_tokens_to_generate,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        return_logprobs=rollout_config.return_logprobs,
    )
    responses = self._sampler.generate([
        rollout_sampler_lib.SamplingRequest(
            request_id=str(next(self._request_ids)),
            prompt=prompt,
            sampling_params=sampling_params,
        )
        for prompt in inputs
    ])

    left_padded_prompt_tokens, prompt_lengths, _ = utils.left_pad_prompt_tokens(
        [response.prompt_token_ids for response in responses],
        rollout_config.max_prompt_length,
        self.pad_id(),
    )
    return base_rollout.RolloutOutput(
        text=[response.text for response in responses],
        logits=None,
        tokens=[response.token_ids for response in responses],
        left_padded_prompt_tokens=left_padded_prompt_tokens,
        logprobs=(
            [response.logprobs for response in responses]  # pyrefly: ignore[bad-argument-type]
            if rollout_config.return_logprobs
            else None
        ),
        prompt_lengths=prompt_lengths,
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

  def pad_id(self) -> int:
    return self._tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return nnx.merge(self._graphdef, self._sampler.transformer_state)

  def close(self) -> None:
    """Stops the engine loop, in server mode."""
    self._sampler.stop()

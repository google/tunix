# Copyright 2026 Google LLC
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

"""Vanilla rollout worker, backed by the Tunix experimental Sampler.

This is the successor to `tunix.rl.rollout.vanilla_rollout`, which samples a
whole batch in lockstep with `tunix.generate.sampler.Sampler`. Select it with
`rollout_engine='vanillav2'`.
"""

from typing import Any, Optional, Tuple

from absl import logging
from flax import nnx
import jax
import jaxtyping
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import utils as engine_utils
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.rl import common
from tunix.rl.rollout import base_rollout


def _resolve_eos_token_ids(
    rollout_config: base_rollout.RolloutConfig,
    tokenizer: Any,
) -> tuple[int, ...]:
  """Resolves the EOS tokens to stamp onto every request.

  EOS is a per-request property, so it is resolved here and attached to each
  request rather than held on the engine.

  Args:
    rollout_config: The rollout worker's config.
    tokenizer: The tokenizer, used for the EOS token when the rollout config
      does not name one.

  Returns:
    The EOS token ids.
  """
  if rollout_config.eos_tokens:
    return tuple(rollout_config.eos_tokens)
  if hasattr(tokenizer, 'eos_id'):
    tok_eos = tokenizer.eos_id()
    return tuple(tok_eos) if isinstance(tok_eos, (list, tuple)) else (tok_eos,)
  return (1,)


def _engine_config_for(
    rollout_config: base_rollout.RolloutConfig,
    model: nnx.Module,
) -> engine_lib.EngineConfig:
  """Maps a rollout config, plus the model's sharding, onto an engine config.

  Args:
    rollout_config: The rollout worker's config.
    model: The model the engine samples from, whose sharding the KV cache
      follows.

  Returns:
    The config for an engine serving this rollout worker.
  """

  # Without a temperature there is nothing to sample from, so decoding is
  # greedy and the top_p knobs do not apply.
  if rollout_config.temperature and rollout_config.temperature > 0.0:
    sampling_mode = model_runner_lib.TOP_P_SAMPLING_MODE
    sampling_parameters: dict[str, float | int] = {
        model_runner_lib.TEMPERATURE: rollout_config.temperature
    }
    if rollout_config.top_p is not None:
      sampling_parameters[model_runner_lib.TOP_P] = rollout_config.top_p
    if rollout_config.top_k is not None:
      sampling_parameters[model_runner_lib.TOP_K] = rollout_config.top_k
  else:
    sampling_mode = model_runner_lib.GREEDY_SAMPLING_MODE
    sampling_parameters = {}

  model_config = model.config  # pyrefly: ignore[attribute-error]
  transformer_state = nnx.state(model)
  dp_axis, tp_axis, dp_size, mesh = engine_utils.derive_sharding(
      model_config, transformer_state
  )

  forbidden_ids = rollout_config.forbidden_tokens or set()

  # Read the field directly. A `getattr` default silently turns prefix caching
  # back on whenever the field is renamed or the config object is rebuilt
  # without it, which is indistinguishable in the logs from asking for it.
  enable_prefix_caching = rollout_config.rollout_vanilla_enable_prefix_caching
  logging.info(
      '[vanillav2] prefix caching %s (rollout_config id=%s)',
      'ENABLED' if enable_prefix_caching else 'DISABLED',
      id(rollout_config),
  )

  model_dtype = (
      rollout_config.data_type
      or engine_utils.model_dtype(model_config, transformer_state)
  )

  return engine_lib.EngineConfig(
      max_num_seqs=rollout_config.max_num_seqs,
      max_prompt_length=rollout_config.max_prompt_length,
      max_tokens_to_generate=rollout_config.max_tokens_to_generate,
      forbidden_token_ids=forbidden_ids,
      enable_prefix_caching=enable_prefix_caching,
      sampling_mode=sampling_mode,
      sampling_parameters=sampling_parameters,
      return_logprobs=rollout_config.return_logprobs,
      dtype=model_dtype,
      dp_axis=dp_axis,
      tp_axis=tp_axis,
      dp_size=dp_size,
      mesh=mesh,
  )


class VanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      rollout_config: base_rollout.RolloutConfig,
  ):
    """Initializes the rollout worker.

    Args:
      model: The model to sample from.
      tokenizer: The tokenizer prompts are encoded with.
      rollout_config: How to sample. The config is used once to build the
        underlying engine and sampler for the worker's lifetime.
    """
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self._transformer = model
    self._tokenizer = tokenizer
    self._rollout_config = rollout_config
    self._engine_config = _engine_config_for(rollout_config, model)
    self._sampler = sampler_lib.Sampler(
        transformer=self._transformer,
        tokenizer=self._tokenizer,
        engine_config=self._engine_config,
        enable_server_mode=True,
        submission_threshold=(
            self._rollout_config.rollout_vanilla_server_mode_submission_threshold
        ),
        submission_timeout_s=(
            self._rollout_config.rollout_vanilla_server_mode_submission_timeout_s
        ),
    )

  @property
  def sampler(self) -> sampler_lib.Sampler:
    """The sampler this worker generates with."""
    return self._sampler

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig | None = None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model.

    Args:
      prompts: The prompts to sample from.
      rollout_config: An optional config used strictly to gather request-level
        parameters (e.g. max_tokens_to_generate, eos_tokens) for this
        generation. The engine is never rebuilt.
      **kwargs: Additional overrides for the sampler.

    Returns:
      The generated samples, in the order the prompts were given.
    """
    if rollout_config is None:
      rollout_config = self._rollout_config

    max_tokens = kwargs.get(
        'max_tokens_to_generate', rollout_config.max_tokens_to_generate
    )
    # EOS rides on the request, so it has to be resolved here — including the
    # tokenizer fallback — rather than left to the engine.
    eos_ids = kwargs.get(
        'eos_token_ids',
        _resolve_eos_token_ids(rollout_config, self._tokenizer),
    )

    sampler_output = self._sampler(
        input_strings=prompts,
        max_tokens_to_generate=max_tokens,
        eos_token_ids=eos_ids,
    )

    logprobs = (
        [np.asarray(lp, dtype=np.float32) for lp in sampler_output.logprobs]
        if sampler_output.logprobs is not None
        else None
    )

    return base_rollout.RolloutOutput(
        text=sampler_output.text,
        logits=sampler_output.logits,  # pyrefly: ignore[bad-argument-type]
        tokens=sampler_output.tokens,  # pyrefly: ignore[bad-argument-type]
        left_padded_prompt_tokens=sampler_output.padded_prompt_tokens,
        logprobs=logprobs,
        routed_experts=sampler_output.routed_experts,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    graphdef, state = self._sampler.model_def_and_state()
    return common.compute_per_token_logps(  # pytype: disable=bad-return-type
        graphdef,
        state,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        stop_gradient=True,
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Syncs trained weights onto the sampler.

    Args:
      params: The new parameters.
      filter_types: Which variable types of the model's state to update. All of
        them when None.
    """
    self._sampler.update_params(params, filter_types)

  def pad_id(self) -> int:
    return self._sampler.pad_id()

  def eos_id(self) -> int:
    return self._sampler.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer

  def shutdown(self) -> None:
    """Shuts down the underlying sampler."""
    self._sampler.shutdown()

  def close(self) -> None:
    """Closes the rollout worker and shuts down the sampler."""
    self.shutdown()



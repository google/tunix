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

from collections.abc import Hashable
from typing import Any

from flax import nnx
import jax
import jaxtyping
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import metrics as metrics_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.rl.rollout import base_rollout


_GIB = 1 << 30


def _resolve_max_device_size_gib(
    hbm_utilization: float,
    mesh: jax.sharding.Mesh,
) -> float:
  """Computes available per-device KV cache GiB from an HBM utilization cap.

  Matches vLLM's `determine_available_memory`: budgets `int(bytes_limit *
  hbm_utilization)` total HBM per device and subtracts `bytes_in_use` (e.g.
  already-allocated model weights), returning the minimum across local devices
  in GiB.

  Args:
    hbm_utilization: Fraction of total device HBM (`bytes_limit`) to budget, in
      `(0, 1]`.
    mesh: The JAX mesh whose local devices are queried.

  Returns:
    The minimum available GiB per device across queried devices.

  Raises:
    ValueError: If `hbm_utilization` is out of `(0, 1]`, no local devices are
      available, `memory_stats()` does not report `bytes_limit` /
      `bytes_in_use`, or `bytes_in_use` already exceeds the utilization cap.
  """
  if not 0.0 < hbm_utilization <= 1.0:
    raise ValueError(
        f'rollout_hbm_utilization must be in (0, 1], got {hbm_utilization}.'
    )
  devices = list(mesh.local_devices)
  if not devices:
    raise ValueError('No addressable JAX devices found to query HBM stats.')

  available_per_device: list[int] = []
  for device in devices:
    stats = device.memory_stats()
    if not stats or 'bytes_limit' not in stats or 'bytes_in_use' not in stats:
      raise ValueError(
          f'Device {device} does not report HBM memory_stats() '
          '(bytes_limit / bytes_in_use).'
      )
    limit = int(stats['bytes_limit'])
    used = int(stats['bytes_in_use'])
    limit_cap = int(limit * hbm_utilization)
    avail = limit_cap - used
    if avail <= 0:
      raise ValueError(
          f'Insufficient HBM on device {device} for rollout_hbm_utilization='
          f'{hbm_utilization}: bytes_in_use ({used}) >= '
          f'int(bytes_limit ({limit}) * {hbm_utilization}) ({limit_cap}).'
      )
    available_per_device.append(avail)

  return min(available_per_device) / _GIB


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
    ValueError: If `rollout_config.rollout_hbm_utilization` is invalid or device
      HBM is insufficient.
  """
  max_device_size_gib = _resolve_max_device_size_gib(
      rollout_config.rollout_hbm_utilization, mesh
  )
  eos_token_ids: set[int] = set()
  if tokenizer.eos_id() is not None:
    eos_token_ids.add(tokenizer.eos_id())
  if rollout_config.eos_tokens:
    eos_token_ids.update(rollout_config.eos_tokens)
  return engine_lib.LLMEngine(
      model,
      tokenizer=tokenizer,
      cache_config=kv_cache_manager_lib.CacheConfig(
          max_device_size_gib=max_device_size_gib,
          max_host_size_gib=rollout_config.host_size_gib,
          page_size=rollout_config.kv_cache_page_size,
          enable_prefix_caching=rollout_config.enable_prefix_caching,
          dtype=model.config.dtype,  # pyrefly: ignore[missing-attribute]
          use_raiden=rollout_config.use_raiden,
      ),
      scheduler_config=scheduler_lib.SchedulerConfig(
          max_num_batched_tokens=rollout_config.max_num_batched_tokens,
          max_num_seqs=rollout_config.max_num_seqs,
          max_chunked_prefill_length=rollout_config.chunked_prefill_length,
          num_scheduler_steps=rollout_config.num_scheduler_steps,
      ),
      model_runner_config=model_runner_lib.ModelRunnerConfig(
          max_top_k=(
              -1 if rollout_config.top_k is None else rollout_config.top_k
          ),
          mesh=mesh,
          return_logprobs=rollout_config.return_logprobs,
          num_scheduler_steps=rollout_config.num_scheduler_steps,
          # Seeds the keys the engine draws for every request. `generate`
          # never sets a per-request seed, so the generations of a GRPO group,
          # which share a prompt, still sample independently.
          seed=0 if rollout_config.seed is None else int(rollout_config.seed),
      ),
      max_model_len=max_model_len,
      eos_token_ids=frozenset(eos_token_ids),
      log_stats_interval_s=rollout_config.log_stats_interval_s,
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
        only reads its per-request settings from the config it is given. Its
        `seed` seeds the engine's sampling keys.
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
      rollout_config: The sampling settings for this call. Its `seed` is not
        used: the engine was seeded once, when it was built.
      prompt_token_ids: The prompts as token ids, used instead of `prompts`.
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
        # A request seed pins the sample, so every generation of a GRPO group,
        # which shares a prompt, would sample the same tokens. Unseeded
        # requests draw fresh keys from the seeded engine instead.
        seed=None,
        pad_output=False,
        return_logprobs=rollout_config.return_logprobs,
        prompt_token_ids=prompt_token_ids,
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

  def pad_id(self) -> int:
    return self._tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return nnx.merge(self._graphdef, self._sampler.transformer_state)

  def get_metrics(self) -> metrics_lib.EngineMetricsSnapshot:
    """Returns a cumulative metrics snapshot from the underlying sampler."""
    return self._sampler.get_metrics()

  def get_perf_metrics(self) -> dict[str, metrics_lib.PerfMetricValueT]:
    """Flushes and returns per-step rollout metrics for logging."""
    return self._sampler.flush_step_metrics().to_perf_metrics()

  def record_batch_start(self, batch_id: Hashable | None = None) -> None:
    """Marks the start of a rollout batch on the underlying sampler."""
    self._sampler.record_batch_start(batch_id)

  def record_batch_completion(
      self, duration_s: float, batch_id: Hashable | None = None
  ) -> None:
    """Records a completed rollout batch duration on the underlying sampler."""
    self._sampler.record_batch_completion(duration_s, batch_id)

  def close(self) -> None:
    """Stops the engine loop, in server mode."""
    self._sampler.stop()

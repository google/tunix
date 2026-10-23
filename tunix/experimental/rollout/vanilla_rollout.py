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

from collections import defaultdict
from typing import Any

from flax import nnx
import jax
import jaxtyping
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.rl.rollout import base_rollout


_GIB = 1 << 30


def _device_hbm_limit_bytes(device: Any) -> int:
  """Returns the total HBM bytes per JAX device from its `device_kind`."""
  kind = device.device_kind
  if 'TPU' not in kind:
    raise ValueError(
        f'Device {device} does not report HBM memory_stats() and has '
        f'unsupported device_kind={kind!r}.'
    )
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')] + 'e'
  if kind in ('TPU v5p', 'TPU v5'):
    return 95 * _GIB
  if kind == 'TPU v5e':
    return 16 * _GIB
  if kind in ('TPU v6e', 'TPU v4'):
    return 32 * _GIB
  if kind.startswith('TPU7') or kind == 'TPU v7':
    return 96 * _GIB
  if kind.startswith('TPU8') or kind == 'TPU v8i':
    return 144 * _GIB
  raise ValueError(f'Unknown TPU device_kind={kind!r} on {device}.')


def _live_arrays_hbm_usage_bytes(devices: list[Any]) -> dict[Any, int]:
  """Computes per-device HBM bytes in use from `jax.live_arrays()` (Pathways)."""
  used: dict[Any, int] = defaultdict(int)
  seen_buffers: set[int] = set()
  for array in jax.live_arrays():
    if array.is_deleted():
      continue
    for shard in array.addressable_shards:
      buf_id = id(shard.data)
      if buf_id not in seen_buffers:
        seen_buffers.add(buf_id)
        used[shard.data.device] += shard.data.nbytes
  return {d: used[d] for d in devices}


def _resolve_max_device_size_gib(
    hbm_utilization: float,
    mesh: jax.sharding.Mesh,
) -> float:
  """Computes available per-device KV cache GiB from an HBM utilization cap.

  Matches vLLM's `determine_available_memory`: budgets `int(bytes_limit *
  hbm_utilization)` total HBM per device and subtracts `bytes_in_use` (e.g.
  already-allocated model weights), returning the minimum across local devices
  in GiB. Falls back to `jax.live_arrays()` + `device_kind` HBM limits when
  running under Pathways (where IFRT proxy devices do not implement
  `memory_stats()`).

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
        'rollout_vanilla_v2_hbm_utilization must be in (0, 1], got'
        f' {hbm_utilization}.'
    )
  devices = list(mesh.local_devices)
  if not devices:
    raise ValueError('No addressable JAX devices found to query HBM stats.')

  available_per_device: list[int] = []
  live_used_by_device: dict[Any, int] | None = None
  for device in devices:
    try:
      stats = device.memory_stats()
    except jax.errors.JaxRuntimeError:
      stats = None
    if not stats or 'bytes_limit' not in stats or 'bytes_in_use' not in stats:
      if live_used_by_device is None:
        live_used_by_device = _live_arrays_hbm_usage_bytes(devices)
      stats = {
          'bytes_limit': _device_hbm_limit_bytes(device),
          'bytes_in_use': live_used_by_device[device],
      }
    limit = int(stats['bytes_limit'])
    used = int(stats['bytes_in_use'])
    limit_cap = int(limit * hbm_utilization)
    avail = limit_cap - used
    if avail <= 0:
      raise ValueError(
          f'Insufficient HBM on device {device} for'
          f' rollout_vanilla_v2_hbm_utilization={hbm_utilization}:'
          f' bytes_in_use ({used}) >='
          f' int(bytes_limit ({limit}) * {hbm_utilization}) ({limit_cap}).'
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
    ValueError: If `rollout_config.rollout_vanilla_v2_hbm_utilization` is
      invalid or device HBM is insufficient.
  """
  max_device_size_gib = _resolve_max_device_size_gib(
      rollout_config.rollout_vanilla_v2_hbm_utilization, mesh
  )
  eos_token_ids: set[int] = set()
  if tokenizer.eos_id() is not None:
    eos_token_ids.add(tokenizer.eos_id())
  if rollout_config.eos_tokens:
    eos_token_ids.update(rollout_config.eos_tokens)
  tp_axis = 'tp' if 'tp' in mesh.axis_names else None
  return engine_lib.LLMEngine(
      model,
      cache_config=kv_cache_manager_lib.CacheConfig(
          max_device_size_gib=max_device_size_gib,
          max_host_size_gib=rollout_config.rollout_vanilla_v2_host_size_gib,
          page_size=rollout_config.rollout_vanilla_v2_kv_cache_page_size,
          enable_prefix_caching=rollout_config.rollout_vanilla_v2_enable_prefix_caching,
          dtype=model.config.dtype,  # pyrefly: ignore[missing-attribute]
          tp_axis=tp_axis,
          mesh=mesh if tp_axis is not None else None,
      ),
      scheduler_config=scheduler_lib.SchedulerConfig(
          max_num_batched_tokens=rollout_config.rollout_vanilla_v2_max_num_batched_tokens,
          max_num_seqs=rollout_config.rollout_vanilla_v2_max_num_seqs,
          max_chunked_prefill_length=rollout_config.rollout_vanilla_v2_chunked_prefill_length,
          num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
      ),
      model_runner_config=model_runner_lib.ModelRunnerConfig(
          max_top_k=rollout_config.top_k or -1,
          mesh=mesh,
          return_logprobs=rollout_config.return_logprobs,
          num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
          seed=0 if rollout_config.seed is None else int(rollout_config.seed),
      ),
      max_model_len=max_model_len,
      eos_token_ids=frozenset(eos_token_ids),
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
        implement `init_kv_cache(cache_config)`.
      tokenizer: A `TokenizerAdapter`, or a raw tokenizer to wrap in one.
      rollout_config: The engine's settings. The engine is built once from this
        config; later `generate` calls only vary the per-request sampling
        parameters, except `return_logprobs` and `top_k`, which are baked into
        the compiled step and cannot exceed the config it is given. Its
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

  def reinitialize_cache(self) -> None:
    """Discards cached KV after the shared model weights change in place."""
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

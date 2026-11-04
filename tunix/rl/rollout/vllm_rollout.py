# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM rollout worker with Tunix sampler."""

from collections.abc import Sequence
from typing import Any, Optional, Tuple

from absl import logging
from flax import nnx
import jax
import jaxtyping
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


class VllmRollout(base_rollout.BaseRollout):
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    self._partial_rollout = rollout_config.partial_rollout
    if self._partial_rollout:
      _check_partial_rollout_config(rollout_config)
    # Only the "keep" policy leaves the KV blocks of frozen requests resident
    # across a weight sync.
    self._keep_kv_across_sync = (
        self._partial_rollout
        and rollout_config.partial_rollout_kv_policy == "keep"
    )
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="vllm_jax",
    )
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_sampler.VllmConfig(
            server_mode=rollout_config.rollout_vllm_server_mode,
            server_mode_submission_threshold=(
                rollout_config.rollout_vllm_server_mode_submission_threshold
            ),
            server_mode_submission_timeout_s=(
                rollout_config.rollout_vllm_server_mode_submission_timeout_s
            ),
            mapping_config=mapping_config,
            return_logprobs=rollout_config.return_logprobs,
            return_routed_experts=rollout_config.return_routed_experts,
            eos_tokens=rollout_config.eos_tokens,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            additional_config=rollout_config.rollout_vllm_additional_config,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            lora_config=rollout_config.rollout_vllm_lora_config,
            mesh=mesh,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            delete_dst_buffers=rollout_config.rollout_vllm_delete_dst_buffers,
            reshard_chunk_size=rollout_config.rollout_vllm_reshard_chunk_size,
            free_kv_cache_during_weight_sync=(
                rollout_config.rollout_vllm_free_kv_cache_during_weight_sync
            ),
            overlap_postprocessing=(
                rollout_config.rollout_vllm_overlap_postprocessing
            ),
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "async_scheduling": (
                    rollout_config.rollout_vllm_async_scheduling
                ),
                "max_num_batched_tokens": (
                    rollout_config.rollout_vllm_max_num_batched_tokens
                ),
                "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                "max_logprobs": (
                    1
                ),  # We only need the logprobs of the sampled tokens
                "logprobs_mode": rollout_config.rollout_vllm_logprobs_mode,
                **rollout_config.rollout_vllm_kwargs,
            },
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
        ),
    )
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._sampler.mesh

  def generate(
      self,
      prompts: list[str] | None,
      rollout_config: base_rollout.RolloutConfig,
      *,
      prompt_token_ids: Sequence[Sequence[int]] | None = None,
      **kwargs: Any,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model.

    Args:
      prompts: Text prompts to sample from, or `None` when `prompt_token_ids` is
        provided.
      rollout_config: Sampling and generation limits for the rollout call.
      prompt_token_ids: Optional explicit unpadded token rows passed directly to
        the vLLM sampler instead of tokenizing `prompts`.
      **kwargs: Extra per-call keyword arguments forwarded to `VllmSampler`
        (such as `cache_salt`).

    Returns:
      A `RolloutOutput` containing generated text, tokens, and logprobs.
    """
    if prompt_token_ids is not None:
      kwargs["prompt_token_ids"] = prompt_token_ids
    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,  # pyrefly: ignore[bad-argument-type]
        echo=False,
        pad_output=True,
        **kwargs,
    )
    # The legacy last-output accessor is not a per-request identity source.
    self.output = output

    return base_rollout.RolloutOutput(
        text=output.text,
        logits=None,
        tokens=output.tokens,  # pyrefly: ignore[bad-argument-type]
        left_padded_prompt_tokens=output.padded_prompt_tokens,
        logprobs=output.logprobs,  # pyrefly: ignore[bad-argument-type]
        routed_experts=output.routed_experts,
        prompt_lengths=output.prompt_lengths,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    if self.output.logprobs is None:
      return jax.numpy.empty((0,))
    return jax.numpy.array(self.output.logprobs)

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    self._sampler.update_params(
        params,
        filter_types,
        preserve_active_kv_cache=self._keep_kv_across_sync,
    )

  def supports_partial_rollout_sync(self) -> bool:
    """Whether `RolloutConfig.partial_rollout` is set."""
    return self._partial_rollout

  def pause_for_weight_sync(self) -> None:
    """Freezes in-flight requests at a token boundary for a weight sync.

    Requests submitted while paused queue in the driver. With the "recompute"
    policy the frozen requests are preempted, so they re-prefill on the new
    weights after `resume_after_weight_sync`; "keep" leaves their KV resident.

    Raises:
      RuntimeError: If `partial_rollout` is not set, if the async batch queue
        fails to drain, or if the "recompute" policy cannot clear caches.
      TimeoutError: If pausing the sampler times out.
    """
    if not self._partial_rollout:
      raise RuntimeError("pause_for_weight_sync requires partial_rollout.")
    self._sampler.pause("keep", clear_cache=not self._keep_kv_across_sync)

  def resume_after_weight_sync(self) -> None:
    """Resumes the requests frozen by `pause_for_weight_sync`."""
    self._sampler.resume()

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer


def _check_partial_rollout_config(
    rollout_config: base_rollout.RolloutConfig,
) -> None:
  """Rejects partial-rollout settings that the vLLM engine cannot run.

  Args:
    rollout_config: The rollout config, with `partial_rollout` set.

  Raises:
    ValueError: If the combination is unsupported.
  """
  if not rollout_config.rollout_vllm_server_mode:
    raise ValueError(
        "partial_rollout needs rollout_vllm_server_mode=True: only the"
        " in-process driver can freeze in-flight requests."
    )
  if rollout_config.partial_rollout_kv_policy == "keep":
    logging.warning(
        "partial_rollout_kv_policy='keep': the KV cache stays allocated"
        " during weight sync, so the sync's second copy of the weights must"
        " fit next to it (rollout_vllm_free_kv_cache_during_weight_sync is"
        " bypassed). Lower rollout_vllm_hbm_utilization if the sync OOMs."
    )
  elif rollout_config.rollout_vllm_async_scheduling:
    # tpu-inference's async-scheduling token substitution assumes that a
    # request of the previous step is a 1-token decode; a request preempted
    # by the pause and re-prefilled in chunks violates that.
    raise ValueError(
        "partial_rollout_kv_policy='recompute' is not supported together"
        " with vLLM async scheduling: the re-prefill of preempted requests"
        " trips its async token substitution. Use"
        " partial_rollout_kv_policy='keep', or set"
        " rollout_vllm_async_scheduling=False."
    )

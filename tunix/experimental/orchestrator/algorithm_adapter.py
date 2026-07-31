# Copyright 2026 Google LLC
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

"""Algorithm-specific hooks for the RL orchestrator (the pluggable seam).

`RLOrchestrator` (the primitive API a learner loop is built on) is deliberately
algorithm-agnostic: generation, training, weight sync, and scoring are the same
regardless of algorithm. The pieces that genuinely differ between algorithms --
how rewards become advantages, how a group is assembled into a train example, and
how the trainer's loss is wired -- live behind this adapter, so one orchestrator
and one loop can serve GRPO, PPO, and future algorithms by swapping the adapter.

Each hook reuses the shared implementations (advantage-estimator / policy-loss
registries, padding helpers) rather than reimplementing them, so the orchestrator
stack stays numerically identical to the agentic learner.
"""

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from tunix.rl import function_registry
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl import rl_cluster as rl_cluster_lib


@runtime_checkable
class AlgorithmAdapter(Protocol):
  """The algorithm-specific bits an otherwise-generic RL loop needs."""

  def compute_advantages(self, rewards: Any, *, num_generations: int) -> Any:
    """Turns per-completion rewards into per-completion advantages."""
    ...

  def assemble_train_example(
      self,
      prompt_token_lists: Any,
      completion_token_lists: Any,
      advantages: Any,
      *,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
  ) -> Any:
    """Pads/masks a group of (prompt, completion) pairs into a train example."""
    ...

  def configure_trainer(self, cluster: Any) -> None:
    """Wires the algorithm's loss + model-input adapter onto the cluster trainer."""
    ...

  def postprocess_group(
      self,
      orchestrator: Any,
      trajectories: Any,
      *,
      compute_rewards: Any,
      mode: Any,
      expected_step: int | None = None,
  ) -> Any:
    """Turns a group of raw trajectories into train example(s).

    The algorithm's whole postprocess, expressed on the orchestrator primitives
    (scoring, metrics) + the shared registries -- the new-API form of the agentic
    `_process_results`.
    """
    ...


def pad_row(
    prompt_tokens: Any,
    completion_tokens: Any,
    completion_mask: Any,
    *,
    max_prompt_length: int,
    max_response_length: int,
    pad_id: int,
) -> tuple[Any, Any, Any]:
  """Pads one (prompt, completion, mask) row to the fixed training widths.

  The single place this padding is expressed on the orchestrated path: the
  prompt is left-padded, the completion and its loss mask are right-padded,
  and both are clipped to `max_response_length`. Group postprocess and
  single-turn assembly share it so their layouts cannot drift apart.
  """
  padded_prompt, padded_completion, _ = (
      agentic_utils.pad_prompt_and_completion(
          prompt_tokens,
          completion_tokens,
          max_prompt_length,
          max_response_length,
          pad_id,
      )
  )
  padded_mask = agentic_utils.right_pad(
      completion_mask, max_response_length, 0
  )[:max_response_length]
  return padded_prompt, padded_completion[:max_response_length], padded_mask


class UnsupportedConfigError(ValueError):
  """A configured behavior this adapter does not implement.

  Raised instead of running, because every case it covers would otherwise
  train with different math than the agentic learner the adapter mirrors, and
  produce a healthy-looking run while doing so.
  """


class GRPOAdapter:
  """GRPO hooks, reusing the shared advantage/loss registries and padding.

  Holds the `GRPOConfig` (algorithm knobs) and dispatches to the same registry
  functions the agentic GRPO learner uses -- no reimplementation of the group
  math, the loss, or the padding.

  This is a subset of the agentic learner's postprocess: it implements the
  on-policy path. Configurations it does not implement are rejected up front
  (see `check_supported_config`) rather than silently ignored.
  """

  def __init__(self, algo_config: Any):
    self._algo_config = algo_config
    self._check_supported_algo_config()

  @property
  def algo_config(self) -> Any:
    return self._algo_config

  def _check_supported_algo_config(self) -> None:
    """Rejects algorithm knobs this adapter would otherwise ignore."""
    algo = self._algo_config

    if getattr(algo, "sampler_is", None) is not None:
      raise UnsupportedConfigError(
          "sampler_is="
          f"{algo.sampler_is!r} requests sampler importance-sampling"
          " correction, which this adapter does not compute: it never fills"
          " sampler_is_weights, so the policy loss would silently skip the"
          " correction and train on uncorrected ratios. Unset sampler_is or"
          " use the agentic GRPO learner."
      )

    # Redundant with the missing-old-logps check that belongs on the scoring
    # path, but cheap and catches the config before a step runs.
    if getattr(algo, "num_iterations", 1) > 1:
      raise UnsupportedConfigError(
          f"num_iterations={algo.num_iterations} trains multiple times per"
          " batch, which requires old per-token logprobs to anchor the ratio"
          " on every iteration after the first. This adapter does not"
          " guarantee they are present. Use num_iterations=1 or the agentic"
          " GRPO learner."
      )

  def check_supported_config(self, cluster: Any) -> None:
    """Rejects cluster-level configuration this adapter does not implement.

    Args:
      cluster: The cluster (or orchestrator) whose `cluster_config` is read.

    Raises:
      UnsupportedConfigError: If sequence packing is enabled.
    """
    self._check_supported_algo_config()

    training_config = getattr(
        getattr(cluster, "cluster_config", None), "training_config", None
    )
    token_budget = getattr(training_config, "max_seq_token_per_tpu", None)
    if token_budget is not None:
      raise UnsupportedConfigError(
          f"max_seq_token_per_tpu={token_budget} enables sequence packing."
          " The agentic GRPO learner defers scoring until after packing and"
          " scores the packed buffer; this adapter scores the unpacked rows"
          " eagerly, so training and scoring would see different layouts."
          " Disable packing or use the agentic GRPO learner."
      )

  def compute_advantages(self, rewards: Any, *, num_generations: int) -> Any:
    estimator = function_registry.get_advantage_estimator(
        self._algo_config.advantage_estimator
    )
    return estimator(rewards=rewards, num_generations=num_generations)

  def assemble_train_example(
      self,
      prompt_token_lists: Any,
      completion_token_lists: Any,
      advantages: Any,
      *,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
  ) -> agentic_rl_learner.TrainExample:
    """Single-turn assembly: left-pad prompts, right-pad completions, build masks.

    Mirrors the padding/masking `_process_results` performs, for the degenerate
    single-completion case (no multi-turn conversation). ref/old per-token logps
    are left None (suitable for on-policy, beta=0 training); a KL/off-policy loop
    fills them via the scoring primitives before calling this.
    """
    padded_prompts = []
    padded_completions = []
    completion_masks = []
    for prompt_tokens, completion_tokens in zip(
        prompt_token_lists, completion_token_lists
    ):
      prompt_tokens = [int(t) for t in prompt_tokens]
      completion_tokens = [int(t) for t in completion_tokens]
      # Every generated token contributes: there is no environment span to
      # exclude in the single-turn case.
      left_prompt, right_completion, mask = pad_row(
          prompt_tokens,
          completion_tokens,
          [1] * len(completion_tokens),
          max_prompt_length=max_prompt_length,
          max_response_length=max_response_length,
          pad_id=pad_id,
      )
      padded_prompts.append(left_prompt)
      padded_completions.append(right_completion)
      completion_masks.append(mask)

    prompt_ids = jnp.asarray(np.stack(padded_prompts))
    completion_ids = jnp.asarray(np.stack(padded_completions))
    completion_mask = jnp.asarray(np.stack(completion_masks))
    return agentic_rl_learner.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=(prompt_ids != pad_id),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=jnp.asarray(advantages),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

  def configure_trainer(self, cluster: Any) -> None:
    """Wires GRPO's policy loss + model-input adapter onto the actor trainer.

    Reuses the same policy-loss registry entry the agentic GRPO learner uses.
    """
    self.check_supported_config(cluster)
    self._algo_config.temperature = cluster.get_rollout_config(
        mode=rl_cluster_lib.Mode.TRAIN
    ).temperature
    policy_loss_fn = function_registry.get_policy_loss_fn(
        self._algo_config.policy_loss_fn
    )
    algo_config = self._algo_config

    def loss_fn(model, train_example, algo_config=algo_config):
      return policy_loss_fn(
          model,
          train_example,
          algo_config=algo_config,
          pad_id=cluster.rollout.pad_id(),
          eos_id=cluster.rollout.eos_id(),
          compute_logps_chunk_size=cluster.cluster_config.training_config.compute_logps_chunk_size,
      )

    cluster.actor_trainer.with_loss_fn(loss_fn, has_aux=True)
    cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {"train_example": x, "algo_config": algo_config}
    )
    cluster.actor_trainer.is_managed_externally = True

  def postprocess_group(
      self,
      orchestrator: Any,
      trajectories: Any,
      *,
      compute_rewards: Any,
      mode: Any,
      expected_step: int | None = None,
  ) -> list[agentic_rl_learner.TrainExample]:
    """GRPO postprocess re-expressed on the orchestrator primitives.

    The new-API form of `GRPOLearner._process_results` for the on-policy path:
    extract -> pad/mask -> score (reference/actor via orchestrator primitives) ->
    reward (caller-supplied) -> advantage (shared estimator) -> assemble. Reuses
    the same padding helpers and registries, so it stays faithful to the agentic
    learner.     (The extensive diagnostic metrics and the sampler-IS/off-policy paths
    of `_process_results` are omitted here; those layer on next.)
    """
    self.check_supported_config(orchestrator)
    algo = self._algo_config
    pad_value = orchestrator.rollout.pad_id()
    eos_value = orchestrator.rollout.eos_id()
    rollout_config = orchestrator.get_rollout_config(mode)
    max_prompt_length = rollout_config.max_prompt_length
    max_response_length = algo.max_response_length
    # The configured size counts groups, not rows: a group contributes
    # num_generations rows. Scoring with the raw value would compile a
    # different shape than the agentic learner for identical math.
    configured_micro_batch_size = (
        orchestrator.cluster_config.training_config.compute_logps_micro_batch_size
    )
    micro_batch_size = (
        configured_micro_batch_size * algo.num_generations
        if configured_micro_batch_size
        else len(trajectories)
    )

    completion_texts = []
    prompt_tokens_list = []
    completion_tokens_list = []
    completion_masks_list = []
    old_logprobs_list = []
    policy_versions_list = []
    trajectory_rewards_list = []
    original_inputs_list = []
    for item in trajectories:
      traj = item.traj
      conversation = traj.get("conversation_text") or []
      assistant_text = next(
          (m["content"] for m in conversation if m["role"] == "assistant"), ""
      )
      completion_texts.append(assistant_text)
      prompt_tokens_list.append(traj.get("prompt_tokens"))
      completion_tokens_list.append(traj.get("conversation_tokens"))
      completion_masks_list.append(traj.get("conversation_masks"))
      old_logprobs_list.append(traj.get("old_logprobs"))
      policy_version = traj.get("policy_version")
      if policy_version is None:
        raise ValueError("policy_version is missing from trajectory task.")
      policy_versions_list.append(policy_version)
      trajectory_rewards_list.append(traj.get("trajectory_reward"))
      original_inputs_list.append(traj["original_input"])

    padded_prompt_ids = []
    padded_completion_ids = []
    padded_completion_masks = []
    padded_old_logprobs = []
    for prompt_tokens, completion_tokens, completion_mask, old_logprobs in zip(
        prompt_tokens_list,
        completion_tokens_list,
        completion_masks_list,
        old_logprobs_list,
    ):
      padded_prompt, padded_completion, padded_mask = pad_row(
          prompt_tokens,
          completion_tokens,
          completion_mask,
          max_prompt_length=max_prompt_length,
          max_response_length=max_response_length,
          pad_id=pad_value,
      )
      padded_prompt_ids.append(padded_prompt)
      padded_completion_ids.append(padded_completion)
      padded_completion_masks.append(padded_mask)
      if algo.use_rollout_logps:
        if old_logprobs is not None:
          padded_old_logprobs.append(
              agentic_utils.right_pad(
                  old_logprobs,
                  length=max_response_length,
                  pad=0.0,
                  dtype=old_logprobs.dtype,
              )[:max_response_length]
          )
        else:
          padded_old_logprobs.append(
              np.zeros(max_response_length, dtype=np.float32)
          )

    prompt_ids = jnp.asarray(padded_prompt_ids)
    prompt_mask = prompt_ids != pad_value
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)

    if algo.use_rollout_logps and padded_old_logprobs:
      old_per_token_logps = jnp.asarray(padded_old_logprobs)
    elif algo.use_rollout_logps:
      old_per_token_logps = None
    else:
      old_per_token_logps = orchestrator.actor_logps(
          prompt_ids, completion_ids, pad_value, eos_value, micro_batch_size
      )

    if algo.force_compute_kl or algo.beta != 0.0:
      ref_per_token_logps = orchestrator.reference_logps(
          prompt_ids, completion_ids, pad_value, eos_value, micro_batch_size
      )
    else:
      ref_per_token_logps = None

    original_inputs = rl_utils.merge_micro_batches(original_inputs_list)
    reward_kwargs = {
        key: value for key, value in original_inputs.items() if key != "prompts"
    }
    reward_kwargs["trajectory_rewards"] = trajectory_rewards_list
    rewards = compute_rewards(
        prompts=original_inputs["prompts"],
        completions=completion_texts,
        mode=mode,
        expected_step=expected_step,
        **reward_kwargs,
    )
    advantages = self.compute_advantages(
        rewards, num_generations=algo.num_generations
    )

    orchestrator.buffer_metrics_async(
        {
            "rewards/advantage/mean": (np.mean(advantages), np.mean),
            "rewards/advantage/max": (np.max(advantages), np.max),
            "rewards/advantage/min": (np.min(advantages), np.min),
        },
        mode=mode,
        step=expected_step,
    )

    return [
        agentic_rl_learner.TrainExample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            old_per_token_logps=old_per_token_logps,
            policy_version=np.array(policy_versions_list, dtype=np.int32),
        )
    ]

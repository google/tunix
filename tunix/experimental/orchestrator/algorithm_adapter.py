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
from tunix.experimental.orchestrator import loss_spec as loss_spec_lib
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


class GRPOAdapter:
  """GRPO hooks, reusing the shared advantage/loss registries and padding.

  Holds the `GRPOConfig` (algorithm knobs) and dispatches to the same registry
  functions the agentic GRPO learner uses -- no reimplementation of the group
  math, the loss, or the padding.
  """

  def __init__(self, algo_config: Any):
    self._algo_config = algo_config

  @property
  def algo_config(self) -> Any:
    return self._algo_config

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
      left_prompt, right_completion, _ = agentic_utils.pad_prompt_and_completion(
          prompt_tokens,
          completion_tokens,
          max_prompt_length,
          max_response_length,
          pad_id,
      )
      padded_prompts.append(left_prompt)
      padded_completions.append(right_completion[:max_response_length])
      real_len = min(len(completion_tokens), max_response_length)
      mask = agentic_utils.right_pad(
          [1] * real_len, max_response_length, 0
      )[:max_response_length]
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

  def loss_spec(self, cluster: Any) -> loss_spec_lib.LossSpec:
    """Describes the loss, so it can be built wherever the trainer runs.

    Args:
      cluster: Supplies the padding ids and chunk size the loss needs.

    Returns:
      A description carrying only data, no captured functions.
    """
    return loss_spec_lib.LossSpec(
        policy_loss_fn=self._algo_config.policy_loss_fn,
        algo_config=self._algo_config,
        pad_id=cluster.rollout.pad_id(),
        eos_id=cluster.rollout.eos_id(),
        compute_logps_chunk_size=(
            cluster.cluster_config.training_config.compute_logps_chunk_size
        ),
    )

  def configure_trainer(self, cluster: Any) -> None:
    """Wires GRPO's policy loss + model-input adapter onto the actor trainer.

    Reuses the same policy-loss registry entry the agentic GRPO learner uses.
    The wiring is expressed as a description rather than built here, so a
    trainer in another process can construct the same loss from it; a cluster
    that knows how to deliver one is given it, and otherwise the loss is built
    and installed locally exactly as before.
    """
    self._algo_config.temperature = cluster.get_rollout_config(
        mode=rl_cluster_lib.Mode.TRAIN
    ).temperature

    spec = self.loss_spec(cluster)
    deliver = getattr(cluster, "configure_loss", None)
    if callable(deliver):
      deliver(spec)
    else:
      spec.install_on(cluster.actor_trainer)

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
    learner. (The extensive diagnostic metrics and the sampler-IS/off-policy paths
    of `_process_results` are omitted here; those layer on next.)
    """
    algo = self._algo_config
    pad_value = orchestrator.rollout.pad_id()
    eos_value = orchestrator.rollout.eos_id()
    rollout_config = orchestrator.get_rollout_config(mode)
    max_prompt_length = rollout_config.max_prompt_length
    max_response_length = algo.max_response_length
    micro_batch_size = (
        orchestrator.cluster_config.training_config.compute_logps_micro_batch_size
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
      padded_prompt, padded_completion, _ = (
          agentic_utils.pad_prompt_and_completion(
              prompt_tokens,
              completion_tokens,
              max_prompt_length,
              max_response_length,
              pad_value,
          )
      )
      padded_prompt_ids.append(padded_prompt)
      padded_completion_ids.append(padded_completion[:max_response_length])
      padded_completion_masks.append(
          agentic_utils.right_pad(completion_mask, max_response_length, 0)[
              :max_response_length
          ]
      )
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

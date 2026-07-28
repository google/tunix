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

  def configure_trainer(self, cluster: Any) -> None:
    """Wires GRPO's policy loss + model-input adapter onto the actor trainer.

    Reuses the same policy-loss registry entry the agentic GRPO learner uses.
    """
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

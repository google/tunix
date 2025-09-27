"""Adapter to convert collected trajectories to GRPO TrainExample format."""

from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl.grpo import grpo_helpers


class GRPOTrajectoryAdapter:
  """Adapter: convert collected trajectories from TrajectoryCollectEngine.

  into GRPO-compatible TrainExample format.
  """

  def __init__(self, pad_id: int, eos_id: int, num_generations: int):
    self.pad_id = pad_id
    self.eos_id = eos_id
    self.num_generations = num_generations

  def to_train_example(
      self,
      trajectories: List[
          Dict[str, Any]
      ],  # collected via .collect(mode="Token")
      prompts: List[str],
      rl_cluster,
  ) -> common.TrainExample:
    """Convert raw trajectories into TrainExample for GRPO.

    Args:
        trajectories: List of dicts from trajectory.collect(mode="Token").
        prompts: Original prompt strings (for reward fn).
        rl_cluster: RL cluster (provides ref logps, old logps, etc).

    Returns:
        TrainExample instance.
    """
    prompt_ids_list = []
    completion_ids_list = []
    prompt_masks = []
    completion_masks = []
    rewards = []

    for traj in trajectories:
      prompt_tokens = traj.get("prompt_tokens", [])
      response_tokens = traj.get("response_tokens", [])
      response_masks = traj.get("response_masks", [])
      traj_reward = traj.get("trajectory_reward", 0.0)

      # Convert to numpy arrays
      prompt_ids = np.array(prompt_tokens, dtype=np.int32)
      completion_ids = np.array(response_tokens, dtype=np.int32)

      # Create masks
      prompt_mask = (prompt_ids != self.pad_id).astype(np.int32)
      completion_mask = np.array(response_masks, dtype=np.int32)
      # truncate/pad if needed
      if len(completion_mask) < len(completion_ids):
        pad_len = len(completion_ids) - len(completion_mask)
        completion_mask = np.concatenate(
            [completion_mask, np.zeros(pad_len, dtype=np.int32)]
        )

      # Append
      prompt_ids_list.append(prompt_ids)
      completion_ids_list.append(completion_ids)
      prompt_masks.append(prompt_mask)
      completion_masks.append(completion_mask)
      rewards.append(traj_reward)

    # Pad to equal length across batch
    prompt_ids = self._pad_to_max_len(prompt_ids_list, self.pad_id)
    completion_ids = self._pad_to_max_len(completion_ids_list, self.pad_id)
    prompt_mask = self._pad_to_max_len(prompt_masks, 0)
    completion_mask = self._pad_to_max_len(completion_masks, 0)
    rewards = jnp.array(rewards)

    # Compute advantages (GRPO group relative)
    advantages = grpo_helpers.compute_advantages(rewards, self.num_generations)

    # Reference logps if KL penalty is used
    ref_per_token_logps = rl_cluster.get_ref_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
    )

    # Old logps if multiple iterations are used
    old_per_token_logps = rl_cluster.get_old_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
    )

    # Wrap into TrainExample
    return common.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  def _pad_to_max_len(
      self, sequences: List[np.ndarray], pad_value: int
  ) -> jnp.ndarray:
    """Pad a list of sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    padded = [
        np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value)
        for seq in sequences
    ]
    return jnp.array(padded)

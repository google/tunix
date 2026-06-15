# Copyright 2025 Google LLC
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

"""Sub-batch checkpoint manager for pipeline resilience."""

import collections.abc
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options

Trajectory = agent_types.Trajectory
TrajectoryItem = agent_types.TrajectoryItem
TrajectoryStatus = agent_types.TrajectoryStatus
Step = agent_types.Step
Hashable = collections.abc.Hashable

SUBSTEP_MULTIPLIER = 1_000_000


def _encode_step(global_step: int, grad_accum_steps: int) -> int:
  """Encodes (global_step, grad_accum_steps) into a monotonic integer."""
  return global_step * SUBSTEP_MULTIPLIER + grad_accum_steps


def _decode_step(encoded_step: int) -> tuple[int, int]:
  """Decodes a monotonic integer into (global_step, grad_accum_steps)."""
  return divmod(encoded_step, SUBSTEP_MULTIPLIER)


def _trajectory_item_to_serializable(item: TrajectoryItem) -> dict[str, Any]:
  """Converts a `TrajectoryItem` to a serializable dictionary."""
  traj_dict = (
      item.traj.to_dict() if isinstance(item.traj, Trajectory) else item.traj
  )

  # Convert numpy arrays in steps to lists
  serializable_steps = []
  for step in traj_dict.get("steps", []):
    s = dict(step)
    for key in (
        "assistant_tokens",
        "assistant_masks",
        "env_tokens",
        "env_msks",
        "logprobs",
    ):
      if s.get(key) is not None:
        s[key] = s[key].tolist()
    serializable_steps.append(s)
  traj_dict["steps"] = serializable_steps

  return {
      "pair_index": item.pair_index,
      "group_id": item.group_id,
      "start_step": item.start_step,
      "traj": traj_dict,
      "metadata": item.metadata,
  }


def _trajectory_item_from_serializable(
    data: dict[str, Any],
) -> TrajectoryItem:
  """Converts a serializable dictionary to a `TrajectoryItem`."""
  traj_dict = data["traj"]

  steps = []
  for step_data in traj_dict.get("steps", []):
    s = dict(step_data)
    for key in (
        "assistant_tokens",
        "assistant_masks",
        "env_tokens",
        "env_msks",
        "logprobs",
    ):
      if s.get(key) is not None:
        s[key] = np.array(s[key])

    if s.get("action") is not None and isinstance(s["action"], dict):
      s["action"] = agent_types.Action(**s["action"])

    steps.append(
        Step(**{
            k: v
            for k, v in s.items()
            if k in {f.name for f in dataclasses.fields(Step)}
        })
    )

  status_name = traj_dict.get("status", "RUNNING")
  try:
    status = TrajectoryStatus[status_name]
  except KeyError:
    status = TrajectoryStatus.RUNNING

  traj = Trajectory(
      task=traj_dict.get("task"),
      steps=steps,
      reward=traj_dict.get("reward", 0.0),
      status=status,
      env_time=traj_dict.get("env_time", {}),
      reward_time=traj_dict.get("reward_time", {}),
  )

  return TrajectoryItem(
      pair_index=data["pair_index"],
      group_id=data["group_id"],
      start_step=data["start_step"],
      traj=traj,
      metadata=data.get("metadata", {}),
  )


def _extract_multisteps_diff(opt_state: Any) -> Any:
  """Extracts (acc_grads, mini_step) from an optax.MultiStepsState PyTree."""
  if isinstance(opt_state, optax.MultiStepsState):
    return {
        "acc_grads": opt_state.acc_grads,
        "mini_step": jnp.atleast_1d(opt_state.mini_step),
    }
  if hasattr(opt_state, "__dict__"):
    for v in opt_state.__dict__.values():
      res = _extract_multisteps_diff(v)
      if res is not None:
        return res
  if isinstance(opt_state, (tuple, list)):
    for v in opt_state:
      res = _extract_multisteps_diff(v)
      if res is not None:
        return res
  if isinstance(opt_state, dict):
    for v in opt_state.values():
      res = _extract_multisteps_diff(v)
      if res is not None:
        return res
  return None


def _inject_multisteps_diff(
    opt_state: Any, diff: Any, grad_accum_steps: int
) -> Any:
  """Injects acc_grads and mini_step back into optax.MultiStepsState tree."""
  if isinstance(opt_state, optax.MultiStepsState):
    acc_grads = diff["acc_grads"] if isinstance(diff, dict) else diff.acc_grads
    mini_step = diff["mini_step"] if isinstance(diff, dict) else diff.mini_step
    if hasattr(mini_step, "ndim") and mini_step.ndim == 1:
      mini_step = mini_step[0]
    return optax.MultiStepsState(
        mini_step=mini_step,
        gradient_step=opt_state.gradient_step,
        inner_opt_state=opt_state.inner_opt_state,
        acc_grads=acc_grads,
        skip_state=opt_state.skip_state,
    )
  if hasattr(opt_state, "__dict__"):
    for k, v in opt_state.__dict__.items():
      res = _inject_multisteps_diff(v, diff, grad_accum_steps)
      if res is not None:
        setattr(opt_state, k, res)
    return opt_state
  if isinstance(opt_state, (tuple, list)):
    new_values = []
    modified = False
    for v in opt_state:
      res = _inject_multisteps_diff(v, diff, grad_accum_steps)
      if res is not None and res is not v:
        new_values.append(res)
        modified = True
      else:
        new_values.append(v)
    if modified:
      if isinstance(opt_state, tuple) and hasattr(opt_state, "_fields"):
        return type(opt_state)(*new_values)  # type: ignore
      return type(opt_state)(new_values)
    return opt_state
  if isinstance(opt_state, dict):
    for k, v in opt_state.items():
      res = _inject_multisteps_diff(v, diff, grad_accum_steps)
      if res is not None and res is not v:
        opt_state[k] = res
    return opt_state
  return opt_state


@dataclasses.dataclass
class SubBatchState:
  """State returned by :meth:`SubBatchCheckpointManager.try_restore`.

  Attributes:
    global_step: Global training step this checkpoint belongs to.
    grad_accum_steps: Explicitly represents the monotonic micro-batch
      accumulation counter (micro_batches_since_last_sync).
    completed_group_ids: Tracks prompt groups that have been 100% fully
      generated and consumed.
    trained_trajectory_counts: Map tracking (group_id, pair_index) -> consumed
      count.
    active_group_trajectories: List of incomplete groups or full groups that
      have not yet been fully consumed by the trainer.
    training_state: PyTree representing the optimizer or accumulator state.
      can either be a pair of (acc_grads, mini_step) or directly a gradient
      accumulator's grads.
    valid_token_count: Optional token count or normalization scalar.
  """

  global_step: int
  grad_accum_steps: int
  completed_group_ids: list[Hashable]
  trained_trajectory_counts: dict[tuple[Hashable, int], int]
  active_group_trajectories: list[TrajectoryItem]
  training_state: Any
  valid_token_count: jax.Array | None = None


@dataclasses.dataclass
class GlobalStepPreservationPolicy(
    ocp.training.preservation_policies.PreservationPolicy
):
  """Policy which keeps the latest N checkpoints for the active global step and purges older global steps."""

  latest_n: int = 3

  def should_preserve(
      self,
      checkpoints: Sequence[ocp.training.CheckpointMetadata],
      *,
      context: ocp.training.preservation_policies.PreservationContext,
  ) -> Sequence[bool]:
    if not checkpoints:
      return []

    latest_step = max(ckpt.step for ckpt in checkpoints)
    last_global_step = _decode_step(latest_step)[0]
    active_ckpts = [
        ckpt
        for ckpt in checkpoints
        if _decode_step(ckpt.step)[0] == last_global_step
    ]
    min_active_step = min(
        ckpt.step for ckpt in active_ckpts[-self.latest_n :]
    )
    return [
        _decode_step(ckpt.step)[0] == last_global_step
        and ckpt.step >= min_active_step
        for ckpt in checkpoints
    ]


# Default checkpointing options for Tunix:
# - Save at every sub-batch.
# - Use custom preservation policy to remove completed global steps and keep the
#   latest N sub-step checkpoints for active global steps.
# - Use simple integer step names.
# - Use async checkpointing.
# - Timeout for async operations is 1200 seconds.
DEFAULT_SUB_BATCH_CHECKPOINTING_OPTIONS = (
    checkpoint_options.TunixCheckpointingOptions(
        save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
            interval=1
        ),
        preservation_policy=GlobalStepPreservationPolicy(latest_n=3),
        step_name_format=ocp.path.step.standard_name_format(),
        enable_async_checkpointing=True,
        async_options=ocp.options.AsyncOptions(timeout_secs=1200),
    )
)


class SubBatchCheckpointManager(checkpoint_manager.BaseCheckpointManager):
  """Sub-batch checkpoint manager for pipeline resilience."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: checkpoint_options.CheckpointingOptions | None = None,
  ):
    if options is None:
      options = DEFAULT_SUB_BATCH_CHECKPOINTING_OPTIONS
    super().__init__(root_directory=root_directory, options=options)

  def save(
      self,
      global_step: int,
      grad_accum_steps: int,
      completed_group_ids: list[Hashable],
      trained_trajectory_counts: dict[tuple[Hashable, int], int],
      active_group_trajectories: list[TrajectoryItem],
      training_state: Any,
      valid_token_count: jax.Array | None = None,
  ) -> None:
    """Persists sub-batch checkpoint."""
    if self._checkpointer is None:
      return
    step = _encode_step(global_step, grad_accum_steps)

    serialized_incomplete = [
        _trajectory_item_to_serializable(item)
        for item in active_group_trajectories
    ]
    serialized_counts = [
        {"group_id": k[0], "pair_index": k[1], "count": v}
        for k, v in trained_trajectory_counts.items()
    ]

    state_dict = {
        "training_state": training_state,
    }
    if valid_token_count is not None:
      state_dict["valid_token_count"] = valid_token_count

    checkpointables = {
        "meta": {
            "global_step": global_step,
            "grad_accum_steps": grad_accum_steps,
        },
        "rollout": {
            "completed_group_ids": list(completed_group_ids),
            "active_group_trajectories": serialized_incomplete,
            "trained_trajectory_counts": serialized_counts,
        },
        "state": state_dict,
    }

    with self._context:
      self._save_checkpointables(
          step,
          checkpointables=checkpointables,
          force=True,
          custom_metadata=None,
      )

  def try_restore(
      self,
      global_step: int,
      target_training_state: Any = None,
      target_valid_token_count: jax.Array | None = None,
  ) -> SubBatchState | None:
    """Fast-forwards both trajectory queueing and gradient accumulation."""
    if self._checkpointer is None:
      return None

    self._checkpointer.wait()

    latest_step = self.latest_step()
    if latest_step is None:
      return None

    last_global_step = _decode_step(latest_step)[0]
    if last_global_step != global_step:
      return None

    abstract_checkpointables = {"meta": None, "rollout": None, "state": None}
    if target_training_state is not None:
      abstract_state = {
          "training_state": target_training_state,
      }
      if target_valid_token_count is not None:
        abstract_state["valid_token_count"] = target_valid_token_count
      abstract_checkpointables["state"] = abstract_state

    restored = self._checkpointer.load_checkpointables(
        latest_step,
        abstract_checkpointables=abstract_checkpointables,
    )

    meta = restored["meta"]
    rollout = restored["rollout"]
    state = restored["state"]

    # Deserialize incomplete rollouts
    active_group_trajectories = [
        _trajectory_item_from_serializable(item)
        for item in rollout["active_group_trajectories"]
    ]

    # Deserialize seen_rollout_counts
    trained_trajectory_counts = {
        (item["group_id"], item["pair_index"]): item["count"]
        for item in rollout["trained_trajectory_counts"]
    }

    valid_token_count = state.get("valid_token_count")

    return SubBatchState(
        global_step=meta["global_step"],
        grad_accum_steps=meta["grad_accum_steps"],
        completed_group_ids=rollout["completed_group_ids"],
        trained_trajectory_counts=trained_trajectory_counts,
        active_group_trajectories=active_group_trajectories,
        training_state=state["training_state"],
        valid_token_count=valid_token_count,
    )

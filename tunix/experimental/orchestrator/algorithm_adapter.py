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

"""Layer 2B: AlgorithmAdapter Math & Loss Wiring (algorithm_adapter.py).

Encapsulates RL returns, GAE / GRPO advantages, loss functions, and
RLTrainerPayload assembly matching Orchestrator V2 and delegating loss
computations directly to `tunix.rl.algo_core`.
"""

import abc
from collections.abc import Callable, Sequence
import functools
import types
from typing import Any

import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.rl import algo_core
from tunix.rl import function_registry


def _algo_model_input(
    train_example: Any,
    *,
    algo_config: Any,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
  """Maps an RLTrainerPayload microbatch to algorithm loss kwargs (e.g.

  GRPO/PPO).
  """
  return {
      "train_example": train_example,
      "algo_config": algo_config,
      "pad_id": pad_id,
      "eos_id": eos_id,
  }


def _extract_tokens_and_masks(
    item: datatypes.TrajectoryItem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Extracts prompt_tokens, conversation_tokens, and conversation_masks from TrajectoryItem."""
  return (
      np.asarray(item.traj["prompt_tokens"], dtype=np.int32).reshape(-1),
      np.asarray(item.traj["conversation_tokens"], dtype=np.int32).reshape(-1),
      np.asarray(item.traj["conversation_masks"], dtype=np.float32).reshape(-1),
  )


def _routed_experts_for(
    item: datatypes.TrajectoryItem, seq_len: int
) -> np.ndarray | None:
  """Aligns a rollout's captured routing to the payload's token sequence.

  Args:
    item: Trajectory item, whose `routed_experts` (if any) is `[captured_len,
      num_layers, top_k]`.
    seq_len: Length of the prompt+completion sequence in the payload.

  Returns:
    `[seq_len, num_layers, top_k]`, or None when nothing was captured. Rows
    beyond what the rollout reported are left `UNSET_ROUTED_EXPERT` so the
    model falls back to its own gate there rather than replaying a wrong
    expert.
  """
  routed = getattr(item, "routed_experts", None)
  if routed is None and isinstance(item.traj, dict):
    routed = item.traj.get("routed_experts")
  if routed is None:
    routed = item.metadata.get("routed_experts")
  if routed is None:
    return None
  routed_arr = np.asarray(routed, dtype=np.int32)
  if routed_arr.ndim != 3:
    raise ValueError(
        "routed_experts must be [length, num_layers, top_k]; got shape"
        f" {routed_arr.shape}"
    )
  if routed_arr.shape[0] >= seq_len:
    return routed_arr[:seq_len]
  pad = np.full(
      (seq_len - routed_arr.shape[0],) + routed_arr.shape[1:],
      datatypes.UNSET_ROUTED_EXPERT,
      dtype=np.int32,
  )
  return np.concatenate([routed_arr, pad], axis=0)


def _extract_old_logps(
    item: datatypes.TrajectoryItem, completion_len: int
) -> np.ndarray | None:
  """Extracts old_per_token_logps from TrajectoryItem."""
  old_lp = item.traj.get("old_logprobs")
  if old_lp is None:
    return None
  old_lp = np.asarray(old_lp, dtype=np.float32).reshape(-1)
  if len(old_lp) != completion_len:
    raise ValueError(
        f"old_logprobs length {len(old_lp)} does not match completion length"
        f" {completion_len}"
    )
  return old_lp



class AlgorithmAdapter(abc.ABC):
  """Abstract algorithm adapter for returns math, advantages, and loss functions."""

  def __init__(
      self,
      group_size: int = 8,
      mini_batch_size: int = 4,
      train_micro_batch_size: int = 1,
      max_turns: int = 1,
      max_packed_len: int = 8192,
      max_response_length: int = 1024,
  ):
    self.group_size = group_size
    self.mini_batch_size = mini_batch_size
    self.train_micro_batch_size = train_micro_batch_size
    self.max_turns = max_turns
    self.max_packed_len = max_packed_len
    self.max_response_length = max_response_length
    self.requires_reference_kl = False
    self.has_critic = False
    self.requires_old_logprobs = False
    self.use_rollout_logps: bool = True
    self.sampler_is: str | None = None
    self.sampler_is_threshold: float = 2.0

  @abc.abstractmethod
  def compute_advantages(
      self, rewards: np.ndarray | jnp.ndarray | Sequence[float], **kwargs: Any
  ) -> Any:
    """Computes returns and advantages from rewards."""
    ...

  @abc.abstractmethod
  def create_trainer_payloads(
      self,
      group: Any,
      rewards: Sequence[float],
      ref_logps: Any | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Assembles scored trajectories and computed advantages into typed RLTrainerPayloads."""
    ...

  @abc.abstractmethod
  def loss_fn(self) -> Callable[..., Any]:
    """Returns the loss function executed on TrainerWorker."""
    ...

  @abc.abstractmethod
  def build_gen_model_input_fn(
      self, pad_id: int, eos_id: int
  ) -> Callable[[Any], dict[str, Any]]:
    """Returns a model input generator function executed on TrainerWorker."""
    ...


class GRPOAdapter(AlgorithmAdapter):
  """Group Relative Policy Optimization (GRPO) adapter."""

  def __init__(
      self,
      group_size: int = 8,
      mini_batch_size: int = 4,
      train_micro_batch_size: int = 1,
      max_turns: int = 1,
      max_packed_len: int = 8192,
      max_response_length: int = 1024,
      clip_epsilon: float = 0.2,
      epsilon_high: float | None = None,
      beta_kl: float = 0.04,
      temperature: float = 1.0,
      loss_algo: str = "grpo",
      policy_loss_fn: str = "grpo",
      advantage_estimator: str = "grpo",
      loss_agg_mode: str = "sequence-mean-token-mean",
      kl_loss_mode: str = "mse_kl",
      kl_clamp_value: float | None = None,
      use_rollout_logps: bool = True,
      sampler_is: str | None = None,
      sampler_is_threshold: float = 2.0,
  ):
    if group_size <= 1:
      raise ValueError(
          f"group_size must be greater than 1 for GRPO. Received: {group_size}"
      )
    if sampler_is not in (None, "token"):
      raise ValueError(
          "sampler_is should be either None or 'token'. Received: "
          f"{sampler_is}"
      )
    super().__init__(
        group_size=group_size,
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        max_turns=max_turns,
        max_packed_len=max_packed_len,
        max_response_length=max_response_length,
    )
    self.clip_epsilon = clip_epsilon
    self.epsilon_high = epsilon_high if epsilon_high is not None else clip_epsilon
    self.loss_algo = loss_algo
    self.policy_loss_fn = policy_loss_fn
    self.advantage_estimator = advantage_estimator
    self.beta_kl = beta_kl
    self.temperature = temperature
    self.loss_agg_mode = loss_agg_mode
    self.kl_loss_mode = kl_loss_mode
    self.kl_clamp_value = kl_clamp_value
    self.requires_reference_kl = beta_kl != 0.0
    self.use_rollout_logps = use_rollout_logps
    self.sampler_is = sampler_is
    self.sampler_is_threshold = sampler_is_threshold

  def compute_advantages(
      self,
      rewards: np.ndarray | jnp.ndarray | Sequence[float],
      num_generations: int | None = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Computes returns and advantages using the registered advantage estimator."""
    del kwargs
    g = num_generations or self.group_size
    estimator = function_registry.get_advantage_estimator(
        self.advantage_estimator
    )
    r = np.asarray(rewards, dtype=np.float32).reshape(-1)
    return jnp.asarray(estimator(rewards=r, num_generations=g))

  def create_trainer_payloads(
      self,
      group: Sequence[datatypes.TrajectoryItem],
      rewards: Sequence[float],
      ref_logps: Any | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Packages group trajectories, advantages, and tool observation masks into unbatched RLTrainerPayloads."""
    del kwargs
    advs = self.compute_advantages(rewards, num_generations=self.group_size)
    payloads = []

    for i, item in enumerate(group):
      p_arr, c_arr, act_arr = _extract_tokens_and_masks(item)
      adv_val = float(advs[i]) if i < len(advs) else 0.0
      ref_lp = (
          ref_logps[i] if ref_logps is not None and i < len(ref_logps) else None
      )

      seq_tokens = (
          np.concatenate([p_arr, c_arr])
          if (len(p_arr) > 0 or len(c_arr) > 0)
          else np.zeros(0, dtype=np.int32)
      )
      seq_adv = np.full(len(c_arr), adv_val, dtype=np.float32)
      old_lp = (
          _extract_old_logps(item, len(c_arr))
          if self.use_rollout_logps
          else None
      )
      payload = datatypes.RLTrainerPayload(
          prompt_ids=p_arr,
          prompt_mask=np.ones(len(p_arr), dtype=np.float32),
          completion_ids=c_arr,
          completion_mask=act_arr,
          advantages=seq_adv,
          old_per_token_logps=old_lp,
          ref_per_token_logps=np.asarray(ref_lp, dtype=np.float32)
          if ref_lp is not None
          else None,
          routed_experts=_routed_experts_for(item, len(seq_tokens)),
      )
      payloads.append(payload)
    return payloads

  def loss_fn(self) -> Callable[..., Any]:
    """Policy loss resolved by name via the function registry."""
    return function_registry.get_policy_loss_fn(self.policy_loss_fn)

  def build_gen_model_input_fn(
      self, pad_id: int, eos_id: int
  ) -> Callable[[Any], dict[str, Any]]:
    """Returns a model input generator function for TrainerWorker."""
    algo_config = types.SimpleNamespace(
        beta=self.beta_kl,
        epsilon=self.clip_epsilon,
        epsilon_high=self.epsilon_high,
        loss_algo=self.loss_algo,
        loss_agg_mode=self.loss_agg_mode,
        temperature=self.temperature,
        kl_loss_mode=self.kl_loss_mode,
        kl_clamp_value=self.kl_clamp_value,
        use_rollout_logps=self.use_rollout_logps,
        sampler_is=self.sampler_is,
        sampler_is_threshold=self.sampler_is_threshold,
    )
    return functools.partial(
        _algo_model_input,
        algo_config=algo_config,
        pad_id=pad_id,
        eos_id=eos_id,
    )


class PPOAdapter(AlgorithmAdapter):
  """Generalized Advantage Estimation (GAE) and PPO Actor-Critic adapter."""

  def __init__(
      self,
      group_size: int = 1,
      mini_batch_size: int = 4,
      max_turns: int = 1,
      max_packed_len: int = 8192,
      max_response_length: int = 1024,
      gamma: float = 0.99,
      lam: float = 0.95,
      clip_epsilon: float = 0.2,
      entropy_coef: float = 0.0,
      policy_loss_fn: str = "ppo",
      use_rollout_logps: bool = True,
  ):
    super().__init__(
        group_size=group_size,
        mini_batch_size=mini_batch_size,
        max_turns=max_turns,
        max_packed_len=max_packed_len,
        max_response_length=max_response_length,
    )
    self.gamma = gamma
    self.policy_loss_fn = policy_loss_fn
    self.lam = lam
    self.clip_epsilon = clip_epsilon
    self.entropy_coef = entropy_coef
    self.has_critic = True
    self.requires_reference_kl = True
    self.requires_old_logprobs = True
    self.use_rollout_logps = use_rollout_logps

  def compute_advantages(
      self,
      rewards: np.ndarray | jnp.ndarray | Sequence[float],
      values: np.ndarray | jnp.ndarray | None = None,
      **kwargs: Any,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes GAE advantages and value function regression targets."""
    del kwargs
    r = jnp.asarray(rewards, dtype=jnp.float32)
    if values is None:
      values = jnp.zeros_like(r)
    else:
      values = jnp.asarray(values, dtype=jnp.float32)

    # 1-step / scalar GAE fallback for sequence-level rewards
    deltas = r - values
    gae_advantages = deltas
    value_targets = r
    return gae_advantages, value_targets

  def create_trainer_payloads(
      self,
      group: Any,
      rewards: Sequence[float],
      ref_logps: Any | None = None,
      values: Any | None = None,
      old_logps: Any | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Builds unbatched RLTrainerPayloads with GAE advantages, value targets, and old_logprobs."""
    del kwargs
    advs, val_targets = self.compute_advantages(rewards, values=values)
    payloads = []
    trajectories = getattr(group, "trajectories", None) or (
        group if isinstance(group, (list, tuple)) else [group]
    )

    for i, item in enumerate(trajectories):
      p_arr, c_arr, act_arr = _extract_tokens_and_masks(item)

      adv_val = float(advs[i]) if i < len(advs) else 0.0
      vt_val = float(val_targets[i]) if i < len(val_targets) else 0.0
      ref_lp = (
          ref_logps[i] if ref_logps is not None and i < len(ref_logps) else None
      )
      old_lp = (
          old_logps[i] if old_logps is not None and i < len(old_logps) else None
      )
      if old_lp is None and self.use_rollout_logps:
        old_lp = _extract_old_logps(item, len(c_arr))

      seq_tokens = (
          np.concatenate([p_arr, c_arr])
          if (len(p_arr) > 0 or len(c_arr) > 0)
          else np.zeros(0, dtype=np.int32)
      )
      seq_adv = np.full(len(c_arr), adv_val, dtype=np.float32)

      payload = datatypes.RLTrainerPayload(
          prompt_ids=p_arr,
          prompt_mask=np.ones(len(p_arr), dtype=np.float32),
          completion_ids=c_arr,
          completion_mask=act_arr,
          advantages=seq_adv,
          old_per_token_logps=np.asarray(old_lp, dtype=np.float32)
          if old_lp is not None
          else None,
          ref_per_token_logps=np.asarray(ref_lp, dtype=np.float32)
          if ref_lp is not None
          else None,
          returns=np.full(len(seq_tokens), vt_val, dtype=np.float32),
      )
      payloads.append(payload)
    return payloads

  def loss_fn(self) -> Callable[..., Any]:
    """Policy loss resolved by name via the function registry."""
    return function_registry.get_policy_loss_fn(self.policy_loss_fn)

  def build_gen_model_input_fn(
      self, pad_id: int, eos_id: int
  ) -> Callable[[Any], dict[str, Any]]:
    """Returns a model input generator function for TrainerWorker."""
    algo_config = types.SimpleNamespace(
        epsilon_low=getattr(self, "epsilon_low", self.clip_epsilon),
        epsilon_high=getattr(self, "epsilon_high", self.clip_epsilon),
        entropy_coef=self.entropy_coef,
        gamma=self.gamma,
        lam=self.lam,
    )
    return functools.partial(
        _algo_model_input,
        algo_config=algo_config,
        pad_id=pad_id,
        eos_id=eos_id,
    )

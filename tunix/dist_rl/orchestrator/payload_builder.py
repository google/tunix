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

"""Trajectory-to-payload conversion and advantage wiring for distributed RL."""

from collections.abc import Sequence

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.rl import advantages as rl_advantages
from tunix.rl import algorithm_config
from tunix.rl import function_registry
from tunix.rl import packing


def _extract_tokens_and_masks(
    item: datatypes.TrajectoryItem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Extracts prompt tokens, conversation tokens and masks from an item."""
  p_arr = np.asarray(item.traj["prompt_tokens"], dtype=np.int32).ravel()
  prompt_len = item.traj.get("prompt_length")
  if prompt_len is not None:
    p_arr = packing.unpad_prompt(p_arr, int(prompt_len))
  return (
      p_arr,
      np.asarray(item.traj["conversation_tokens"], dtype=np.int32).ravel(),
      np.asarray(item.traj["conversation_masks"], dtype=np.float32).ravel(),
  )


def _routed_experts_for(
    item: datatypes.TrajectoryItem,
    seq_len: int,
    *,
    encode_mode: packing.RoutedExpertsEncodeMode | None = None,
    num_experts: int | None = None,
) -> np.ndarray | None:
  """Aligns a rollout's captured routing to the payload's token sequence.

  Args:
    item: Trajectory item, whose `routed_experts` (if any) is `[captured_len,
      num_layers, top_k]` (`int16` or `+1`-shifted `uint8`) or `[captured_len,
      num_layers, top_k // 4, 5]` (`+1`-shifted `uint10`).
    seq_len: Length of the prompt+completion sequence in the payload.
    encode_mode: Optional compact encoding mode (`"auto"`, `"uint8"`,
      `"uint10"`, or `"int16"`). When None, preserves the input's existing
      compact `uint8`/`uint10` representation or coerces raw integer arrays to
      `int16`.
    num_experts: Optional total number of experts for `encode_mode="auto"`.

  Returns:
    `[seq_len - 1 or seq_len, num_layers, top_k]` (or 4D `uint10`) prefix view,
    or None when nothing was captured. Downstream assemblers (`pack_chunk` and
    `_routed_experts_aligned`) pre-fill their destination buffers with
    `UNSET_ROUTED_EXPERT` so an unrouted final token (`seq_len - 1`) falls back
    to the trainer's own gate without allocating a padded copy here.
  """
  routed = item.traj.get("routed_experts")
  if routed is None:
    return None
  raw = np.asarray(routed)
  try:
    enc = packing.routed_experts_encoding(raw, batched=False)
  except ValueError as exc:
    raise ValueError(
        "routed_experts must be [length, num_layers, top_k]; got shape"
        f" {raw.shape}"
    ) from exc
  routed_arr = (
      raw if enc in ("uint8", "uint10") else np.asarray(raw, dtype=np.int16)
  )
  min_len = max(seq_len - 1, 0)
  if routed_arr.shape[0] < min_len:
    raise ValueError(
        f"routed_experts length must be >= {min_len} (seq_len - 1 for"
        f" seq_len={seq_len}); got shape {routed_arr.shape}"
    )
  sliced = routed_arr[:seq_len]
  if encode_mode is not None:
    return packing.encode_routed_experts(
        sliced, mode=encode_mode, num_experts=num_experts, batched=False
    )
  return sliced


def _extract_old_logps(
    item: datatypes.TrajectoryItem, completion_len: int
) -> np.ndarray | None:
  """Extracts old_per_token_logps from TrajectoryItem."""
  old_lp = item.traj.get("old_logprobs")
  if old_lp is None:
    if completion_len == 0:
      return np.zeros(0, dtype=np.float32)
    return None
  old_lp = np.asarray(old_lp, dtype=np.float32).reshape(-1)
  if len(old_lp) != completion_len:
    raise ValueError(
        f"old_logprobs length {len(old_lp)} does not match completion length"
        f" {completion_len}"
    )
  return old_lp


def create_trainer_payloads(
    group: Sequence[datatypes.TrajectoryItem],
    rewards: Sequence[float] | np.ndarray,
    *,
    algo_config: algorithm_config.AlgorithmConfig,
) -> list[datatypes.RLTrainerPayload]:
  """Packages one prompt group of scored trajectories into trainer payloads.

  Args:
    group: The `G` trajectories sampled for one prompt, in rollout order.
    rewards: One scalar reward per trajectory, aligned with `group`.
    algo_config: Algorithm configuration. `advantage_estimator` selects the
      advantage math and `num_generations` is the group size handed to it;
      `use_rollout_logps` controls whether each rollout's `old_logprobs` are
      attached as `old_per_token_logps`; the `routed_experts_*` fields control
      how captured MoE routing is encoded.

  Returns:
    One unbatched `RLTrainerPayload` per trajectory, in `group` order. Invalid
    trajectories (`TrajectoryItem.is_valid` is False or a non-finite reward) get
    a zero advantage and a zeroed `completion_mask`, so they neither shape the
    group baseline nor contribute to the loss, while the static `[G]` batch
    dimension is kept.
  """
  r = np.asarray(rewards, dtype=np.float32).reshape(-1)
  if r.shape[0] != len(group):
    raise ValueError(
        "rewards must hold one entry per trajectory; got"
        f" {r.shape[0]} rewards for {len(group)} trajectories."
    )
  valid_mask = np.array(
      [item.is_valid for item in group], dtype=bool
  ) & np.isfinite(r)

  if algo_config.advantage_estimator == "gae":
    # "gae" is not registered in `function_registry`. The distributed PPO path
    # has no critic worker producing values yet, so this is the one-step
    # estimate with V = 0: `advantage = reward` and `returns = reward`.
    # `PPOConfig.gamma` / `gae_lambda` are therefore not applied here.
    # TODO(sizhi): Switch to token-level GAE once the critic worker exists.
    advs = np.where(valid_mask, r, 0.0).astype(np.float32)
    returns: np.ndarray | None = advs
  else:
    # A sample std / leave-one-out baseline needs at least
    # `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` valid peers per prompt group, so
    # a group below that threshold trains on nothing at all.
    if (
        int(np.sum(valid_mask))
        < rl_advantages.MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE
    ):
      valid_mask[:] = False
    advantage_estimator = function_registry.get_advantage_estimator(
        algo_config.advantage_estimator
    )
    advs = np.asarray(
        advantage_estimator(
            rewards=r,
            num_generations=algo_config.num_generations,
            valid_mask=valid_mask,
        ),
        dtype=np.float32,
    ).reshape(-1)
    if advs.shape[0] != len(group):
      raise ValueError(
          f"Advantage estimator {algo_config.advantage_estimator!r} returned"
          f" {advs.shape[0]} advantages for {len(group)} trajectories."
      )
    returns = None

  payloads = []
  for i, item in enumerate(group):
    p_arr, c_arr, act_arr = _extract_tokens_and_masks(item)
    if not valid_mask[i]:
      # While collector-filtered rollouts (`overlong_filter=True`) already
      # arrive with zeroed `conversation_masks`, a degenerate group (`< 2`
      # valid peers) also clears `valid_mask` for its lone healthy survivor.
      # Zeroing `act_arr` here keeps the static `[G]` batch dimension for
      # `BatchAssembler` while preventing that survivor's tokens from
      # inflating the `token-mean` loss denominator or contributing KL loss.
      act_arr = np.zeros_like(act_arr)
    # Custom estimators are not required to honor `valid_mask`, so the zero
    # advantage for invalid trajectories is enforced here.
    adv_val = float(advs[i]) if valid_mask[i] else 0.0
    payloads.append(
        datatypes.RLTrainerPayload(
            prompt_ids=p_arr,
            prompt_mask=np.ones(len(p_arr), dtype=np.float32),
            completion_ids=c_arr,
            completion_mask=act_arr,
            advantages=np.full(len(c_arr), adv_val, dtype=np.float32),
            old_per_token_logps=(
                _extract_old_logps(item, len(c_arr))
                if algo_config.use_rollout_logps
                else None
            ),
            returns=(
                None
                if returns is None
                else np.full(len(c_arr), float(returns[i]), dtype=np.float32)
            ),
            routed_experts=_routed_experts_for(
                item,
                len(p_arr) + len(c_arr),
                encode_mode=algo_config.routed_experts_encode_mode,
                num_experts=algo_config.routed_experts_num_experts,
            ),
            routed_experts_ref=(
                item.routed_experts_ref if valid_mask[i] else None
            ),
        )
    )
  return payloads

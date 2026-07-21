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

"""Algorithm adapter: the algorithm-specific seam of the orchestrator loop.

The loop driver is algorithm-agnostic; everything specific to an RL algorithm
lives behind `AlgorithmAdapter`:

  * `make_trajectory_requests` turns a batch of prompts into groups of G rollout
    requests (each group sharing a `group_id`);
  * `postprocess_group` turns a completed group into a `TrainExampleV1` (rewards
    -> advantages -> assemble);
  * `loss_spec` names the trainer-side loss and its config.

`AgenticGRPOAdapter` implements group-relative advantages and delegates padding
to the shared `trajectories_to_train_example` assembler.
"""

import dataclasses
import typing
from typing import Any

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import group_assembler
from tunix.experimental.orchestrator import request_ledger
from tunix.experimental.orchestrator import train_example_assembler


@dataclasses.dataclass(kw_only=True)
class LossSpec:
  """Trainer-side loss selection: a registry name plus its config."""

  name: str
  config: dict[str, Any] = dataclasses.field(default_factory=dict)


@typing.runtime_checkable
class AlgorithmAdapter(typing.Protocol):
  """The algorithm-specific surface the loop driver calls."""

  def make_trajectory_requests(
      self, rows: list[dict[str, Any]], step: int
  ) -> list[list[request_ledger.RequestRecord]]:
    """Returns one group of request records per prompt row (G per group)."""
    ...

  def postprocess_group(
      self,
      group: group_assembler.AssembledGroup,
      *,
      tokenizer_info: datatypes.TokenizerInfo,
      shape_config: datatypes.ShapeConfig,
  ) -> datatypes.TrainExampleV1:
    """Turns a completed group into a training example."""
    ...

  def loss_spec(self) -> LossSpec:
    """Returns the trainer-side loss name and config."""
    ...


def _concat_int(arrays: list[np.ndarray]) -> np.ndarray:
  if not arrays:
    return np.zeros(0, dtype=np.int32)
  return np.concatenate([np.asarray(a) for a in arrays])


class AgenticGRPOAdapter:
  """GRPO adapter: group-relative advantages over env rewards."""

  def __init__(
      self,
      *,
      group_size: int,
      mode: str = "train",
      incarnation: int = 0,
      use_rollout_logps: bool = False,
      sampling_params: datatypes.SamplingParams | None = None,
      loss_fn_name: str = "grpo",
      loss_config: dict[str, Any] | None = None,
      advantage_eps: float = 1e-8,
  ):
    if group_size < 2:
      raise ValueError(f"group_size must be >= 2 for GRPO, got {group_size}")
    self._group_size = group_size
    self._mode = mode
    self._incarnation = incarnation
    self._use_rollout_logps = use_rollout_logps
    self._sampling_params = sampling_params or datatypes.SamplingParams(
        max_tokens=128
    )
    self._loss_fn_name = loss_fn_name
    self._loss_config = dict(loss_config or {})
    self._advantage_eps = advantage_eps

  def make_trajectory_requests(
      self, rows: list[dict[str, Any]], step: int
  ) -> list[list[request_ledger.RequestRecord]]:
    groups = []
    for row_index, row in enumerate(rows):
      group_id = f"{self._mode}/{step}/{row_index}"
      records = []
      for sample_index in range(self._group_size):
        request = datatypes.RolloutRequest(
            request_id=f"{group_id}:{sample_index}",
            prompt_id=str(row.get("prompt_id", row_index)),
            prompt_text=row["prompt_text"],
            sampling_params=self._sampling_params,
        )
        records.append(
            request_ledger.RequestRecord(
                request=request,
                group_id=group_id,
                sample_index=sample_index,
                incarnation=self._incarnation,
                mode=self._mode,
            )
        )
      groups.append(records)
    return groups

  def postprocess_group(
      self,
      group: group_assembler.AssembledGroup,
      *,
      tokenizer_info: datatypes.TokenizerInfo,
      shape_config: datatypes.ShapeConfig,
  ) -> datatypes.TrainExampleV1:
    rewards = np.asarray(
        [result.env_reward for _, result in group.members], dtype=np.float32
    )
    advantages = self._group_relative_advantages(rewards)
    samples = [
        self._result_to_sample(record, result)
        for record, result in group.members
    ]
    return train_example_assembler.trajectories_to_train_example(
        samples,
        advantages,
        tokenizer_info=tokenizer_info,
        shape_config=shape_config,
        use_rollout_logps=self._use_rollout_logps,
    )

  def loss_spec(self) -> LossSpec:
    return LossSpec(name=self._loss_fn_name, config=dict(self._loss_config))

  def _group_relative_advantages(self, rewards: np.ndarray) -> np.ndarray:
    """Standard GRPO advantage: (r - mean) / (std + eps), atomic per group."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + self._advantage_eps)

  def _result_to_sample(
      self,
      record: request_ledger.RequestRecord,
      result: datatypes.RolloutResult,
  ) -> train_example_assembler.SampleTokens:
    completion_tokens = _concat_int([seg.tokens for seg in result.segments])
    completion_mask = _concat_int([seg.loss_mask for seg in result.segments])
    old_logprobs = None
    if result.segments and all(
        seg.logprobs is not None for seg in result.segments
    ):
      old_logprobs = np.concatenate(
          [np.asarray(seg.logprobs) for seg in result.segments]
      )
    del record  # Correlation only; the result carries the tokens.
    return train_example_assembler.SampleTokens(
        prompt_tokens=np.asarray(result.prompt_tokens),
        completion_tokens=completion_tokens,
        completion_mask=completion_mask,
        policy_version=result.policy_version,
        old_logprobs=old_logprobs,
    )

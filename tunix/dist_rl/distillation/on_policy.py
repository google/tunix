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

"""Sampled-token on-policy distillation, applied before batch assembly.

Teacher guidance adds a reverse-KL signal to the base policy advantage.
Scoring inputs are host arrays from the behavior policy, so neither the teacher
nor the old student participates in autodiff.
The resulting advantages use the existing padding/packing and policy loss.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from tunix.dist_rl.common import datatypes


@dataclasses.dataclass(frozen=True, kw_only=True)
class OnPolicyDistillationConfig:
  """Weights for teacher guidance and the base task advantage.

  ``task_reward_coefficient`` scales the *computed advantage*, after group
  reward normalization; scaling rewards before GRPO normalization would cancel
  a positive coefficient. Zero selects pure distillation. ``coefficient`` is
  independent of the existing reference-policy KL coefficient.
  """

  coefficient: float = 1.0
  task_reward_coefficient: float = 0.0

  def __post_init__(self) -> None:
    for name in ("coefficient", "task_reward_coefficient"):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"OPD {name} must be finite and non-negative.")


@dataclasses.dataclass(frozen=True)
class OPDStatistics:
  """Sufficient statistics for token-weighted, cross-group OPD metrics."""

  reverse_kl_sum: float = 0.0
  teacher_logprob_sum: float = 0.0
  student_logprob_sum: float = 0.0
  token_count: float = 0.0


def apply_opd_to_payloads(
    payloads: Sequence[datatypes.RLTrainerPayload],
    teacher_log_probs: Sequence[np.ndarray] | None,
    config: OnPolicyDistillationConfig,
) -> tuple[list[datatypes.RLTrainerPayload], OPDStatistics]:
  """Combines fixed teacher/behavior scores with unbatched token advantages.

  For every trainable response token, computes
  ``task_weight * advantage - coefficient * (old_student_logp - teacher_logp)``.
  Tool observations and masked/failed trajectories get exactly zero advantage.
  A sampled-token reverse-KL signal may be negative; it is not a full-vocabulary
  KL at each position. No input payload or score array is mutated.

  Args:
    payloads: One unpadded, one-dimensional payload per trajectory.
    teacher_log_probs: Completion-aligned teacher scores, including observation
      positions. May be None when the coefficient is zero or all tokens are
      masked. Scores are validated only where the loss mask is positive.
    config: Teacher and base-advantage weights.

  Returns:
    Updated payloads and masked sufficient statistics for logging.
  """
  if teacher_log_probs is not None and len(teacher_log_probs) != len(payloads):
    raise ValueError("OPD teacher result count must match payload count.")
  result = []
  reverse_sum = teacher_sum = student_sum = token_count = 0.0
  for index, payload in enumerate(payloads):
    tokens = np.asarray(payload.completion_ids)
    mask = np.asarray(payload.completion_mask, dtype=np.float32)
    advantage = np.asarray(payload.advantages, dtype=np.float32)
    if tokens.ndim != 1 or mask.shape != tokens.shape:
      raise ValueError(
          "OPD requires unbatched completion IDs and matching mask."
      )
    if advantage.shape != tokens.shape:
      raise ValueError("OPD requires one advantage per completion token.")
    if not np.all(np.isfinite(mask)) or np.any(mask < 0):
      raise ValueError("OPD completion mask must be finite and non-negative.")
    active = mask > 0
    combined = np.zeros(tokens.shape, dtype=np.float32)
    if config.task_reward_coefficient:
      if not np.all(np.isfinite(advantage[active])):
        raise ValueError("OPD task advantages must be finite on active tokens.")
      combined[active] = config.task_reward_coefficient * advantage[active]

    if config.coefficient and np.any(active):
      if payload.old_per_token_logps is None:
        raise ValueError("OPD requires sampled student logprobs.")
      if teacher_log_probs is None:
        raise ValueError("OPD requires teacher logprobs for active tokens.")
      student = np.asarray(payload.old_per_token_logps, dtype=np.float32)
      teacher = np.asarray(teacher_log_probs[index], dtype=np.float32)
      if student.shape != tokens.shape or teacher.shape != tokens.shape:
        raise ValueError(
            "OPD student and teacher logprobs must match completion shape."
        )
      if not (
          np.all(np.isfinite(student[active]))
          and np.all(np.isfinite(teacher[active]))
      ):
        raise ValueError("OPD logprobs must be finite on active tokens.")
      difference = student[active] - teacher[active]
      combined[active] -= config.coefficient * difference
      weights = mask[active].astype(np.float64)
      reverse_sum += float(np.sum(difference * weights))
      teacher_sum += float(np.sum(teacher[active] * weights))
      student_sum += float(np.sum(student[active] * weights))
      token_count += float(np.sum(weights))
    if not np.all(np.isfinite(combined)):
      raise ValueError("OPD produced non-finite advantages.")
    result.append(dataclasses.replace(payload, advantages=combined))
  return result, OPDStatistics(
      reverse_kl_sum=reverse_sum,
      teacher_logprob_sum=teacher_sum,
      student_logprob_sum=student_sum,
      token_count=token_count,
  )

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

"""External-teacher batch contracts for diffusion distillation."""

from typing import Protocol, TypeVar

import flax
from tunix.diffusion import types as diffusion_types

RawBatchT_contra = TypeVar("RawBatchT_contra", contravariant=True)


@flax.struct.dataclass(frozen=True)
class DiffusionDistillationBatch:
  """Student inputs and external teacher logits for diffusion distillation.

  An upstream model-aware integration is responsible for generating an
  on-policy student rollout and scoring that same rollout with the teacher.
  Tunix only consumes the resulting target-aligned tensors, so it does not own
  model-specific rollout, corruption, or checkpoint behavior.

  Attributes:
    student_batch: On-policy inputs, rollout targets, and per-target loss
      weights used to score the student.
    teacher_logits: Immutable teacher logits with shape ``[batch, length,
      vocab]`` aligned with ``student_batch.target_ids``.
    hard_target_ids: Optional expert target IDs for a hard-label imitation
      anchor. These targets use the same student logits and loss weights as the
      on-policy batch, but need not equal the rollout targets.
    hard_target_batch: Optional clean-state inputs, expert target IDs, and loss
      weights for a hard-label imitation anchor evaluated independently of the
      on-policy student state.
  """

  student_batch: diffusion_types.DiffusionTokenBatch
  teacher_logits: diffusion_types.Array
  hard_target_ids: diffusion_types.Array | None = None
  hard_target_batch: diffusion_types.DiffusionTokenBatch | None = None

  @classmethod
  def create(
      cls,
      *,
      student_batch: diffusion_types.DiffusionTokenBatch,
      teacher_logits: diffusion_types.Array,
      hard_target_ids: diffusion_types.Array | None = None,
      hard_target_batch: diffusion_types.DiffusionTokenBatch | None = None,
  ) -> "DiffusionDistillationBatch":
    """Constructs and validates a diffusion distillation batch."""

    return cls(
        student_batch=student_batch,
        teacher_logits=teacher_logits,
        hard_target_ids=hard_target_ids,
        hard_target_batch=hard_target_batch,
    ).validate()

  def validate(self) -> "DiffusionDistillationBatch":
    """Validates the student batch and target alignment of teacher logits."""

    if not isinstance(self.student_batch, diffusion_types.DiffusionTokenBatch):
      raise TypeError("student_batch must be a DiffusionTokenBatch")
    self.student_batch.validate()
    diffusion_types.validate_diffusion_logits(
        self.student_batch, self.teacher_logits
    )
    if self.hard_target_ids is not None and self.hard_target_batch is not None:
      raise ValueError(
          "hard_target_ids and hard_target_batch are mutually exclusive"
      )
    if self.hard_target_ids is not None:
      diffusion_types.DiffusionTokenBatch.create(
          model_inputs=self.student_batch.model_inputs,
          target_ids=self.hard_target_ids,
          loss_weights=self.student_batch.loss_weights,
      )
    if self.hard_target_batch is not None:
      if not isinstance(
          self.hard_target_batch, diffusion_types.DiffusionTokenBatch
      ):
        raise TypeError("hard_target_batch must be a DiffusionTokenBatch")
      self.hard_target_batch.validate()
      if (
          self.hard_target_batch.target_ids.shape
          != self.student_batch.target_ids.shape
      ):
        raise ValueError(
            "hard_target_batch target_ids must match student_batch target_ids "
            "shape"
        )
    return self


class PreparedDiffusionDistillationBatchAdapter(Protocol[RawBatchT_contra]):
  """Adapts a prepared external rollout to DiffusionDistillationBatch."""

  def __call__(self, batch: RawBatchT_contra, /) -> DiffusionDistillationBatch:
    ...

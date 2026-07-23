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
    student_batch: Inputs, hard targets, and per-target loss weights used to
      score the student.
    teacher_logits: Immutable teacher logits with shape ``[batch, length,
      vocab]`` aligned with ``student_batch.target_ids``.
  """

  student_batch: diffusion_types.DiffusionTokenBatch
  teacher_logits: diffusion_types.Array

  @classmethod
  def create(
      cls,
      *,
      student_batch: diffusion_types.DiffusionTokenBatch,
      teacher_logits: diffusion_types.Array,
  ) -> "DiffusionDistillationBatch":
    """Constructs and validates a diffusion distillation batch."""

    return cls(
        student_batch=student_batch,
        teacher_logits=teacher_logits,
    ).validate()

  def validate(self) -> "DiffusionDistillationBatch":
    """Validates the student batch and target alignment of teacher logits."""

    if not isinstance(self.student_batch, diffusion_types.DiffusionTokenBatch):
      raise TypeError("student_batch must be a DiffusionTokenBatch")
    self.student_batch.validate()
    diffusion_types.validate_diffusion_logits(
        self.student_batch, self.teacher_logits
    )
    return self


class PreparedDiffusionDistillationBatchAdapter(Protocol[RawBatchT_contra]):
  """Converts a freshly prepared external rollout to the OPD contract."""

  def __call__(self, batch: RawBatchT_contra, /) -> DiffusionDistillationBatch:
    ...

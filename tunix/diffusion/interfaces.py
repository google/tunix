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

"""Callable interfaces for framework-neutral diffusion integrations."""

from typing import Protocol, TypeVar

from flax import nnx
from tunix.diffusion import types

RawBatchT_contra = TypeVar("RawBatchT_contra", contravariant=True)


class DiffusionBatchAdapter(Protocol[RawBatchT_contra]):
  """Converts an external batch to Tunix's canonical diffusion batch."""

  def __call__(self, batch: RawBatchT_contra, /) -> types.DiffusionTokenBatch:
    ...


class DiffusionLogitsFn(Protocol):
  """Computes target-aligned token logits for a diffusion model."""

  def __call__(
      self,
      model: nnx.Module,
      model_inputs: types.ModelInputs,
      /,
  ) -> types.Array:
    ...


def compute_diffusion_logits(
    model: nnx.Module,
    batch: types.DiffusionTokenBatch,
    logits_fn: DiffusionLogitsFn,
) -> types.Array:
  """Computes diffusion logits and validates target alignment."""

  batch.validate()
  logits = logits_fn(model, batch.model_inputs)
  return types.validate_diffusion_logits(batch, logits)

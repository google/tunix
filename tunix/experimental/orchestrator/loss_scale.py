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

"""Denominator-aware loss scales for gradient accumulation.

The orchestrator owns the normalization: it hands each micro-batch a
`loss_scale` that the trainer multiplies onto its already-normalized micro-batch
loss, and `update()` then SUMS the scaled gradients (no further division). With

    loss_scale_k = denominator_k / sum_j(denominator_j)

the summed result equals the global mean exactly, where the denominator is the
masked token count for token-mean aggregation or the sequence count for
sequence-mean aggregation. Equal micro-batches give `1/N` each, reproducing a
plain mean over N accumulation steps.
"""

from collections.abc import Sequence


def loss_scales_from_denominators(denominators: Sequence[float]) -> list[float]:
  """Returns per-micro-batch loss scales that sum-normalize to a global mean.

  Args:
    denominators: One non-negative denominator per micro-batch (e.g. masked
      token counts), in accumulation order.

  Returns:
    Scales `denominator_k / sum(denominators)`, summing to 1.0 (or all zero if
    every denominator is zero).

  Raises:
    ValueError: If `denominators` is empty or contains a negative value.
  """
  if not denominators:
    raise ValueError("need at least one denominator")
  if any(d < 0 for d in denominators):
    raise ValueError(f"denominators must be non-negative, got {denominators}")
  total = float(sum(denominators))
  if total <= 0.0:
    # A mini-batch with no contributing tokens produces no gradient.
    return [0.0 for _ in denominators]
  return [float(d) / total for d in denominators]

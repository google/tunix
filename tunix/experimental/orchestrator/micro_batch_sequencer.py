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

"""Sequences a fully-assembled mini-batch into ordered micro-steps.

Given a mini-batch already split into micro-batch payloads, this computes each
micro-batch's normalization denominator (masked tokens for token-mean
aggregation, contributing sequences for sequence-mean) and the matching
`loss_scale`, and emits the ordered micro-steps a trainer replays.

A mini-batch is fully assembled before any of its `fwd_bwd` calls, so the global
denominators are known up front; that is what makes the scales sum-correct.
`num_iterations` (mu) replays the same micro-batches under fresh accumulation
groups -- one optimizer update per replay.
"""

import dataclasses

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import loss_scale as loss_scale_lib


@dataclasses.dataclass(kw_only=True)
class MicroStep:
  """One `fwd_bwd` call within an accumulation group.

  Attributes:
    accum_id: The accumulation group id (one per optimizer update).
    micro_index: Position of this micro-batch within its group (0..N-1).
    payload: The micro-batch to run.
    loss_scale: Multiplier for this micro-batch's normalized loss (I7).
  """

  accum_id: str
  micro_index: int
  payload: datatypes.TrainerPayload
  loss_scale: float


def micro_batch_denominator(
    payload: datatypes.TrainerPayload, loss_agg_mode: str
) -> float:
  """Returns a micro-batch's normalization denominator for the aggregation mode.

  Args:
    payload: The micro-batch; its `loss_mask` supplies the counts.
    loss_agg_mode: "token-mean" (masked token count) or a "sequence-mean" variant
      (count of sequences with at least one contributing token).

  Returns:
    The denominator as a float.
  """
  mask = np.asarray(payload.loss_mask)
  if loss_agg_mode.startswith("sequence-mean"):
    if mask.ndim < 2:
      return float((mask > 0).any().astype(np.int64))
    return float((mask.sum(axis=-1) > 0).sum())
  return float(mask.sum())


def plan_micro_steps(
    micro_batches: list[datatypes.TrainerPayload],
    *,
    accum_id: str,
    loss_agg_mode: str = "token-mean",
) -> list[MicroStep]:
  """Plans one accumulation group's micro-steps with I7 loss scales.

  Args:
    micro_batches: The mini-batch's micro-batches, in accumulation order.
    accum_id: The accumulation group id for this update.
    loss_agg_mode: Loss aggregation mode driving the denominator choice.

  Returns:
    Ordered micro-steps; `len(...)` is the `expected_micro_steps` for `update()`.

  Raises:
    ValueError: If `micro_batches` is empty.
  """
  if not micro_batches:
    raise ValueError("cannot plan an accumulation group with zero micro-batches")
  denominators = [
      micro_batch_denominator(mb, loss_agg_mode) for mb in micro_batches
  ]
  scales = loss_scale_lib.loss_scales_from_denominators(denominators)
  return [
      MicroStep(
          accum_id=accum_id,
          micro_index=index,
          payload=payload,
          loss_scale=scale,
      )
      for index, (payload, scale) in enumerate(zip(micro_batches, scales))
  ]

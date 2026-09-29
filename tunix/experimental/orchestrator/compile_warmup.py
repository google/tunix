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

"""Abstract trainer micro-batch for compiling the train step ahead of rollout.

The trainer compiles its step for the first batch structure it is given, which
on a cold start is the first real micro-batch, after step-0 rollout. A
sequence-packed micro-batch's structure does not depend on the data: every
chunk is `[batch_size, max_packed_len]` rows of fixed dtypes, and only the set
of optional fields varies. So one can be built before any rollout returns, by
running a synthetic group through the same algorithm adapter, assembler and
program-side updates the real batches go through, and compiled while step-0
rollout runs.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.rl import common as rl_common

# Lengths of the synthetic trajectories. Packed rows are padded to
# `max_packed_len` whatever these are, so they only need to be short.
_PROMPT_LEN = 8
_COMPLETION_LEN = 8


def build_trainer_warmup_payload(
    *,
    algo: algorithm_adapter.AlgorithmAdapter,
    assembler: Any,
    rollout_logprobs: bool,
    routed_experts_shape: tuple[int, int] | None = None,
    sampler_is: str | None = None,
    sampler_is_threshold: float = 2.0,
    seq_logprob_error_threshold: float | None = None,
) -> datatypes.RLTrainerPayload | None:
  """Returns an abstract micro-batch shaped like the ones `train_stage` sends.

  Args:
    algo: The program's algorithm adapter; its `create_trainer_payloads` builds
      the per-trajectory payloads.
    assembler: The program's assembler. Only a `SequencePackedBatchAssembler`
      has data-independent batch shapes; anything else returns None.
    rollout_logprobs: Whether rollouts carry per-token log-probabilities
      (`GenerationArgs.return_logprobs`).
    routed_experts_shape: The trainer's `(num_layers, top_k)` when rollouts
      carry router-replay experts, else None.
    sampler_is: As on `StandardRLProgram`.
    sampler_is_threshold: As on `StandardRLProgram`.
    seq_logprob_error_threshold: As on `StandardRLProgram`.

  Returns:
    An `RLTrainerPayload` of `jax.ShapeDtypeStruct` leaves and empty metadata,
    or None when the assembler's batch shapes depend on the data.
  """
  if not isinstance(assembler, batch_assembly.SequencePackedBatchAssembler):
    return None
  num_generations = algo.num_generations
  # A private assembler, so the program's buffer and batch counter are
  # untouched. mini_batch_size=1 drains the one group fed below.
  packer = batch_assembly.SequencePackedBatchAssembler(
      batch_size=assembler.batch_size,
      num_generations=num_generations,
      mini_batch_size=1,
      max_packed_len=assembler.max_packed_len,
      pad_id=assembler.pad_id,
      max_segments_per_packed_row=assembler.max_segments_per_packed_row,
      segment_align_multiple=assembler.segment_align_multiple,
  )
  # Any id but the pad id, so every token reads as a real one.
  token = assembler.pad_id + 1
  traj: dict[str, Any] = {
      "prompt_tokens": np.full(_PROMPT_LEN, token, np.int32),
      "conversation_tokens": np.full(_COMPLETION_LEN, token, np.int32),
      "conversation_masks": np.ones(_COMPLETION_LEN, np.float32),
      "status": datatypes.TrajectoryStatus.SUCCEEDED.name,
  }
  if rollout_logprobs:
    traj["old_logprobs"] = np.zeros(_COMPLETION_LEN, np.float32)
  if routed_experts_shape is not None:
    # One layer and one slot while packing, widened to the trainer's routing
    # shape in the abstract payload below: a real [B, T, L, K] int16 array is
    # gigabytes of host memory.
    traj["routed_experts"] = np.zeros(
        (_PROMPT_LEN + _COMPLETION_LEN, 1, 1), np.int16
    )
  group = [
      datatypes.TrajectoryItem(
          prompt_id="trainer_compile_warmup", group_index=i, traj=dict(traj)
      )
      for i in range(num_generations)
  ]
  payloads = algo.create_trainer_payloads(
      group, rewards=[float(i % 2) for i in range(num_generations)]
  )
  # A group that overflows one chunk yields several of the same shape.
  batches = packer.feed(payloads) + packer.flush()
  batch = batches[0].payload

  # The rest mirrors what `train_stage` does to each micro-batch before
  # `train_step`, since each can add a field.
  if getattr(algo, "requires_reference_kl", False):
    batch = batch_assembly.with_ref_per_token_logps(
        batch, np.zeros(np.shape(batch.completion_ids), np.float32)
    )
  algo_config = getattr(algo, "algo_config", None)
  if (
      batch.old_per_token_logps is not None
      and algo_config is not None
      and algo_config.use_rollout_logps
  ):
    trainer_logps = np.zeros(np.shape(batch.completion_ids), np.float32)
    _, sampler_is_weights, filtered_completion_mask = (
        rl_common.sampler_trainer_agreement(
            batch.old_per_token_logps,
            trainer_logps,
            batch.completion_mask,
            sampler_is=sampler_is,
            sampler_is_threshold=sampler_is_threshold,
            seq_logprob_error_threshold=seq_logprob_error_threshold,
            segment_ids=batch.segment_ids,
        )
    )
    updates: dict[str, Any] = {}
    if seq_logprob_error_threshold is not None:
      updates["completion_mask"] = filtered_completion_mask
    if sampler_is_weights is not None:
      updates["sampler_is_weights"] = sampler_is_weights
    if sampler_is == "token" or seq_logprob_error_threshold is not None:
      updates["old_per_token_logps"] = trainer_logps
    batch = batch.replace(**updates)

  # The trainer drops metadata before it builds its inputs, so it is not part
  # of what a compiled step is keyed on.
  abstract = jax.tree.map(
      lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)),
      batch.replace(metadata={}),
  )
  if routed_experts_shape is not None:
    rows, length = np.shape(batch.completion_ids)
    abstract = abstract.replace(
        routed_experts=jax.ShapeDtypeStruct(
            (rows, length, *routed_experts_shape), jnp.int16
        )
    )
  return abstract


def describe_payload(payload: Any) -> str:
  """Returns `name[shape] dtype` for each array field of a payload."""
  fields = []
  for field in dataclasses.fields(payload):
    value = getattr(payload, field.name)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
      shape = ",".join(str(dim) for dim in value.shape)
      fields.append(f"{field.name}[{shape}] {np.dtype(value.dtype).name}")
  num_segments = getattr(payload, "num_segments", None)
  if num_segments is not None:
    fields.append(f"num_segments={num_segments}")
  return ", ".join(fields)

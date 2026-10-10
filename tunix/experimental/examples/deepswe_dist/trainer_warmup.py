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

"""Synthetic trainer micro-batch for compiling the trainer at bring-up.

The orchestrator hands every worker's `compile` hook a `dummy_data` payload.
With `None` the training engine defers compilation to the first real
micro-batch, which then pays the whole XLA compile inside the timed run. This
module builds one micro-batch with the same pytree structure, shapes and dtypes
as the live ones, so the engine can compile against it during bring-up.

The payload is produced by the live code path rather than written by hand:
synthetic `TrajectoryItem`s carrying the fields the trajectory collector emits
go through the algorithm's `create_trainer_payloads` and then through a fresh
batch assembler built exactly as `RLProgram` builds its own. Only the token
values are synthetic; no dataset or tokenizer is touched.
"""

from __future__ import annotations

import logging
import dataclasses
from typing import Any

import jax
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly
from tunix.rl import packing

# A synthetic token id inside every vocabulary; its value never reaches a
# shape, so any small non-pad id serves.
_SYNTHETIC_TOKEN_ID = 1


def _items_per_row(
    item_tokens: int, *, budget: int, max_segments: int, align: int
) -> int:
  """Number of `item_tokens`-long items one packed row of `budget` holds."""
  count, cursor = 0, 0
  while count < max_segments:
    start = packing.align_offset(cursor, align) if count else 0
    if start + item_tokens > budget:
      break
    cursor = start + item_tokens
    count += 1
  return max(count, 1)


def _synthetic_group(
    group_idx: int,
    *,
    num_generations: int,
    prompt_len: int,
    completion_len: int,
) -> list[datatypes.TrajectoryItem]:
  """One prompt group shaped like the trajectory collector's Token-mode output."""
  items = []
  for g in range(num_generations):
    items.append(
        datatypes.TrajectoryItem(
            prompt_id=f"warmup_{group_idx}",
            group_index=g,
            traj={
                "prompt_tokens": np.full(
                    prompt_len, _SYNTHETIC_TOKEN_ID, dtype=np.int32
                ),
                "conversation_tokens": np.full(
                    completion_len, _SYNTHETIC_TOKEN_ID, dtype=np.int32
                ),
                "conversation_masks": np.ones(completion_len, dtype=np.float32),
                "status": datatypes.TrajectoryStatus.SUCCEEDED.name,
                "old_logprobs": np.zeros(completion_len, dtype=np.float32),
                "routed_experts": None,
                "policy_version": 0,
            },
            metadata={"warmup": True},
            is_valid=True,
        )
    )
  return items


def build_warmup_payload(
    *,
    algo: Any,
    batch_config: batch_assembly.BatchConfig,
    num_generations: int,
    mini_batch_size: int,
    train_micro_batch_size: int,
    prompt_len: int = 1000,
    completion_len: int = 1000,
) -> datatypes.RLTrainerPayload:
  """Returns one synthetic trainer micro-batch shaped like the live ones.

  Args:
    algo: The run's algorithm adapter; its `create_trainer_payloads` turns the
      synthetic trajectories into unbatched payloads, so optional fields are
      present or absent exactly as in the run (e.g. `old_per_token_logps` is
      absent when `use_rollout_logps` is off).
    batch_config: The `BatchConfig` the program uses (after `RLProgram` fills in
      `max_response_length`).
    num_generations: Rollouts per prompt group (G).
    mini_batch_size: Prompt groups per optimizer update.
    train_micro_batch_size: The algorithm's `train_micro_batch_size`.
    prompt_len: Synthetic prompt length per trajectory, clipped to the budget.
    completion_len: Synthetic completion length per trajectory, clipped to the
      budget.

  Returns:
    The first `RLTrainerPayload` the assembler emits, with any per-micro-batch
    fields `RLProgram` adds before `train_step` mirrored onto it.
  """
  # A fresh assembler: feeding the program's own would advance its lineage
  # batch counter.
  assembler = batch_assembly.create_batch_assembler(
      num_generations=num_generations,
      mini_batch_size=mini_batch_size,
      train_micro_batch_size=train_micro_batch_size,
      batch_config=batch_config,
  )
  if batch_config.max_prompt_length is not None:
    prompt_len = min(prompt_len, batch_config.max_prompt_length)
  if batch_config.max_response_length is not None:
    completion_len = min(completion_len, batch_config.max_response_length)

  if isinstance(assembler, batch_assembly.SequencePackedBatchAssembler):
    budget = assembler.max_packed_len
    if prompt_len + completion_len > budget:
      prompt_len = min(prompt_len, budget // 2)
      completion_len = budget - prompt_len
    max_segments = packing.effective_max_segments(
        budget, assembler.max_segments_per_packed_row
    )
    per_row = _items_per_row(
        prompt_len + completion_len,
        budget=budget,
        max_segments=max_segments,
        align=assembler.segment_align_multiple,
    )
    num_items = assembler.batch_size * per_row
  else:
    num_items = assembler.batch_size
  # Stay below the optimizer-update boundary so `feed` holds every item and the
  # single `flush` below emits them.
  num_items = min(num_items, assembler.rollouts_per_optimizer_update - 1)
  num_groups = max(1, -(-num_items // num_generations))

  emitted: list[batch_assembly.AssembledBatch] = []
  for group_idx in range(num_groups):
    group = _synthetic_group(
        group_idx,
        num_generations=num_generations,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )
    # Mixed rewards keep the group-relative advantages non-degenerate.
    rewards = [float(g % 2) for g in range(num_generations)]
    payloads = algo.create_trainer_payloads(group, rewards)
    emitted.extend(assembler.feed(payloads))
  emitted.extend(assembler.flush())
  if not emitted:
    raise RuntimeError("Warm-up batch assembler emitted no micro-batch.")
  if len(emitted) > 1:
    logging.warning(
        "Warm-up assembler emitted %d micro-batches; compiling against the"
        " first.",
        len(emitted),
    )
  # `metadata` is static pytree data but the engine drops it before compiling
  # (`_prepare_batch`); clearing it keeps the synthetic lineage out of the RPC.
  payload = dataclasses.replace(emitted[0].payload, metadata={})

  # Fields RLProgram attaches per micro-batch before `train_step`.
  if getattr(algo, "requires_reference_kl", False):
    payload = batch_assembly.with_ref_per_token_logps(
        payload,
        np.zeros(np.asarray(payload.completion_ids).shape, dtype=np.float32),
    )
  algo_config = getattr(algo, "algo_config", None)
  if (
      payload.old_per_token_logps is not None
      and getattr(algo_config, "use_rollout_logps", False)
      and getattr(algo_config, "sampler_is", None) is not None
  ):
    logging.warning(
        "Warm-up payload omits the sampler_is_weights that the sampler/trainer"
        " agreement adds in the live run; the first real micro-batch will"
        " recompile."
    )
  return payload


def describe_payload(payload: datatypes.RLTrainerPayload) -> str:
  """One-line summary of a payload's leaves (name, shape, dtype) and statics."""
  leaves = jax.tree_util.tree_flatten_with_path(payload)[0]
  parts = [
      f"{jax.tree_util.keystr(path)}={tuple(np.shape(leaf))}:"
      f"{np.asarray(leaf).dtype}"
      for path, leaf in leaves
  ]
  return (
      f"num_segments={payload.num_segments} leaves=[" + ", ".join(parts) + "]"
  )

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

"""Layer 2A: Universal Batch Assembly (batch_assembly.py) following Orchestrator V2.

Generic tensor packing utility for unbatched `RLTrainerPayload` objects (or
custom objects with token arrays). Supports:
- 1D Sequence Packing (`SequencePackedBatchAssembler`) for Flash/FlexAttention
(>90% MXU).
- Simple 2D Rectangular Padding (`PaddedBatchAssembler`).

# TODO: Align SequencePackedBatchAssembler with the rest of the ecosystem and
potentially move to a common library.
"""

import collections
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Any, Generic, NamedTuple, Protocol, TypeVar
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.rl import packing
from tunix.rl import utils as rl_utils

T = TypeVar("T")

_BATCH_ID_PREFIX: str = "batch"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PaddingStats:
  """Per-row token occupancy of one assembled microbatch.

  Attributes:
    row_valid_tokens: `[B]` number of real (non-padding) tokens in each row.
    row_num_sequences: `[B]` number of trajectories placed in each row. Rows
      with 0 sequences are fully padded filler rows.
    row_capacity: Static token capacity of every row (`max_packed_len` for
      sequence packing, `max_prompt_length + max_response_length` for padding).
  """

  row_valid_tokens: np.ndarray
  row_num_sequences: np.ndarray
  row_capacity: int

  def __post_init__(self):
    if self.row_valid_tokens.ndim != 1:
      raise ValueError(
          "row_valid_tokens must be 1D, got shape"
          f" {self.row_valid_tokens.shape}."
      )
    if self.row_num_sequences.shape != self.row_valid_tokens.shape:
      raise ValueError(
          "row_num_sequences shape"
          f" {self.row_num_sequences.shape} does not match row_valid_tokens"
          f" shape {self.row_valid_tokens.shape}."
      )
    if self.row_valid_tokens.size == 0:
      raise ValueError("PaddingStats requires at least one row.")
    if self.row_capacity <= 0:
      raise ValueError(
          f"row_capacity must be positive, got {self.row_capacity}."
      )
    if np.any(self.row_valid_tokens < 0) or np.any(
        self.row_valid_tokens > self.row_capacity
    ):
      raise ValueError(
          f"row_valid_tokens must lie in [0, {self.row_capacity}], got"
          f" {self.row_valid_tokens.tolist()}."
      )
    if np.any(self.row_num_sequences < 0):
      raise ValueError(
          "row_num_sequences must be non-negative, got"
          f" {self.row_num_sequences.tolist()}."
      )
    if self.valid_tokens == 0:
      raise ValueError("PaddingStats requires at least one valid token.")

  @property
  def num_rows(self) -> int:
    return int(self.row_valid_tokens.size)

  @property
  def valid_tokens(self) -> int:
    return int(self.row_valid_tokens.sum())

  @property
  def capacity_tokens(self) -> int:
    return self.num_rows * self.row_capacity

  @property
  def padding_ratio(self) -> float:
    """Fraction of the `[B, row_capacity]` microbatch that is padding."""
    return 1.0 - self.valid_tokens / self.capacity_tokens

  @property
  def row_fill(self) -> np.ndarray:
    """`[B]` fraction of each row occupied by real tokens."""
    return self.row_valid_tokens / self.row_capacity

  @property
  def row_imbalance(self) -> float:
    """Max-over-mean of per-row valid tokens; 1.0 means perfectly balanced."""
    return float(self.row_valid_tokens.max() / self.row_valid_tokens.mean())


def summarize_padding_stats(
    stats: Sequence[PaddingStats],
) -> dict[str, float]:
  """Aggregates per-microbatch padding stats of one optimizer step.

  Args:
    stats: Padding stats of every microbatch sent to the trainer in the step.

  Returns:
    Mapping of metric name to value:
      ratio: token-weighted padding fraction over the whole step.
      microbatch_ratio_{mean,max,min}: padding fraction across microbatches.
      microbatch_imbalance: max-over-mean of valid tokens across microbatches.
      row_imbalance_{mean,max}: per-microbatch row max-over-mean, aggregated.
      row_fill_{min,max}: per-row occupancy extremes over all rows.
      empty_rows: number of fully padded rows holding no sequence.
      sequences_per_row_{mean,max}: trajectories per row over all rows.
      valid_tokens / capacity_tokens: step totals.
  """
  if not stats:
    raise ValueError("summarize_padding_stats requires at least one entry.")
  valid = np.array([s.valid_tokens for s in stats], dtype=np.float64)
  capacity = np.array([s.capacity_tokens for s in stats], dtype=np.float64)
  mb_ratio = np.array([s.padding_ratio for s in stats])
  row_imbalance = np.array([s.row_imbalance for s in stats])
  row_fill = np.concatenate([s.row_fill for s in stats])
  row_num_sequences = np.concatenate([s.row_num_sequences for s in stats])
  return {
      "ratio": float(1.0 - valid.sum() / capacity.sum()),
      "microbatch_ratio_mean": float(mb_ratio.mean()),
      "microbatch_ratio_max": float(mb_ratio.max()),
      "microbatch_ratio_min": float(mb_ratio.min()),
      "microbatch_imbalance": float(valid.max() / valid.mean()),
      "row_imbalance_mean": float(row_imbalance.mean()),
      "row_imbalance_max": float(row_imbalance.max()),
      "row_fill_min": float(row_fill.min()),
      "row_fill_max": float(row_fill.max()),
      "empty_rows": float(np.sum(row_num_sequences == 0)),
      "sequences_per_row_mean": float(row_num_sequences.mean()),
      "sequences_per_row_max": float(row_num_sequences.max()),
      "valid_tokens": float(valid.sum()),
      "capacity_tokens": float(capacity.sum()),
  }


class AssembledBatch(NamedTuple):
  """Microbatch payload paired with optimizer-update completion status."""

  payload: datatypes.RLTrainerPayload
  is_final_batch: bool
  padding_stats: PaddingStats
  trajectory_ids: tuple[str, ...] = ()


@dataclasses.dataclass
class BatchConfig:
  """Configuration for batch assembly.

  Attributes:
    pad_id: Token ID used for padding prompts and completions.
    max_prompt_length: Maximum prompt length for padding or budget validation.
    max_response_length: Maximum response length for padding or budget
      validation.
    max_seq_token_per_tpu: Maximum packed sequence tokens per TPU. When
      configured, SequencePackedBatchAssembler is used instead of
      PaddedBatchAssembler.
    max_segments_per_packed_row: Maximum segments per packed row when sequence
      packing is enabled.
    segment_alignment_boundary: Enforces that every packed sequence starts at an
      index that is a multiple of this value, padding any gaps. By default (1),
      sequences are packed tightly back-to-back. Use larger values to align
      with chunk boundaries for models with chunked processing (e.g., 64 for
      Qwen3.5 GatedDeltaNet).
    trainer_fsdp: Trainer FSDP mesh dimension size for sequence packing.
    trainer_dp: Trainer DP mesh dimension size for sequence packing.
  """

  pad_id: int = 0
  max_prompt_length: int | None = None
  max_response_length: int | None = None
  max_seq_token_per_tpu: int | None = None
  max_segments_per_packed_row: int | None = None
  segment_alignment_boundary: int = packing.DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY
  trainer_fsdp: int | None = None
  trainer_dp: int | None = None


def _extract_trajectory_id(item: Any) -> str:
  """Extracts the standardized trajectory id from payload metadata."""
  metadata = getattr(item, "metadata", None) or {}
  return str(metadata.get("traj_id", ""))


class BatchAssembler(Generic[T], Protocol):
  """Universal batch assembly protocol for microbatch packing.

  Attributes:
    num_generations: Number of rollout trajectories / generations generated per
      prompt group (G). Must be a positive integer.
  """

  num_generations: int
  mini_batch_size: int

  @property
  def rollouts_per_optimizer_update(self) -> int:
    """Total number of rollouts expected per optimizer update."""
    return self.mini_batch_size * self.num_generations

  def feed(
      self,
      items: Sequence[T],
  ) -> Iterator[AssembledBatch]:
    """Ingests rollouts and yields microbatches, flushing at the update boundary."""
    ...

  def flush(
      self,
  ) -> Iterator[AssembledBatch]:
    """Drains remaining buffered items, padding to the required static tensor shape."""
    ...

  # TODO (tunix-dev): we should not allow `start_batch_index` to be None once failure recovery logic is implemented.
  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Resets internal state and discards optimizer-update progress.

    Unlike `flush()`, which emits remaining items as padded microbatches,
    `reset()` unconditionally drops any partially accumulated items or bins
    without packing or emitting them, and resets the optimizer-update rollout
    counter back to zero.

    This is typically invoked during pipeline aborts or error recovery (e.g.,
    when an RL program stage encounters an exception and incomplete rollouts
    must be purged to prevent state leakage into subsequent steps) or when
    restarting the orchestrator.

    Args:
      start_batch_index: Optional batch index to reset the microbatch lineage
        tracking counter (e.g., when resuming from a checkpoint). If None, the
        existing batch counter is preserved to maintain monotonic lineage
        tracking IDs.
    """
    ...


def _left_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  arr = np.asarray(values, dtype=np.int32).reshape(-1)[-length:]
  out = np.full(length, pad_id, dtype=np.int32)
  mask = np.zeros(length, dtype=np.float32)
  if arr.size:
    out[-arr.size :] = arr
    mask[-arr.size :] = 1.0
  return out, mask


def _routed_experts_aligned(
    routed: np.ndarray,
    prompt_len: int,
    completion_len: int,
    max_prompt_length: int,
    max_response_length: int,
    *,
    out: np.ndarray,
) -> np.ndarray:
  """Lays one row of routing out over the padded `[prompt | completion]`.

  Mirrors how the token ids themselves are padded -- prompts right-aligned in
  the prompt window (keeping the tail, as `_left_pad` does) and completions
  left-aligned in the response window -- so replayed routing stays attached to
  the token it was captured for. Everything else is left unset.

  Args:
    routed: `[prompt_len + completion_len - 1 or prompt_len + completion_len,
      num_layers, top_k]` routing for one generation. Only axis 0 (the per-token
      axis) is ever sliced below; the trailing `[num_layers, top_k]` axes are
      carried through untouched.
    prompt_len: Unpadded prompt length, i.e. where the completion starts.
    completion_len: Unpadded completion length.
    max_prompt_length: Padded prompt width.
    max_response_length: Padded completion width.
    out: Pre-initialized `[max_prompt_length + max_response_length, num_layers,
      top_k]` buffer (filled with `UNSET_ROUTED_EXPERT`) to populate in-place.

  Returns:
    `[max_prompt_length + max_response_length, num_layers, top_k]`.
  """
  routed = np.asarray(routed, dtype=np.int16)
  min_len = max(prompt_len + completion_len - 1, 0)
  if routed.shape[0] < min_len:
    raise ValueError(
        f"routed_experts length must be >= {min_len} (prompt_len +"
        f" completion_len - 1 for prompt_len={prompt_len},"
        f" completion_len={completion_len}); got shape {routed.shape}"
    )
  # Prompts are left-padded, so an over-long one keeps its tail; completions are
  # right-padded, so an over-long one keeps its head.
  kept_prompt_start = max(prompt_len - max_prompt_length, 0)
  kept_prompt_len = prompt_len - kept_prompt_start
  kept_completion_end = prompt_len + min(completion_len, max_response_length)
  prompt_part = routed[kept_prompt_start:prompt_len]
  completion_part = routed[prompt_len:kept_completion_end]

  prompt_start_in_out = max_prompt_length - kept_prompt_len
  out[prompt_start_in_out : prompt_start_in_out + len(prompt_part)] = (
      prompt_part
  )
  out[max_prompt_length : max_prompt_length + len(completion_part)] = (
      completion_part
  )
  return out


def _right_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_value: float | int = 0,
    dtype: Any = np.int32,
) -> tuple[np.ndarray, np.ndarray]:
  arr = np.asarray(values, dtype=dtype).reshape(-1)[:length]
  out = np.full(length, pad_value, dtype=dtype)
  mask = np.zeros(length, dtype=np.float32)
  if arr.size:
    out[: arr.size] = arr
    mask[: arr.size] = 1.0
  return out, mask


def _completion_aligned(
    values: Any | None,
    completion_len: int,
    max_response_length: int,
    *,
    fill_value: float = 0.0,
    prompt_len: int | None = None,
    full_completion_len: int | None = None,
) -> np.ndarray:
  if values is None:
    arr = np.full(completion_len, fill_value, dtype=np.float32)
  else:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1:
      arr = np.full(completion_len, float(arr[0]), dtype=np.float32)
    elif prompt_len is not None and arr.size in (
        prompt_len + (full_completion_len or 0),
        prompt_len + completion_len,
    ):
      # Sequence-aligned `[P + C]` source: slice out the completion span.
      arr = arr[prompt_len:]
    if arr.size >= completion_len:
      arr = arr[:completion_len]
    else:
      arr = np.pad(arr, (0, completion_len - arr.size), constant_values=0.0)
  out, _ = _right_pad(
      arr,
      max_response_length,
      pad_value=0.0,
      dtype=np.float32,
  )
  return out


def with_ref_per_token_logps(
    batch: datatypes.RLTrainerPayload,
    ref_logps: datatypes.LogprobsResponse | np.ndarray,
) -> datatypes.RLTrainerPayload:
  """Returns a trainer batch carrying ref logps aligned to completion_ids."""
  if not isinstance(batch, datatypes.RLTrainerPayload):
    raise TypeError(
        "with_ref_per_token_logps expects a padded RLTrainerPayload from "
        f"BatchAssembler; got {type(batch).__name__}."
    )
  if isinstance(ref_logps, datatypes.LogprobsResponse):
    if ref_logps.error is not None:
      raise RuntimeError(ref_logps.error.message)
    ref_logps = ref_logps.per_token_logps
  ref_logps_arr = np.asarray(ref_logps, dtype=np.float32)
  completion_shape = np.asarray(batch.completion_ids).shape
  if ref_logps_arr.shape != completion_shape:
    raise ValueError(
        "Reference logps shape must match padded completion_ids shape: "
        f"got {ref_logps_arr.shape}, expected {completion_shape}."
    )
  return dataclasses.replace(batch, ref_per_token_logps=ref_logps_arr)


def _as_1d(values: Any, dtype: Any) -> np.ndarray:
  return np.asarray(values, dtype=dtype).reshape(-1)


def _require_unbatched(item: datatypes.RLTrainerPayload) -> None:
  """Requires that the given RLTrainerPayload is unbatched."""
  if item.completion_ids is None:
    raise ValueError(
        "RLTrainerPayload.completion_ids is required for sequence packing."
    )
  for name in (
      "prompt_ids",
      "prompt_mask",
      "completion_ids",
      "completion_mask",
  ):
    value = getattr(item, name)
    if value is None:
      continue
    rank = np.asarray(value).ndim
    if rank != 1:
      raise ValueError(
          f"RLTrainerPayload.{name} has rank {rank}; sequence packing takes"
          " UNBATCHED payloads -- pass the unbatched payloads that produced it."
      )


def to_pack_item(item: datatypes.RLTrainerPayload) -> packing.PackItem:
  """Converts an RLTrainerPayload to a packing.PackItem."""
  _require_unbatched(item)
  prompt = (
      np.zeros(0, dtype=np.int32)
      if item.prompt_ids is None
      else _as_1d(item.prompt_ids, np.int32)
  )
  completion = _as_1d(item.completion_ids, np.int32)
  p_len = prompt.shape[0]
  c_len = completion.shape[0]

  def resolve(values: Any, *, fill: float, name: str) -> np.ndarray:
    if values is None:
      return np.full(c_len, fill, dtype=np.float32)
    arr = _as_1d(values, np.float32)
    if arr.size == 1:
      return np.full(c_len, float(arr[0]), dtype=np.float32)
    if arr.size == p_len + c_len:
      arr = arr[p_len:]
    if arr.size == c_len:
      return arr
    raise ValueError(
        f"RLTrainerPayload.{name} has unexpected size {arr.size} which doesn't"
        f" match either completion length {c_len}; or whole sequence length"
        f" {p_len + c_len}."
    )

  completion_mask = resolve(
      item.completion_mask, fill=1.0, name="completion_mask"
  )

  per_token = {
      name: resolve(getattr(item, name), fill=0.0, name=name)
      for name in packing.PER_TOKEN_FIELDS
      if getattr(item, name) is not None
  }

  routed = None
  if item.routed_experts is not None:
    # The adapter aligns routing to the `[prompt | completion]` sequence
    # (see `algorithm_adapter._routed_experts_for`); `PackItem` re-validates
    # the `(p + c - 1 or p + c, num_layers, top_k)` shape.
    routed = np.asarray(item.routed_experts, dtype=np.int16)

  return packing.PackItem(
      prompt_ids=prompt,
      completion_ids=completion,
      completion_mask=completion_mask,
      advantages=resolve(item.advantages, fill=0.0, name="advantages"),
      per_token=per_token,
      routed_experts=routed,
  )


def to_rl_trainer_payload(
    chunk: packing.PackedChunk,
    *,
    max_segments: int,
    trajectory_ids: tuple[str, ...] = (),
    lineage_context: lineage.LineageContext | None = None,
) -> datatypes.RLTrainerPayload:
  """Converts a packing.PackedChunk to an RLTrainerPayload."""
  n_rows = chunk.ids.shape[0]
  metadata: dict[str, Any] = {"trajectory_ids": trajectory_ids}
  if lineage_context is not None:
    metadata["lineage"] = lineage_context
  return datatypes.RLTrainerPayload(
      prompt_ids=np.zeros((n_rows, 0), dtype=np.int32),
      prompt_mask=np.zeros((n_rows, 0), dtype=np.float32),
      completion_ids=chunk.ids,
      completion_mask=chunk.completion_mask,
      advantages=chunk.advantages,
      segment_ids=chunk.segment_ids,
      segment_positions=chunk.segment_positions,
      num_segments=max_segments + 1,
      # `packing.pack_chunk` pre-allocates one `[B, T, L, K]` buffer for the
      # whole chunk, or none if no item carries routing.
      routed_experts=chunk.routed_experts,
      metadata=metadata,
      **chunk.per_token,  # pyrefly: ignore[bad-argument-type]
  )


def _log_router_replay_coverage(
    payload: datatypes.RLTrainerPayload, *, batch_id: str, num_segments: int
) -> None:
  """Logs how many real packed tokens the trainer will actually replay.

  Judged on layer 0 only so it stays cheap on `[B, T, L, K]` payloads. The MoE
  layer forces a token only if all `top_k` slots are `>= 0` and distinct, and
  re-gates it otherwise. A trajectory's last token has no routing (vLLM routes
  `P + G - 1` positions), so correctly aligned routing forces
  `real - segments` tokens.

  Args:
    payload: Packed payload carrying `routed_experts` and `segment_ids`.
    batch_id: Microbatch tracking id for the log line.
    num_segments: Number of trajectories packed into the payload.
  """
  real = np.asarray(payload.segment_ids) > 0
  layer0 = np.sort(np.asarray(payload.routed_experts)[..., 0, :], axis=-1)
  forced = np.all(layer0 >= 0, axis=-1) & ~np.any(
      layer0[..., 1:] == layer0[..., :-1], axis=-1
  )
  logging.info(
      "Router replay: %d/%d real tokens forced in %s (%d segments; aligned"
      " routing forces real - segments).",
      int(np.count_nonzero(forced & real)),
      int(np.count_nonzero(real)),
      batch_id,
      num_segments,
  )


def _merge_batch_lineage(
    items: Sequence[Any],
    *,
    batch_id: str,
    attributes: Mapping[str, Any] | None = None,
) -> lineage.LineageContext | None:
  """Extracts and merges lineage contexts from a sequence of batch items.

  Args:
    items: Sequence of items that may carry lineage context in their metadata.
    batch_id: Tracking ID to assign to the merged batch context.
    attributes: Optional key-value metadata attached to the merge event.

  Returns:
    The merged LineageContext, or None if no upstream lineage contexts exist.
  """
  lineages = [
      it.metadata["lineage"]
      for it in items
      if isinstance(getattr(it, "metadata", None), Mapping)
      and it.metadata.get("lineage") is not None
  ]
  if not lineages:
    return None

  return lineage.LineageContext.merge(
      batch_id=batch_id,
      contexts=lineages,
      component="orchestrator.assembler",
      operation="pack",
      attributes=dict(attributes) if attributes else None,
  )


class SequencePackedBatchAssembler:
  """Sequence Packing: Concatenates items into dense `[B, max_packed_len]` buffers."""

  def __init__(
      self,
      *,
      batch_size: int,
      num_generations: int,
      mini_batch_size: int,
      max_packed_len: int = 8192,
      pad_id: int = 0,
      max_segments_per_packed_row: int | None = None,
      segment_alignment_boundary: int = (
          packing.DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY
      ),
      start_batch_index: int = 0,
  ):
    """Initializes SequencePackedBatchAssembler.

    Args:
      batch_size: Target batch dim for packed sequences.
      num_generations: Number of rollout generations per prompt group (G).
      mini_batch_size: Number of prompt groups per model update.
      max_packed_len: Maximum packed sequence length per row.
      pad_id: Token ID used for padding.
      max_segments_per_packed_row: Upper bound on the number of real segments
        that may be packed into a single row.
      segment_alignment_boundary: Enforces that every packed sequence starts at
        an index that is a multiple of this value, padding any gaps. By default
        (1), sequences are packed tightly back-to-back. Use larger values to
        align with chunk boundaries for models with chunked processing (e.g.,
        64 for Qwen3.5 GatedDeltaNet).
      start_batch_index: Initial microbatch index offset for tracking IDs.
    """
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_generations <= 0:
      raise ValueError(f"num_generations must be positive, got {num_generations}.")
    if mini_batch_size <= 0:
      raise ValueError(
          f"mini_batch_size must be positive, got {mini_batch_size}."
      )
    if max_packed_len <= 0:
      raise ValueError(f"max_packed_len must be positive, got {max_packed_len}")
    if (
        max_segments_per_packed_row is not None
        and max_segments_per_packed_row <= 0
    ):
      raise ValueError(
          "max_segments_per_packed_row must be positive or None, got"
          f" {max_segments_per_packed_row}."
      )
    if segment_alignment_boundary <= 0:
      raise ValueError(
          "segment_alignment_boundary must be positive, got"
          f" {segment_alignment_boundary}."
      )
    self.batch_size = batch_size
    self.max_packed_len = max_packed_len
    self.pad_id = pad_id
    self.num_generations = num_generations
    self.mini_batch_size = mini_batch_size
    self.max_segments_per_packed_row = max_segments_per_packed_row
    self.segment_alignment_boundary = segment_alignment_boundary
    self._batch_counter = start_batch_index

    # Each entry is a `(PackItem, trajectory_id, raw_payload)` converted once at ingest.
    self._buffer: list[
        tuple[packing.PackItem, str, datatypes.RLTrainerPayload]
    ] = []
    self._rollouts_since_update: int = 0

  @property
  def rollouts_per_optimizer_update(self) -> int:
    """Total number of rollouts expected per optimizer update."""
    return self.mini_batch_size * self.num_generations

  def _emit_one_chunk(
      self, *, max_segments: int, drain_all: bool
  ) -> AssembledBatch:
    """Packs the head of the buffer into one microbatch, keeping leftovers."""
    pack_items = [item for item, _, _ in self._buffer]
    carried = packing.carried_per_token_fields(pack_items)
    id_to_entry = {
        id(item): entry for entry, item in zip(self._buffer, pack_items)
    }
    bins, leftover = packing.fill_one_chunk(
        pack_items,
        pack_size=self.batch_size,
        budget=self.max_packed_len,
        max_segments=max_segments,
        segment_alignment_boundary=self.segment_alignment_boundary,
    )
    placed = []
    for bin_items in bins:
      placed.extend(bin_items)
    traj_ids = tuple(id_to_entry[id(item)][1] for item in placed)
    placed_items = [id_to_entry[id(item)][2] for item in placed]
    num_unrouted = sum(item.routed_experts is None for item in placed)
    if 0 < num_unrouted < len(placed):
      logging.warning(
          "Router replay: %d of %d packed trajectories carry no"
          " routed_experts; their tokens use the trainer's own gate.",
          num_unrouted,
          len(placed),
      )
    chunk = packing.pack_chunk(
        bins,
        budget=self.max_packed_len,
        pad_id=self.pad_id,
        carried=carried,
        segment_alignment_boundary=self.segment_alignment_boundary,
    )
    batch_tracking_id = f"{_BATCH_ID_PREFIX}_{self._batch_counter}"
    merged_lineage = _merge_batch_lineage(
        placed_items,
        batch_id=batch_tracking_id,
        attributes={
            "packing_type": "sequence_packed",
            "num_items": len(placed_items),
            "packed_len": self.max_packed_len,
        },
    )
    self._batch_counter += 1
    payload = to_rl_trainer_payload(
        chunk,
        max_segments=max_segments,
        trajectory_ids=traj_ids,
        lineage_context=merged_lineage,
    )
    if payload.routed_experts is not None:
      _log_router_replay_coverage(
          payload, batch_id=batch_tracking_id, num_segments=len(placed)
      )
    self._buffer = [id_to_entry[id(item)] for item in leftover]
    padding_stats = PaddingStats(
        row_valid_tokens=np.array(
            [sum(item.num_tokens for item in b) for b in bins], dtype=np.int64
        ),
        row_num_sequences=np.array([len(b) for b in bins], dtype=np.int64),
        row_capacity=self.max_packed_len,
    )
    return AssembledBatch(
        payload=payload,
        is_final_batch=drain_all and not self._buffer,
        padding_stats=padding_stats,
        trajectory_ids=traj_ids,
    )

  def _drain_buffer(self, *, drain_all: bool) -> Iterator[AssembledBatch]:
    """Drains buffered items into microbatches using FFD packing.

    When `drain_all` is False, only whole chunks whose token mass can fill a
    full microbatch are emitted, so the streaming tail is held back until more
    rollouts arrive. When `drain_all` is True (optimizer-update boundary or
    `flush`), the buffer is drained completely and the last chunk is marked
    final.

    Args:
      drain_all: Whether to flush all buffered items regardless of chunk
        capacity.

    Yields:
      Assembled microbatches packed one chunk at a time.
    """
    max_segments = packing.effective_max_segments(
        self.max_packed_len, self.max_segments_per_packed_row
    )
    chunk_capacity = self.batch_size * self.max_packed_len
    while self._buffer:
      if not drain_all:
        buffered_tokens = sum(item[0].num_tokens for item in self._buffer)
        if buffered_tokens < chunk_capacity:
          break
      yield self._emit_one_chunk(max_segments=max_segments, drain_all=drain_all)

  def feed(
      self,
      items: Sequence[datatypes.RLTrainerPayload],
  ) -> Iterator[AssembledBatch]:
    """Ingests items and yields microbatches, flushing at the update boundary."""
    for item in items:
      pack_item = to_pack_item(item)
      packing.validate_items([pack_item], self.max_packed_len)
      self._buffer.append((pack_item, _extract_trajectory_id(item), item))
    self._rollouts_since_update += len(items)
    is_update_done = (
        self._rollouts_since_update >= self.rollouts_per_optimizer_update
    )
    if is_update_done:
      self._rollouts_since_update %= self.rollouts_per_optimizer_update
    return self._drain_buffer(drain_all=is_update_done)

  def flush(
      self,
  ) -> Iterator[AssembledBatch]:
    """Flushes any remaining buffered items, marking the last chunk final."""
    self._rollouts_since_update = 0
    return self._drain_buffer(drain_all=True)

  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Resets the internal buffer and optimizer-update rollout counter.

    Unlike `flush()`, which emits remaining items as padded microbatches,
    `reset()` unconditionally drops any partially accumulated items or bins
    without packing or emitting them, and resets the optimizer-update rollout
    counter back to zero.

    This is typically invoked during pipeline aborts or error recovery (e.g.,
    in `RLProgram` when a stage encounters an exception and in-flight
    rollouts must be dropped to avoid cross-update contamination) or when
    restarting the assembler.

    Args:
      start_batch_index: Optional batch index to reset the microbatch lineage
        tracking counter (e.g., when resuming from a checkpoint). If None, the
        existing `_batch_counter` is preserved to maintain monotonic lineage
        tracking IDs across step boundaries.
    """
    self._buffer.clear()
    self._rollouts_since_update = 0
    if start_batch_index is not None:
      self._batch_counter = start_batch_index


class PaddedBatchAssembler:
  """Simple 2D rectangular batching into fixed `[B, P + C]` trainer payloads."""

  def __init__(
      self,
      *,
      batch_size: int,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
      num_generations: int,
      mini_batch_size: int,
      start_batch_index: int = 0,
  ):
    """Initializes PaddedBatchAssembler.

    Args:
      batch_size: Hardware train microbatch size (number of sequences per
        batch).
      max_prompt_length: Maximum padded prompt sequence length.
      max_response_length: Maximum padded response sequence length.
      pad_id: Token ID used for padding prompts and completions.
      num_generations: Number of rollout generations per prompt group (G).
      mini_batch_size: Number of prompt groups per optimizer update.
      start_batch_index: Initial microbatch index offset for tracking IDs.
    """
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if max_prompt_length <= 0:
      raise ValueError(
          f"max_prompt_length must be positive, got {max_prompt_length}."
      )
    if max_response_length <= 0:
      raise ValueError(
          f"max_response_length must be positive, got {max_response_length}."
      )
    if num_generations <= 0:
      raise ValueError(f"num_generations must be positive, got {num_generations}.")
    if mini_batch_size <= 0:
      raise ValueError(
          f"mini_batch_size must be positive, got {mini_batch_size}."
      )
    self.batch_size = batch_size
    self.max_prompt_length = max_prompt_length
    self.max_response_length = max_response_length
    self.pad_id = pad_id
    self.num_generations = num_generations
    self.mini_batch_size = mini_batch_size
    self._batch_counter = start_batch_index

    self._buffer: collections.deque[datatypes.RLTrainerPayload] = (
        collections.deque()
    )
    self._rollouts_since_update: int = 0

  @property
  def rollouts_per_optimizer_update(self) -> int:
    """Total number of rollouts expected per optimizer update."""
    return self.mini_batch_size * self.num_generations

  @property
  def max_seq_len(self) -> int:
    return self.max_prompt_length + self.max_response_length

  def _emit_one_chunk(self, *, drain_all: bool) -> AssembledBatch:
    """Pops up to `batch_size` items from the buffer and pads one microbatch."""
    chunk = [
        self._buffer.popleft()
        for _ in range(min(self.batch_size, len(self._buffer)))
    ]
    payload, padding_stats = self._pack_chunk(chunk)
    return AssembledBatch(
        payload=payload,
        is_final_batch=drain_all and not self._buffer,
        padding_stats=padding_stats,
        trajectory_ids=tuple(_extract_trajectory_id(it) for it in chunk),
    )

  def _drain_buffer(self, *, drain_all: bool) -> Iterator[AssembledBatch]:
    """Drains buffered items into padded microbatches one chunk at a time."""
    while len(self._buffer) >= self.batch_size or (drain_all and self._buffer):
      yield self._emit_one_chunk(drain_all=drain_all)

  def feed(
      self,
      items: Sequence[datatypes.RLTrainerPayload],
  ) -> Iterator[AssembledBatch]:
    """Ingests items and yields microbatches, flushing at the update boundary."""
    self._buffer.extend(items)
    self._rollouts_since_update += len(items)
    is_update_done = (
        self._rollouts_since_update >= self.rollouts_per_optimizer_update
    )
    if is_update_done:
      self._rollouts_since_update %= self.rollouts_per_optimizer_update
    return self._drain_buffer(drain_all=is_update_done)

  def flush(
      self,
  ) -> Iterator[AssembledBatch]:
    """Flushes any remaining items padded to batch_size."""
    self._rollouts_since_update = 0
    return self._drain_buffer(drain_all=True)

  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Resets internal buffering state, discarding all pending rollouts.

    Unlike `flush()`, which packs and emits buffered items as a padded batch,
    `reset()` unconditionally clears the internal rollout buffer without
    emitting any batches. It also resets the optimizer-update rollout counter
    (`_rollouts_since_update`) back to zero.

    This is typically invoked during pipeline aborts or error recovery (e.g.,
    in `RLProgram` when a stage encounters an exception and in-flight
    rollouts must be dropped to avoid cross-update contamination) or when
    restarting the assembler.

    Args:
      start_batch_index: Optional batch index to reset the microbatch lineage
        tracking counter (e.g., when resuming from a checkpoint). If None, the
        existing `_batch_counter` is preserved to maintain monotonic lineage
        tracking IDs across step boundaries.
    """
    self._buffer.clear()
    self._rollouts_since_update = 0
    if start_batch_index is not None:
      self._batch_counter = start_batch_index

  def pack(
      self,
      items: Sequence[datatypes.RLTrainerPayload],
  ) -> list[datatypes.RLTrainerPayload]:
    """Pads items into rectangular 2D batches `[B, P + C]`."""
    item_list = list(items)
    if not item_list:
      return []

    payloads: list[datatypes.RLTrainerPayload] = []
    for i in range(0, len(item_list), self.batch_size):
      chunk = item_list[i : i + self.batch_size]
      payload, _ = self._pack_chunk(chunk)
      payloads.append(payload)
    return payloads

  def _pack_chunk(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> tuple[datatypes.RLTrainerPayload, PaddingStats]:
    """Pads a single `<= batch_size` chunk into one rectangular payload.

    Args:
      chunk: At most `batch_size` unbatched payloads.

    Returns:
      The padded `[batch_size, ...]` payload and its per-row padding stats,
      recorded from the same truncated lengths used to fill the rows. Filler
      rows past `len(chunk)` count as empty.
    """
    if len(chunk) > self.batch_size:
      raise ValueError(
          f"Chunk of {len(chunk)} items exceeds batch_size {self.batch_size}."
      )
    # Optional per-token fields are emitted for the whole batch only when all
    # rows carry them.
    optional_fields = (
        "ref_per_token_logps",
        "old_per_token_logps",
        "returns",
        "old_values",
        "sampler_is_weights",
    )
    present_fields = []
    partially_present_fields = []
    for name in optional_fields:
      num_present = sum(getattr(it, name) is not None for it in chunk)
      if num_present == len(chunk):
        present_fields.append(name)
      elif num_present > 0:
        partially_present_fields.append(name)

    if partially_present_fields:
      logging.warning(
          "Partially present optional fields: %s",
          partially_present_fields,
      )

    batched_prompt_ids = np.full(
        (self.batch_size, self.max_prompt_length), self.pad_id, dtype=np.int32
    )
    batched_prompt_mask = np.zeros(
        (self.batch_size, self.max_prompt_length), dtype=np.float32
    )
    batched_completion_ids = np.full(
        (self.batch_size, self.max_response_length), self.pad_id, dtype=np.int32
    )
    batched_completion_mask = np.zeros(
        (self.batch_size, self.max_response_length), dtype=np.float32
    )
    batched_advantages = np.zeros(
        (self.batch_size, self.max_response_length), dtype=np.float32
    )
    stacked_optional: dict[str, np.ndarray] = {
        name: np.zeros(
            (self.batch_size, self.max_response_length), dtype=np.float32
        )
        for name in present_fields
    }
    # Router replay is all-or-nothing per batch: a partially replayed batch
    # would silently mix replayed and freshly routed rows.
    replay_routing = bool(chunk) and all(
        it.routed_experts is not None for it in chunk
    )
    batched_routed_experts: np.ndarray | None = None
    truncated_prompts = truncated_completions = 0
    row_valid_tokens = np.zeros(self.batch_size, dtype=np.int64)
    row_num_sequences = np.zeros(self.batch_size, dtype=np.int64)

    for row_idx, item in enumerate(chunk):
      p_full = np.asarray(item.prompt_ids, dtype=np.int32).reshape(-1)
      c_full = np.asarray(item.completion_ids, dtype=np.int32).reshape(-1)
      truncated_prompts += p_full.size > self.max_prompt_length
      truncated_completions += c_full.size > self.max_response_length
      c = c_full[: self.max_response_length]
      row_valid_tokens[row_idx] = (
          min(p_full.size, self.max_prompt_length) + c.size
      )
      row_num_sequences[row_idx] = 1

      p_ids, p_default_mask = _left_pad(
          p_full, self.max_prompt_length, pad_id=self.pad_id
      )
      c_ids, c_valid = _right_pad(
          c, self.max_response_length, pad_value=self.pad_id, dtype=np.int32
      )
      batched_prompt_ids[row_idx] = p_ids
      batched_completion_ids[row_idx] = c_ids

      # A caller-supplied prompt mask is prompt-aligned, so it must be
      # left-padded exactly like the prompt ids to stay in register. If its
      # length disagrees with the prompt the alignment is undefined, so fall
      # back to the validity mask derived from the ids themselves.
      p_mask = p_default_mask
      if item.prompt_mask is not None:
        src = np.asarray(item.prompt_mask, dtype=np.float32).reshape(-1)
        if src.size == p_full.size:
          src = src[-self.max_prompt_length :]
          p_mask = np.zeros(self.max_prompt_length, dtype=np.float32)
          if src.size:
            p_mask[-src.size :] = src
      batched_prompt_mask[row_idx] = p_mask

      action_source = item.completion_mask
      if action_source is None:
        c_mask = c_valid
      else:
        c_mask = _completion_aligned(
            action_source,
            c.size,
            self.max_response_length,
            prompt_len=p_full.size,
            full_completion_len=c_full.size,
        )
      batched_completion_mask[row_idx] = c_mask

      batched_advantages[row_idx] = _completion_aligned(
          item.advantages,
          c.size,
          self.max_response_length,
          fill_value=0.0,
          prompt_len=p_full.size,
          full_completion_len=c_full.size,
      )

      for name, batch_arr in stacked_optional.items():
        batch_arr[row_idx] = _completion_aligned(
            getattr(item, name),
            c.size,
            self.max_response_length,
            fill_value=0.0,
            prompt_len=p_full.size,
            full_completion_len=c_full.size,
        )

      # `replay_routing` already guarantees this is set; binding it locally
      # also narrows the optional field for the type checker.
      routed = item.routed_experts
      if replay_routing and routed is not None:
        routed_arr = np.asarray(routed, dtype=np.int16)
        if batched_routed_experts is None:
          batched_routed_experts = np.full(
              (
                  self.batch_size,
                  self.max_prompt_length + self.max_response_length,
              )
              + routed_arr.shape[1:],
              datatypes.UNSET_ROUTED_EXPERT,
              dtype=np.int16,
          )
        _routed_experts_aligned(
            routed_arr,
            p_full.size,
            c.size,
            self.max_prompt_length,
            self.max_response_length,
            out=batched_routed_experts[row_idx],
        )

    if truncated_prompts or truncated_completions:
      logging.warning(
          "PaddedBatchAssembler truncated %d prompt(s) to %d tokens and %d "
          "completion(s) to %d tokens; raise max_prompt_length / "
          "max_response_length to avoid dropping training signal.",
          truncated_prompts,
          self.max_prompt_length,
          truncated_completions,
          self.max_response_length,
      )

    batch_tracking_id = f"{_BATCH_ID_PREFIX}_{self._batch_counter}"
    merged_lineage = _merge_batch_lineage(
        chunk,
        batch_id=batch_tracking_id,
        attributes={
            "packing_type": "padded",
            "num_items": len(chunk),
            "batch_size": self.batch_size,
        },
    )
    self._batch_counter += 1
    payload_metadata: dict[str, Any] = {
        "trajectory_ids": tuple(_extract_trajectory_id(it) for it in chunk)
    }
    if merged_lineage:
      payload_metadata["lineage"] = merged_lineage

    payload = datatypes.RLTrainerPayload(
        advantages=batched_advantages,
        prompt_ids=batched_prompt_ids,
        prompt_mask=batched_prompt_mask,
        completion_ids=batched_completion_ids,
        completion_mask=batched_completion_mask,
        ref_per_token_logps=(
            stacked_optional["ref_per_token_logps"]
            if "ref_per_token_logps" in stacked_optional
            else None
        ),
        old_per_token_logps=(
            stacked_optional["old_per_token_logps"]
            if "old_per_token_logps" in stacked_optional
            else None
        ),
        returns=(
            stacked_optional["returns"]
            if "returns" in stacked_optional
            else None
        ),
        old_values=(
            stacked_optional["old_values"]
            if "old_values" in stacked_optional
            else None
        ),
        sampler_is_weights=(
            stacked_optional["sampler_is_weights"]
            if "sampler_is_weights" in stacked_optional
            else None
        ),
        routed_experts=batched_routed_experts,
        metadata=payload_metadata,
    )
    padding_stats = PaddingStats(
        row_valid_tokens=row_valid_tokens,
        row_num_sequences=row_num_sequences,
        row_capacity=self.max_seq_len,
    )
    return payload, padding_stats


def create_batch_assembler(
    *,
    num_generations: int,
    mini_batch_size: int,
    train_micro_batch_size: int,
    batch_config: BatchConfig,
) -> BatchAssembler:
  """Builds the batch assembler based on sequence packing or padding parameters.

  If `batch_config.max_seq_token_per_tpu` is provided, a
  `SequencePackedBatchAssembler` is used. The packing `batch_size` (pack_size)
  is computed from `batch_config.trainer_fsdp` and `batch_config.trainer_dp`
  (or defaults to `train_micro_batch_size` with a warning if neither is set).
  The packing budget is validated against `batch_config.max_prompt_length` and
  `max_response_length`.

  If `batch_config.max_seq_token_per_tpu` is None and
  `batch_config.max_prompt_length` is specified, a `PaddedBatchAssembler` is
  used.

  Otherwise, falls back to `SequencePackedBatchAssembler`.

  Args:
    num_generations: Number of rollout generations per prompt group (G).
    mini_batch_size: Number of prompt groups per model update.
    train_micro_batch_size: Micro-batch size for training.
    batch_config: BatchConfig containing packing, padding, and mesh dimension
      settings.

  Returns:
    A BatchAssembler instance.
  """
  if batch_config.max_seq_token_per_tpu is not None:
    if batch_config.trainer_fsdp is None and batch_config.trainer_dp is None:
      logging.warning(
          "trainer_fsdp and trainer_dp are not set, defaulting pack_size to "
          "train_micro_batch_size=%d.",
          train_micro_batch_size,
      )
      pack_size = train_micro_batch_size
    else:
      pack_size = (batch_config.trainer_fsdp or 1) * (
          batch_config.trainer_dp or 1
      )

    if (
        batch_config.max_prompt_length is not None
        and batch_config.max_response_length is not None
    ):
      rl_utils.validate_packing_budget(
          batch_config.max_seq_token_per_tpu,
          batch_config.max_prompt_length,
          batch_config.max_response_length,
      )

    logging.info(
        "Using SequencePackedBatchAssembler with max_seq_token_per_tpu: %d, "
        "max_segments_per_packed_row: %s, segment_alignment_boundary: %d, "
        "pack_size: %d",
        batch_config.max_seq_token_per_tpu,
        batch_config.max_segments_per_packed_row,
        batch_config.segment_alignment_boundary,
        pack_size,
    )
    return SequencePackedBatchAssembler(
        batch_size=pack_size,
        num_generations=num_generations,
        mini_batch_size=mini_batch_size,
        max_packed_len=batch_config.max_seq_token_per_tpu,
        pad_id=batch_config.pad_id,
        max_segments_per_packed_row=batch_config.max_segments_per_packed_row,
        segment_alignment_boundary=batch_config.segment_alignment_boundary,
    )

  if batch_config.max_prompt_length is not None:
    if batch_config.max_response_length is None:
      raise ValueError(
          "max_response_length must be specified in batch_config when"
          " max_prompt_length is set for PaddedBatchAssembler."
      )
    return PaddedBatchAssembler(
        batch_size=train_micro_batch_size,
        max_prompt_length=batch_config.max_prompt_length,
        max_response_length=batch_config.max_response_length,
        pad_id=batch_config.pad_id,
        num_generations=num_generations,
        mini_batch_size=mini_batch_size,
    )

  return SequencePackedBatchAssembler(
      batch_size=train_micro_batch_size,
      num_generations=num_generations,
      mini_batch_size=mini_batch_size,
      pad_id=batch_config.pad_id,
      max_segments_per_packed_row=batch_config.max_segments_per_packed_row,
      segment_alignment_boundary=batch_config.segment_alignment_boundary,
  )

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
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Generic, NamedTuple, Protocol, TypeVar
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage

T = TypeVar("T")

_BATCH_ID_PREFIX: str = "batch"


class AssembledBatch(NamedTuple):
  """Microbatch payload paired with global step completion status."""

  payload: datatypes.RLTrainerPayload
  is_final_batch: bool
  trajectory_ids: tuple[str, ...] = ()


def _extract_trajectory_id(item: Any) -> str:
  """Extracts the standardized trajectory id from payload metadata."""
  metadata = getattr(item, "metadata", None) or {}
  return str(metadata.get("traj_id", ""))


class BatchAssembler(Generic[T], Protocol):
  """Universal batch assembly protocol for microbatch packing.

  Attributes:
    group_size: Number of rollout trajectories / generations generated per
      prompt group (G). Must be a positive integer.
  """

  group_size: int
  mini_batch_size: int

  @property
  def total_step_rollouts(self) -> int:
    """Total number of rollouts expected per global training step."""
    return self.mini_batch_size * self.group_size

  def feed(
      self,
      items: Sequence[T],
  ) -> list[AssembledBatch]:
    """Ingests rollouts, emitting ready microbatches and auto-flushing at step end."""
    ...

  def flush(
      self,
  ) -> list[AssembledBatch]:
    """Drains remaining buffered items, padding to the required static tensor shape."""
    ...

  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Clears buffered items and resets the step rollouts counter."""
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
) -> np.ndarray:
  """Lays one row of routing out over the padded `[prompt | completion]`.

  Mirrors how the token ids themselves are padded -- prompts right-aligned in
  the prompt window (keeping the tail, as `_left_pad` does) and completions
  left-aligned in the response window -- so replayed routing stays attached to
  the token it was captured for. Everything else is left unset.

  Args:
    routed: `[prompt_len + completion_len, num_layers, top_k]` for one
      generation. Only axis 0 (the per-token axis) is ever sliced below; the
      trailing `[num_layers, top_k]` axes are carried through untouched.
    prompt_len: Unpadded prompt length, i.e. where the completion starts.
    completion_len: Unpadded completion length.
    max_prompt_length: Padded prompt width.
    max_response_length: Padded completion width.

  Returns:
    `[max_prompt_length + max_response_length, num_layers, top_k]`.
  """
  routed = np.asarray(routed, dtype=np.int32)
  # Prompts are left-padded, so an over-long one keeps its tail; completions are
  # right-padded, so an over-long one keeps its head.
  kept_prompt_start = max(prompt_len - max_prompt_length, 0)
  kept_completion_end = prompt_len + min(completion_len, max_response_length)
  prompt_part = routed[kept_prompt_start:prompt_len]
  completion_part = routed[prompt_len:kept_completion_end]

  out = np.full(
      (max_prompt_length + max_response_length,) + routed.shape[1:],
      datatypes.UNSET_ROUTED_EXPERT,
      dtype=np.int32,
  )
  prompt_end = max_prompt_length
  out[prompt_end - len(prompt_part) : prompt_end] = prompt_part
  out[prompt_end : prompt_end + len(completion_part)] = completion_part
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
      group_size: int,
      mini_batch_size: int,
      max_packed_len: int = 8192,
      pad_id: int = 0,
      target_occupancy: float = 0.90,
      start_batch_index: int = 0,
  ):
    """Initializes SequencePackedBatchAssembler.

    Args:
      batch_size: Target batch dim for packed sequences.
      group_size: Number of rollout generations per prompt group (G).
      mini_batch_size: Number of prompt groups per model update.
      max_packed_len: Maximum packed sequence length per row.
      pad_id: Token ID used for padding.
      target_occupancy: Occupancy ratio above which a bin is sealed before full.
      start_batch_index: Initial microbatch index offset for tracking IDs.
    """
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if group_size <= 0:
      raise ValueError(f"group_size must be positive, got {group_size}.")
    if mini_batch_size <= 0:
      raise ValueError(
          f"mini_batch_size must be positive, got {mini_batch_size}."
      )
    if max_packed_len <= 0:
      raise ValueError(
          f"max_packed_len must be positive, got {max_packed_len}."
      )
    self.batch_size = batch_size
    self.max_packed_len = max_packed_len
    self.pad_id = pad_id
    self.group_size = group_size
    self.mini_batch_size = mini_batch_size
    self.target_occupancy = target_occupancy
    self._batch_counter = start_batch_index

    self._current_bin: list[datatypes.RLTrainerPayload] = []
    self._current_len: int = 0
    self._sealed_bins: list[list[datatypes.RLTrainerPayload]] = []
    self._step_rollouts: int = 0

  @property
  def total_step_rollouts(self) -> int:
    """Total number of rollouts expected per global training step."""
    return self.mini_batch_size * self.group_size

  def _seal_row(
      self, b_items: Sequence[datatypes.RLTrainerPayload]
  ) -> dict[str, Any]:
    all_tokens = []
    all_loss_masks = []
    all_action_masks = []
    all_segment_ids = []
    all_segment_positions = []
    all_advantages = []
    all_old_logprobs = []
    all_ref_logprobs = []

    for seg_idx, it in enumerate(b_items, start=1):
      toks = (
          np.asarray(it.token_ids, dtype=np.int32).reshape(-1)
          if it.token_ids is not None
          else np.zeros(0, dtype=np.int32)
      )
      seq_len = len(toks)
      all_tokens.append(toks)

      loss_mask = (
          it.loss_mask
          if it.loss_mask is not None
          else np.zeros(seq_len, dtype=np.float32)
      )
      all_loss_masks.append(np.asarray(loss_mask, dtype=np.float32).reshape(-1))

      action_mask = (
          it.action_mask
          if it.action_mask is not None
          else np.zeros(seq_len, dtype=np.float32)
      )
      all_action_masks.append(
          np.asarray(action_mask, dtype=np.float32).reshape(-1)
      )

      adv_arr = (
          np.asarray(it.advantages, dtype=np.float32).reshape(-1)
          if it.advantages is not None
          else np.zeros(seq_len, dtype=np.float32)
      )
      all_advantages.append(adv_arr)

      all_segment_ids.append(np.full(seq_len, seg_idx, dtype=np.int32))
      all_segment_positions.append(np.arange(seq_len, dtype=np.int32))

      if it.old_per_token_logps is not None:
        all_old_logprobs.append(
            np.asarray(it.old_per_token_logps, dtype=np.float32).reshape(-1)
        )

      if it.ref_per_token_logps is not None:
        all_ref_logprobs.append(
            np.asarray(it.ref_per_token_logps, dtype=np.float32).reshape(-1)
        )

    _cat = lambda s, dt: np.concatenate(s) if s else np.zeros(0, dtype=dt)
    _pad = lambda arr, val, dt: np.pad(
        arr[: self.max_packed_len],
        (0, max(0, self.max_packed_len - arr.size)),
        constant_values=val,
    ).astype(dt)

    row_old_lp = (
        _pad(_cat(all_old_logprobs, np.float32), 0.0, np.float32)
        if all_old_logprobs
        else None
    )
    row_ref_lp = (
        _pad(_cat(all_ref_logprobs, np.float32), 0.0, np.float32)
        if all_ref_logprobs
        else None
    )

    return {
        "tokens": _pad(_cat(all_tokens, np.int32), self.pad_id, np.int32),
        "loss_mask": _pad(_cat(all_loss_masks, np.float32), 0.0, np.float32),
        "action_mask": _pad(
            _cat(all_action_masks, np.float32), 0.0, np.float32
        ),
        "segment_ids": _pad(_cat(all_segment_ids, np.int32), 0, np.int32),
        "segment_positions": _pad(
            _cat(all_segment_positions, np.int32), 0, np.int32
        ),
        "advantages": _pad(_cat(all_advantages, np.float32), 0.0, np.float32),
        "old_logprobs": row_old_lp,
        "ref_logprobs": row_ref_lp,
        "traj_ids": tuple(_extract_trajectory_id(it) for it in b_items),
    }

  def _seal_batch(
      self,
      chunk_of_bins: Sequence[Sequence[datatypes.RLTrainerPayload]],
  ) -> tuple[datatypes.RLTrainerPayload, tuple[str, ...]]:
    rows = [self._seal_row(b) for b in chunk_of_bins]
    any_old_lp = any(r["old_logprobs"] is not None for r in rows)
    any_ref_lp = any(r["ref_logprobs"] is not None for r in rows)

    while len(rows) < self.batch_size:
      rows.append({
          "tokens": np.full(self.max_packed_len, self.pad_id, dtype=np.int32),
          "loss_mask": np.zeros(self.max_packed_len, dtype=np.float32),
          "action_mask": np.zeros(self.max_packed_len, dtype=np.float32),
          "segment_ids": np.zeros(self.max_packed_len, dtype=np.int32),
          "segment_positions": np.zeros(self.max_packed_len, dtype=np.int32),
          "advantages": np.zeros(self.max_packed_len, dtype=np.float32),
          "old_logprobs": None,
          "ref_logprobs": None,
          "traj_ids": (),
      })

    _stack = lambda key: np.stack([r[key] for r in rows], axis=0)

    batch_old_lp = (
        np.stack(
            [
                r["old_logprobs"]
                if r["old_logprobs"] is not None
                else np.zeros(self.max_packed_len, dtype=np.float32)
                for r in rows
            ],
            axis=0,
        )
        if any_old_lp
        else None
    )
    batch_ref_lp = (
        np.stack(
            [
                r["ref_logprobs"]
                if r["ref_logprobs"] is not None
                else np.zeros(self.max_packed_len, dtype=np.float32)
                for r in rows
            ],
            axis=0,
        )
        if any_ref_lp
        else None
    )

    all_traj_ids = tuple(t for r in rows for t in r["traj_ids"])
    batch_tracking_id = f"{_BATCH_ID_PREFIX}_{self._batch_counter}"
    all_items = [it for b in chunk_of_bins for it in b]
    merged_lineage = _merge_batch_lineage(
        all_items,
        batch_id=batch_tracking_id,
        attributes={
            "packing_type": "sequence_packed",
            "num_items": len(all_items),
            "packed_len": self.max_packed_len,
        },
    )
    # TODO (tunix-dev): consolidate trajectory_ids from metadata and lineage.
    payload_metadata: dict[str, Any] = {"trajectory_ids": all_traj_ids}
    if merged_lineage:
      payload_metadata["lineage"] = merged_lineage
    self._batch_counter += 1

    payload = datatypes.RLTrainerPayload(
        token_ids=_stack("tokens"),
        token_mask=_stack("segment_ids"),
        loss_mask=_stack("loss_mask"),
        advantages=_stack("advantages"),
        action_mask=_stack("action_mask"),
        old_per_token_logps=batch_old_lp,
        ref_per_token_logps=batch_ref_lp,
        segment_ids=_stack("segment_ids"),
        segment_positions=_stack("segment_positions"),
        metadata=payload_metadata,
    )
    return payload, all_traj_ids

  def _seal_current_bin(self) -> None:
    if self._current_bin:
      self._sealed_bins.append(self._current_bin)
      self._current_bin, self._current_len = [], 0

  def _pop_batches(
      self,
      *,
      drain_all: bool = False,
  ) -> list[AssembledBatch]:
    out: list[AssembledBatch] = []
    while len(self._sealed_bins) >= self.batch_size or (
        drain_all and self._sealed_bins
    ):
      chunk = self._sealed_bins[: self.batch_size]
      self._sealed_bins = self._sealed_bins[self.batch_size :]
      payload, traj_ids = self._seal_batch(chunk)
      out.append(
          AssembledBatch(
              payload=payload,
              is_final_batch=drain_all and not self._sealed_bins,
              trajectory_ids=traj_ids,
          )
      )
    return out

  def feed(
      self,
      items: Sequence[datatypes.RLTrainerPayload],
  ) -> list[AssembledBatch]:
    """Ingests items into dense bins, auto-flushing on the step boundary."""
    self._step_rollouts += len(items)
    is_step_done = self._step_rollouts >= self.total_step_rollouts

    out: list[AssembledBatch] = []

    for it in items:
      it_len = int(np.size(it.token_ids)) if it.token_ids is not None else 0
      if self._current_len + it_len > self.max_packed_len:
        self._seal_current_bin()
        out.extend(self._pop_batches())
        self._current_bin = [it]
        self._current_len = min(it_len, self.max_packed_len)
      else:
        self._current_bin.append(it)
        self._current_len += it_len
        if (
            self._current_len >= self.target_occupancy * self.max_packed_len
            and not is_step_done
        ):
          self._seal_current_bin()
          out.extend(self._pop_batches())

    if is_step_done:
      self._seal_current_bin()
      out.extend(self._pop_batches(drain_all=True))
      if out and not out[-1].is_final_batch:
        out[-1] = AssembledBatch(
            payload=out[-1].payload,
            is_final_batch=True,
            trajectory_ids=out[-1].trajectory_ids,
        )
      self._step_rollouts %= self.total_step_rollouts

    return out

  def flush(
      self,
  ) -> list[AssembledBatch]:
    """Flushes any remaining open bins padded to batch_size."""
    self._seal_current_bin()
    self._step_rollouts = 0
    return self._pop_batches(drain_all=True)

  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Clears internal bins and resets step rollouts counter."""
    self._current_bin = []
    self._current_len = 0
    self._sealed_bins = []
    self._step_rollouts = 0
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
      group_size: int,
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
      group_size: Number of rollout generations per prompt group (G).
      mini_batch_size: Number of prompt groups per global training step.
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
    if group_size <= 0:
      raise ValueError(f"group_size must be positive, got {group_size}.")
    if mini_batch_size <= 0:
      raise ValueError(
          f"mini_batch_size must be positive, got {mini_batch_size}."
      )
    self.batch_size = batch_size
    self.max_prompt_length = max_prompt_length
    self.max_response_length = max_response_length
    self.pad_id = pad_id
    self.group_size = group_size
    self.mini_batch_size = mini_batch_size
    self._batch_counter = start_batch_index

    self._buffer: collections.deque[datatypes.RLTrainerPayload] = (
        collections.deque()
    )
    self._step_rollouts: int = 0

  @property
  def total_step_rollouts(self) -> int:
    """Total number of rollouts expected per global training step."""
    return self.mini_batch_size * self.group_size

  @property
  def max_seq_len(self) -> int:
    return self.max_prompt_length + self.max_response_length

  def feed(
      self,
      items: Sequence[datatypes.RLTrainerPayload],
  ) -> list[AssembledBatch]:
    """Ingests items, emitting full microbatches and auto-flushing at step end."""
    self._buffer.extend(items)
    self._step_rollouts += len(items)

    out: list[AssembledBatch] = []

    while len(self._buffer) >= self.batch_size:
      is_step_done = self._step_rollouts >= self.total_step_rollouts
      will_be_empty = len(self._buffer) == self.batch_size
      is_final = is_step_done and will_be_empty

      chunk = [self._buffer.popleft() for _ in range(self.batch_size)]
      traj_ids = tuple(_extract_trajectory_id(it) for it in chunk)
      payload = self.pack(chunk)[0]
      out.append(
          AssembledBatch(
              payload=payload,
              is_final_batch=is_final,
              trajectory_ids=traj_ids,
          )
      )

    if self._step_rollouts >= self.total_step_rollouts:
      if self._buffer:
        remainder = list(self._buffer)
        self._buffer.clear()
        traj_ids = tuple(_extract_trajectory_id(it) for it in remainder)
        payload = self.pack(remainder)[0]
        out.append(
            AssembledBatch(
                payload=payload,
                is_final_batch=True,
                trajectory_ids=traj_ids,
            )
        )
      elif out:
        out[-1] = AssembledBatch(
            payload=out[-1].payload,
            is_final_batch=True,
            trajectory_ids=out[-1].trajectory_ids,
        )
      self._step_rollouts %= self.total_step_rollouts

    return out

  def flush(
      self,
  ) -> list[AssembledBatch]:
    """Flushes any remaining items padded to batch_size."""
    if not self._buffer:
      return []
    remainder = list(self._buffer)
    self._buffer.clear()
    self._step_rollouts = 0
    traj_ids = tuple(_extract_trajectory_id(it) for it in remainder)
    return [
        AssembledBatch(
            payload=self.pack(remainder)[0],
            is_final_batch=True,
            trajectory_ids=traj_ids,
        )
    ]

  def reset(self, *, start_batch_index: int | None = None) -> None:
    """Clears internal buffer and resets step rollouts counter."""
    self._buffer.clear()
    self._step_rollouts = 0
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
      payloads.append(self._pack_chunk(chunk))
    return payloads

  def _pack_chunk(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> datatypes.RLTrainerPayload:
    """Pads a single `<= batch_size` chunk into one rectangular payload."""
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

    prompt_ids, prompt_mask = [], []
    completion_ids, completion_mask, completion_valid = [], [], []
    advantages = []
    optional_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in present_fields
    }
    # Router replay is all-or-nothing per batch: a partially replayed batch
    # would silently mix replayed and freshly routed rows.
    replay_routing = all(it.routed_experts is not None for it in chunk)
    routed_experts_rows: list[np.ndarray] = []
    truncated_prompts = truncated_completions = 0

    for item in chunk:
      p_full = np.asarray(item.prompt_ids, dtype=np.int32).reshape(-1)
      c_full = np.asarray(item.completion_ids, dtype=np.int32).reshape(-1)
      truncated_prompts += p_full.size > self.max_prompt_length
      truncated_completions += c_full.size > self.max_response_length
      c = c_full[: self.max_response_length]

      p_ids, p_default_mask = _left_pad(
          p_full, self.max_prompt_length, pad_id=self.pad_id
      )
      c_ids, c_valid = _right_pad(
          c, self.max_response_length, pad_value=self.pad_id, dtype=np.int32
      )
      prompt_ids.append(p_ids)
      completion_ids.append(c_ids)
      completion_valid.append(c_valid)

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
      prompt_mask.append(p_mask)

      # Action mask over the completion: prefer an explicit action_mask, fall
      # back to completion_mask, then to "every generated token is an action".
      # completion_mask will be used in the loss_fn that's defined in
      # algo_core.py which masks out the non-action tokens so here we make sure
      # that completion_mask is aligned with the action masks.
      # TODO(tunix-dev): either deprecate action_mask or completion_mask as now
      # they are identical.
      action_source = (
          item.action_mask
          if item.action_mask is not None
          else item.completion_mask
      )
      if action_source is None:
        c_mask = c_valid.copy()
      else:
        c_mask = _completion_aligned(
            action_source,
            c.size,
            self.max_response_length,
            prompt_len=p_full.size,
            full_completion_len=c_full.size,
        )
      completion_mask.append(c_mask)

      advantages.append(
          _completion_aligned(
              item.advantages,
              c.size,
              self.max_response_length,
              fill_value=0.0,
              prompt_len=p_full.size,
              full_completion_len=c_full.size,
          )
      )

      for name in optional_rows:
        optional_rows[name].append(
            _completion_aligned(
                getattr(item, name),
                c.size,
                self.max_response_length,
                fill_value=0.0,
                prompt_len=p_full.size,
                full_completion_len=c_full.size,
            )
        )

      # `replay_routing` already guarantees this is set; binding it locally
      # also narrows the optional field for the type checker.
      routed = item.routed_experts
      if replay_routing and routed is not None:
        routed_experts_rows.append(
            _routed_experts_aligned(
                # `routed_experts` is declared ArrayLike, which admits jax
                # arrays and scalars; concretise it here as the sibling fields
                # above do.
                np.asarray(routed, dtype=np.int32),
                p_full.size,
                c.size,
                self.max_prompt_length,
                self.max_response_length,
            )
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

    # Zero-pad trailing rows so every chunk yields a static [B, ...] shape.
    while len(prompt_ids) < self.batch_size:
      prompt_ids.append(
          np.full(self.max_prompt_length, self.pad_id, dtype=np.int32)
      )
      prompt_mask.append(np.zeros(self.max_prompt_length, dtype=np.float32))
      completion_ids.append(
          np.full(self.max_response_length, self.pad_id, dtype=np.int32)
      )
      completion_mask.append(np.zeros(self.max_response_length, np.float32))
      completion_valid.append(np.zeros(self.max_response_length, np.float32))
      advantages.append(np.zeros(self.max_response_length, dtype=np.float32))
      for rows in optional_rows.values():
        rows.append(np.zeros(self.max_response_length, dtype=np.float32))
      if routed_experts_rows:
        routed_experts_rows.append(
            np.full_like(routed_experts_rows[0], datatypes.UNSET_ROUTED_EXPERT)
        )

    batched_prompt_ids = np.stack(prompt_ids)
    batched_prompt_mask = np.stack(prompt_mask)
    batched_completion_ids = np.stack(completion_ids)
    batched_completion_mask = np.stack(completion_mask)

    # loss_mask tracks the trainable tokens including prompt and completion
    # tokens.
    loss_mask = np.concatenate(
        [np.zeros_like(batched_prompt_mask), batched_completion_mask], axis=1
    )

    stacked_optional = {
        name: np.stack(rows) for name, rows in optional_rows.items()
    }
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

    return datatypes.RLTrainerPayload(
        loss_mask=loss_mask,
        action_mask=loss_mask,
        advantages=np.stack(advantages),
        prompt_ids=batched_prompt_ids,
        prompt_mask=batched_prompt_mask,
        completion_ids=batched_completion_ids,
        completion_mask=batched_completion_mask,
        ref_per_token_logps=stacked_optional.get("ref_per_token_logps"),
        old_per_token_logps=stacked_optional.get("old_per_token_logps"),
        returns=stacked_optional.get("returns"),
        old_values=stacked_optional.get("old_values"),
        sampler_is_weights=stacked_optional.get("sampler_is_weights"),
        routed_experts=(
            np.stack(routed_experts_rows) if routed_experts_rows else None
        ),
        metadata=payload_metadata,
    )

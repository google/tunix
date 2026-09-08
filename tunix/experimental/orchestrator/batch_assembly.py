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
import dataclasses
from typing import Any, Generic, NamedTuple, Protocol, Sequence, TypeVar
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.rl import packing

T = TypeVar("T")


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

  def feed(self, items: Sequence[T]) -> list[AssembledBatch]:
    """Ingests rollouts, emitting ready microbatches and auto-flushing at step end."""
    ...

  def flush(self) -> list[AssembledBatch]:
    """Drains remaining buffered items, padding to the required static tensor shape."""
    ...

  def reset(self) -> None:
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

  return packing.PackItem(
      prompt_ids=prompt,
      completion_ids=completion,
      completion_mask=completion_mask,
      advantages=resolve(item.advantages, fill=0.0, name="advantages"),
      per_token=per_token,
  )


def to_rl_trainer_payload(
    rows: Sequence[packing.PackedRow],
    *,
    max_segments: int,
    trajectory_ids: tuple[str, ...] = (),
) -> datatypes.RLTrainerPayload:
  """Converts a sequence of packing.PackedRow to an RLTrainerPayload."""
  stack = lambda attr: np.stack([getattr(r, attr) for r in rows])
  per_token_kwargs = {
      name: np.stack([r.per_token[name] for r in rows])
      for name in rows[0].per_token
  }
  return datatypes.RLTrainerPayload(
      prompt_ids=np.zeros((len(rows), 0), dtype=np.int32),
      prompt_mask=np.zeros((len(rows), 0), dtype=np.float32),
      completion_ids=stack("ids"),
      completion_mask=stack("completion_mask"),
      advantages=stack("advantages"),
      segment_ids=stack("segment_ids"),
      segment_positions=stack("segment_positions"),
      num_segments=max_segments + 1,
      metadata={"trajectory_ids": trajectory_ids},
      **per_token_kwargs,  # pyrefly: ignore[bad-argument-type]
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
      max_segments_per_packed_row: int | None = None,
  ):
    """Initializes SequencePackedBatchAssembler.

    Args:
      batch_size: Target batch dim for packed sequences.
      group_size: Number of rollout generations per prompt group (G).
      mini_batch_size: Number of prompt groups per model update.
      max_packed_len: Maximum packed sequence length per row.
      pad_id: Token ID used for padding.
      target_occupancy: Occupancy ratio above which a bin is sealed before full.
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
      raise ValueError(f"max_packed_len must be positive, got {max_packed_len}")
    if (
        max_segments_per_packed_row is not None
        and max_segments_per_packed_row <= 0
    ):
      raise ValueError(
          "max_segments_per_packed_row must be positive or None, got"
          f" {max_segments_per_packed_row}."
      )
    self.batch_size = batch_size
    self.max_packed_len = max_packed_len
    self.pad_id = pad_id
    self.group_size = group_size
    self.mini_batch_size = mini_batch_size
    self.max_segments_per_packed_row = max_segments_per_packed_row

    # Each entry is a `(PackItem, trajectory_id)` converted once at ingest.
    self._buffer: list[tuple[packing.PackItem, str]] = []
    self._step_rollouts: int = 0

  @property
  def total_step_rollouts(self) -> int:
    """Total number of rollouts expected per global training step."""
    return self.mini_batch_size * self.group_size

  def _emit_one_chunk(
      self, *, max_segments: int, drain_all: bool
  ) -> AssembledBatch:
    """Packs the head of the buffer into one microbatch, keeping leftovers."""
    pack_items = [item for item, _ in self._buffer]
    carried = packing.carried_per_token_fields(pack_items)
    id_to_entry = {
        id(item): entry for entry, item in zip(self._buffer, pack_items)
    }
    bins, leftover = packing.fill_one_chunk(
        pack_items,
        pack_size=self.batch_size,
        budget=self.max_packed_len,
        max_segments=max_segments,
    )
    placed = []
    for bin_items in bins:
      placed.extend(bin_items)
    traj_ids = tuple(id_to_entry[id(item)][1] for item in placed)
    rows = packing.pack_chunk(
        bins,
        budget=self.max_packed_len,
        pad_id=self.pad_id,
        carried=carried,
    )
    payload = to_rl_trainer_payload(
        rows, max_segments=max_segments, trajectory_ids=traj_ids
    )
    self._buffer = [id_to_entry[id(item)] for item in leftover]
    return AssembledBatch(
        payload=payload,
        is_final_batch=drain_all and not self._buffer,
        trajectory_ids=traj_ids,
    )

  def _drain_buffer(self, *, drain_all: bool) -> list[AssembledBatch]:
    """Drains buffered items into microbatches using FFD packing.

    When `drain_all` is False, only whole chunks whose token mass can fill a
    full microbatch are emitted, so the streaming tail is held back until more
    rollouts arrive. When `drain_all` is True (step boundary or `flush`), the
    buffer is drained completely and the last chunk is marked final.
    """
    out: list[AssembledBatch] = []
    max_segments = packing.effective_max_segments(
        self.max_packed_len, self.max_segments_per_packed_row
    )
    chunk_capacity = self.batch_size * self.max_packed_len
    while self._buffer:
      if not drain_all:
        buffered_tokens = sum(item.num_tokens for item, _ in self._buffer)
        if buffered_tokens < chunk_capacity:
          break
      out.append(
          self._emit_one_chunk(max_segments=max_segments, drain_all=drain_all)
      )
    return out

  def feed(
      self, items: Sequence[datatypes.RLTrainerPayload]
  ) -> list[AssembledBatch]:
    """Ingests items into the buffer, auto-flushing on the step boundary."""
    for item in items:
      pack_item = to_pack_item(item)
      packing.validate_items([pack_item], self.max_packed_len)
      self._buffer.append((pack_item, _extract_trajectory_id(item)))
    self._step_rollouts += len(items)
    is_step_done = self._step_rollouts >= self.total_step_rollouts

    out = self._drain_buffer(drain_all=is_step_done)
    if is_step_done:
      self._step_rollouts %= self.total_step_rollouts
    return out

  def flush(self) -> list[AssembledBatch]:
    """Flushes any remaining buffered items, marking the last chunk final."""
    self._step_rollouts = 0
    return self._drain_buffer(drain_all=True)

  def reset(self) -> None:
    """Clears internal buffer and resets step rollouts counter."""
    self._buffer.clear()
    self._step_rollouts = 0


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
      self, items: Sequence[datatypes.RLTrainerPayload]
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

  def flush(self) -> list[AssembledBatch]:
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

  def reset(self) -> None:
    """Clears internal buffer and resets step rollouts counter."""
    self._buffer.clear()
    self._step_rollouts = 0

  def pack(
      self, items: Sequence[datatypes.RLTrainerPayload]
  ) -> list[datatypes.RLTrainerPayload]:
    """Pads items into rectangular 2D batches `[B, P + C]`."""
    item_list = list(items)
    if not item_list:
      return []

    payloads: list[datatypes.RLTrainerPayload] = []
    for i in range(0, len(item_list), self.batch_size):
      payloads.append(self._pack_chunk(item_list[i : i + self.batch_size]))
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
    completion_ids, completion_mask = [], []
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

      action_source = item.completion_mask
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

    stacked_optional = {
        name: np.stack(rows) for name, rows in optional_rows.items()
    }
    return datatypes.RLTrainerPayload(
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
        metadata={
            "trajectory_ids": tuple(_extract_trajectory_id(it) for it in chunk)
        },
    )

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
- 1D Sequence Packing (`SequencePackedBatchAssembler`) for Flash/FlexAttention (>90% MXU).
- Simple 2D Rectangular Padding (`PaddedBatchAssembler`).
"""

from typing import Any, Generic, Protocol, Sequence, TypeVar
import numpy as np
from tunix.experimental.common import datatypes
from tunix.rl import common as rl_common

T = TypeVar("T")


class BatchAssembler(Generic[T], Protocol):
  """Universal batch assembly protocol for microbatch packing."""

  def pack(self, items: Sequence[T]) -> list[datatypes.RLTrainerPayload]:
    """Packs items into hardware-sized microbatch trainer payloads."""
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
    out[-arr.size:] = arr
    mask[-arr.size:] = 1.0
  return out, mask


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
    out[:arr.size] = arr
    mask[:arr.size] = 1.0
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
    elif (
        prompt_len is not None
        and full_completion_len is not None
        and arr.size == prompt_len + full_completion_len
    ):
      arr = arr[prompt_len:prompt_len + full_completion_len]
    elif full_completion_len is not None and arr.size >= full_completion_len:
      arr = arr[:full_completion_len]
    elif arr.size >= completion_len:
      arr = arr[:completion_len]
    else:
      arr = np.pad(arr, (0, completion_len - arr.size), constant_values=0.0)
    arr = arr[:completion_len]
  out, _ = _right_pad(
      arr,
      max_response_length,
      pad_value=0.0,
      dtype=np.float32,
  )
  return out


class SequencePackedBatchAssembler:
  """1D Sequence Packing: Concatenates items into dense [1, max_packed_len] buffers."""

  def __init__(self, max_packed_len: int = 8192, pad_id: int = 0):
    self.max_packed_len = max_packed_len
    self.pad_id = pad_id

  def pack(self, items: Sequence[datatypes.RLTrainerPayload]) -> list[datatypes.RLTrainerPayload]:
    """Bin-packs items into dense 1D buffers with segment boundaries."""
    if not items:
      return []

    # Calculate token lengths from explicit fields
    item_lengths = []
    for it in items:
      item_lengths.append(len(it.token_ids) if it.token_ids is not None else 0)

    item_list = sorted(zip(items, item_lengths), key=lambda x: x[1], reverse=True)

    bins: list[list[datatypes.RLTrainerPayload]] = []
    bin_lengths: list[int] = []

    for item, length in item_list:
      placed = False
      for b_idx, current_len in enumerate(bin_lengths):
        if current_len + length <= self.max_packed_len:
          bins[b_idx].append(item)
          bin_lengths[b_idx] += length
          placed = True
          break
      if not placed:
        bins.append([item])
        bin_lengths.append(length)

    payloads: list[datatypes.RLTrainerPayload] = []
    for b_items in bins:
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

      concat_tokens = np.concatenate(all_tokens)
      concat_loss_masks = np.concatenate(all_loss_masks)
      concat_action_masks = np.concatenate(all_action_masks)
      concat_segment_ids = np.concatenate(all_segment_ids)
      concat_segment_positions = np.concatenate(all_segment_positions)
      concat_advantages = np.concatenate(all_advantages)

      pad_len = max(0, self.max_packed_len - len(concat_tokens))
      padded_tokens = np.pad(concat_tokens[: self.max_packed_len], (0, pad_len), constant_values=self.pad_id)
      padded_loss_mask = np.pad(concat_loss_masks[: self.max_packed_len], (0, pad_len), constant_values=0.0)
      padded_action_mask = np.pad(concat_action_masks[: self.max_packed_len], (0, pad_len), constant_values=0.0)
      padded_segment_ids = np.pad(concat_segment_ids[: self.max_packed_len], (0, pad_len), constant_values=0)
      padded_segment_positions = np.pad(concat_segment_positions[: self.max_packed_len], (0, pad_len), constant_values=0)
      padded_advantages = np.pad(concat_advantages[: self.max_packed_len], (0, pad_len), constant_values=0.0)

      batch_old_lp = None
      if all_old_logprobs:
        concat_old = np.concatenate(all_old_logprobs)
        batch_old_lp = np.pad(concat_old[: self.max_packed_len], (0, pad_len), constant_values=0.0)[np.newaxis, :]

      batch_ref_lp = None
      if all_ref_logprobs:
        concat_ref = np.concatenate(all_ref_logprobs)
        batch_ref_lp = np.pad(concat_ref[: self.max_packed_len], (0, pad_len), constant_values=0.0)[np.newaxis, :]

      payload = datatypes.RLTrainerPayload(
          token_ids=padded_tokens[np.newaxis, :],
          token_mask=padded_segment_ids[np.newaxis, :],
          loss_mask=padded_loss_mask[np.newaxis, :],
          advantages=padded_advantages[np.newaxis, :],
          action_mask=padded_action_mask[np.newaxis, :],
          prompt_ids=padded_tokens[np.newaxis, :],
          prompt_mask=padded_segment_ids[np.newaxis, :],
          completion_ids=padded_tokens[np.newaxis, :],
          completion_mask=padded_loss_mask[np.newaxis, :],
          old_per_token_logps=batch_old_lp,
          ref_per_token_logps=batch_ref_lp,
          segment_ids=padded_segment_ids[np.newaxis, :],
          segment_positions=padded_segment_positions[np.newaxis, :],
      )
      payloads.append(payload)

    return payloads


class GRPOTrainExampleAssembler:
  """Pads v2 RLTrainerPayloads into GRPO TrainExample microbatches.

  GRPO loss consumes separate prompt and completion tensors. The generic
  sequence assemblers operate on concatenated token streams, so this assembler
  keeps the GRPO fields aligned while still using Orchestrator V2's generic
  `BatchAssembler.pack()` contract.
  """

  def __init__(
      self,
      *,
      batch_size: int,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
  ):
    if batch_size <= 0:
      raise ValueError("train microbatch size must be positive.")
    self.batch_size = batch_size
    self.max_prompt_length = max_prompt_length
    self.max_response_length = max_response_length
    self.pad_id = pad_id

  def pack(
      self, items: Sequence[datatypes.RLTrainerPayload]
  ) -> list[rl_common.TrainExample]:
    item_list = list(items)
    if not item_list:
      return []

    microbatches = []
    for start in range(0, len(item_list), self.batch_size):
      chunk = item_list[start:start + self.batch_size]
      microbatches.append(self._pack_chunk(chunk))
    return microbatches

  def _pack_chunk(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> rl_common.TrainExample:
    prompt_ids = []
    prompt_mask = []
    completion_ids = []
    completion_mask = []
    advantages = []
    ref_logps = []
    old_logps = []
    has_ref_logps = any(x.ref_per_token_logps is not None for x in chunk)
    has_old_logps = any(x.old_per_token_logps is not None for x in chunk)

    for item in chunk:
      p = np.asarray(item.prompt_ids, dtype=np.int32).reshape(-1)
      c_full = np.asarray(item.completion_ids, dtype=np.int32).reshape(-1)
      c_mask_src = (
          np.asarray(item.completion_mask, dtype=np.float32).reshape(-1)
          if item.completion_mask is not None
          else np.ones(c_full.shape, dtype=np.float32)
      )
      c = c_full[:self.max_response_length]
      c_mask_src = c_mask_src[:c.size]

      p_ids, p_mask = _left_pad(
          p, self.max_prompt_length, pad_id=self.pad_id
      )
      c_ids, c_default_mask = _right_pad(
          c,
          self.max_response_length,
          pad_value=self.pad_id,
          dtype=np.int32,
      )
      c_mask = np.zeros(self.max_response_length, dtype=np.float32)
      if c_mask_src.size:
        c_mask[:c_mask_src.size] = c_mask_src
      else:
        c_mask = c_default_mask

      prompt_ids.append(p_ids)
      prompt_mask.append(p_mask)
      completion_ids.append(c_ids)
      completion_mask.append(c_mask)

      adv_arr = (
          np.asarray(item.advantages, dtype=np.float32).reshape(-1)
          if item.advantages is not None
          else None
      )
      advantages.append(
          _completion_aligned(
              adv_arr,
              c.size,
              self.max_response_length,
              fill_value=0.0,
              prompt_len=p.size,
              full_completion_len=c_full.size,
          )
      )

      if has_ref_logps:
        ref_logps.append(
            _completion_aligned(
                item.ref_per_token_logps,
                c.size,
                self.max_response_length,
                full_completion_len=c_full.size,
            )
        )
      if has_old_logps:
        old_logps.append(
            _completion_aligned(
                item.old_per_token_logps,
                c.size,
                self.max_response_length,
                full_completion_len=c_full.size,
            )
        )

    while len(prompt_ids) < self.batch_size:
      prompt_ids.append(np.full(self.max_prompt_length, self.pad_id, np.int32))
      prompt_mask.append(np.zeros(self.max_prompt_length, dtype=np.float32))
      completion_ids.append(
          np.full(self.max_response_length, self.pad_id, np.int32)
      )
      completion_mask.append(
          np.zeros(self.max_response_length, dtype=np.float32)
      )
      advantages.append(np.zeros(self.max_response_length, dtype=np.float32))
      if has_ref_logps:
        ref_logps.append(np.zeros(self.max_response_length, dtype=np.float32))
      if has_old_logps:
        old_logps.append(np.zeros(self.max_response_length, dtype=np.float32))

    return rl_common.TrainExample(
        prompt_ids=np.stack(prompt_ids),
        prompt_mask=np.stack(prompt_mask),
        completion_ids=np.stack(completion_ids),
        completion_mask=np.stack(completion_mask),
        advantages=np.stack(advantages),
        ref_per_token_logps=np.stack(ref_logps) if has_ref_logps else None,
        old_per_token_logps=np.stack(old_logps) if has_old_logps else None,
    )


class PaddedBatchAssembler:
  """Simple 2D Rectangular Batching: Pads sequences to standard [batch_size, max_seq_len] tensors."""

  def __init__(self, batch_size: int = 4, max_seq_len: int = 2048, pad_id: int = 0):
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.pad_id = pad_id

  def pack(self, items: Sequence[datatypes.RLTrainerPayload]) -> list[datatypes.RLTrainerPayload]:
    """Pads items into rectangular 2D batches [B, max_seq_len]."""
    if not items:
      return []

    item_list = list(items)
    payloads: list[datatypes.RLTrainerPayload] = []

    for i in range(0, len(item_list), self.batch_size):
      chunk = item_list[i : i + self.batch_size]

      b_tokens = []
      b_loss_masks = []
      b_action_masks = []
      b_advs = []
      b_old_lps = []
      b_ref_lps = []

      for it in chunk:
        toks = (
            np.asarray(it.token_ids, dtype=np.int32).reshape(-1)
            if it.token_ids is not None
            else np.zeros(0, dtype=np.int32)
        )
        seq_len = len(toks)

        loss_mask = (
            np.asarray(it.loss_mask, dtype=np.float32).reshape(-1)
            if it.loss_mask is not None
            else np.zeros(seq_len, dtype=np.float32)
        )
        action_mask = (
            np.asarray(it.action_mask, dtype=np.float32).reshape(-1)
            if it.action_mask is not None
            else np.zeros(seq_len, dtype=np.float32)
        )
        adv_arr = (
            np.asarray(it.advantages, dtype=np.float32).reshape(-1)
            if it.advantages is not None
            else np.zeros(seq_len, dtype=np.float32)
        )

        pad_len = max(0, self.max_seq_len - seq_len)
        b_tokens.append(np.pad(toks[: self.max_seq_len], (0, pad_len), constant_values=self.pad_id))
        b_loss_masks.append(np.pad(loss_mask[: self.max_seq_len], (0, pad_len), constant_values=0.0))
        b_action_masks.append(np.pad(action_mask[: self.max_seq_len], (0, pad_len), constant_values=0.0))
        b_advs.append(np.pad(adv_arr[: self.max_seq_len], (0, pad_len), constant_values=0.0))

        if it.old_per_token_logps is not None:
          old_arr = np.asarray(it.old_per_token_logps, dtype=np.float32).reshape(-1)
          b_old_lps.append(
              np.pad(old_arr[: self.max_seq_len], (0, pad_len), constant_values=0.0)
          )

        if it.ref_per_token_logps is not None:
          ref_arr = np.asarray(it.ref_per_token_logps, dtype=np.float32).reshape(-1)
          b_ref_lps.append(
              np.pad(ref_arr[: self.max_seq_len], (0, pad_len), constant_values=0.0)
          )

      # Pad rows up to batch_size
      while len(b_tokens) < self.batch_size:
        b_tokens.append(np.full(self.max_seq_len, self.pad_id, dtype=np.int32))
        b_loss_masks.append(np.zeros(self.max_seq_len, dtype=np.float32))
        b_action_masks.append(np.zeros(self.max_seq_len, dtype=np.float32))
        b_advs.append(np.zeros(self.max_seq_len, dtype=np.float32))
        if b_old_lps:
          b_old_lps.append(np.zeros(self.max_seq_len, dtype=np.float32))
        if b_ref_lps:
          b_ref_lps.append(np.zeros(self.max_seq_len, dtype=np.float32))

      payload = datatypes.RLTrainerPayload(
          token_ids=np.stack(b_tokens),
          token_mask=np.stack(b_loss_masks),
          loss_mask=np.stack(b_loss_masks),
          advantages=np.stack(b_advs),
          action_mask=np.stack(b_action_masks),
          prompt_ids=np.stack(b_tokens),
          prompt_mask=np.stack(b_loss_masks),
          completion_ids=np.stack(b_tokens),
          completion_mask=np.stack(b_loss_masks),
          old_per_token_logps=np.stack(b_old_lps) if b_old_lps else None,
          ref_per_token_logps=np.stack(b_ref_lps) if b_ref_lps else None,
      )
      payloads.append(payload)

    return payloads

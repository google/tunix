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

# TODO: Align SequencePackedBatchAssembler with the rest of the ecosystem and potentially move to a common library.
"""

import logging
from typing import Any, Generic, Protocol, Sequence, TypeVar
import numpy as np
from jax import numpy as jnp
from tunix.experimental.common import datatypes
from tunix.rl import common as rl_common

T = TypeVar("T")

_logger = logging.getLogger(__name__)

# Per-token payload fields that are only meaningful for some algorithms. They
# are emitted all-or-nothing per microbatch and left as None when unused, so a
# GRPO batch never allocates PPO-only buffers.
_OPTIONAL_PER_TOKEN_FIELDS = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
    "sampler_is_weights",
)


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


def _completion_values(
    values: Any | None,
    completion_len: int,
    *,
    fill_value: float = 0.0,
    prompt_len: int | None = None,
    full_completion_len: int | None = None,
) -> np.ndarray:
  """Resolves a source array onto exactly `completion_len` completion columns.

  Accepts sources laid out over the completion (`[C]`), over the whole
  prompt+completion sequence (`[P + C]`), or as a single per-sequence scalar.

  Args:
    values: Source array, scalar, or None.
    completion_len: Number of valid (post-truncation) completion tokens.
    fill_value: Value used when `values` is None.
    prompt_len: Length of the unpadded prompt, used to detect and strip a
      sequence-aligned source.
    full_completion_len: Length of the completion before truncation, used to
      detect a sequence-aligned source.

  Returns:
    A `[completion_len]` float32 array.
  """
  if values is None:
    return np.full(completion_len, fill_value, dtype=np.float32)
  arr = np.asarray(values, dtype=np.float32).reshape(-1)
  if arr.size == 1:
    # Per-sequence scalar (e.g. a GRPO advantage): broadcast over completion.
    return np.full(completion_len, float(arr[0]), dtype=np.float32)
  if prompt_len is not None and arr.size in (
      prompt_len + (full_completion_len or 0),
      prompt_len + completion_len,
  ):
    # Sequence-aligned `[P + C]` source: slice out the completion span.
    arr = arr[prompt_len:]
  if arr.size >= completion_len:
    return arr[:completion_len]
  return np.pad(arr, (0, completion_len - arr.size), constant_values=0.0)


def _completion_aligned(
    values: Any | None,
    completion_len: int,
    max_response_length: int,
    *,
    fill_value: float = 0.0,
    prompt_len: int | None = None,
    full_completion_len: int | None = None,
) -> np.ndarray:
  """Right-pads `_completion_values` out to a full `[max_response_length]` row."""
  arr = _completion_values(
      values,
      completion_len,
      fill_value=fill_value,
      prompt_len=prompt_len,
      full_completion_len=full_completion_len,
  )
  out, _ = _right_pad(
      arr,
      max_response_length,
      pad_value=0.0,
      dtype=np.float32,
  )
  return out


def _split_prompt_completion(
    item: datatypes.RLTrainerPayload,
) -> tuple[np.ndarray, np.ndarray]:
  """Recovers the unpadded prompt / completion spans of an unbatched payload.

  Prefers the explicit `prompt_ids` / `completion_ids` fields. When a payload
  only carries a concatenated `token_ids` stream (as `SequencePackedBatchAssembler`
  consumers do), the boundary is inferred from the first position where
  `loss_mask` becomes non-zero, which is where the trainable completion starts.

  Args:
    item: Unbatched RL trainer payload.

  Returns:
    A `(prompt_tokens, completion_tokens)` pair of 1D int32 arrays.
  """
  if item.prompt_ids is not None or item.completion_ids is not None:
    prompt = (
        np.asarray(item.prompt_ids, dtype=np.int32).reshape(-1)
        if item.prompt_ids is not None
        else np.zeros(0, dtype=np.int32)
    )
    completion = (
        np.asarray(item.completion_ids, dtype=np.int32).reshape(-1)
        if item.completion_ids is not None
        else np.zeros(0, dtype=np.int32)
    )
    return prompt, completion

  tokens = (
      np.asarray(item.token_ids, dtype=np.int32).reshape(-1)
      if item.token_ids is not None
      else np.zeros(0, dtype=np.int32)
  )
  if item.loss_mask is None:
    return tokens, np.zeros(0, dtype=np.int32)
  loss_mask = np.asarray(item.loss_mask, dtype=np.float32).reshape(-1)
  trainable = np.flatnonzero(loss_mask[: tokens.size])
  split = int(trainable[0]) if trainable.size else tokens.size
  return tokens[:split], tokens[split:]


class SequencePackedBatchAssembler:
  """1D Sequence Packing: Concatenates items into dense [1, max_packed_len] buffers."""
  # TODO: align implementation with current path.
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
      item_lengths.append(len(it.token_ids) if it.token_ids is not None else 0)  # pyrefly: ignore[bad-argument-type]

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
          old_per_token_logps=batch_old_lp,
          ref_per_token_logps=batch_ref_lp,
          segment_ids=padded_segment_ids[np.newaxis, :],
          segment_positions=padded_segment_positions[np.newaxis, :],
      )
      payloads.append(payload)

    return payloads


class GRPOTrainExampleAssembler:
  """Pads GRPO payloads into `TrainExample` microbatches.

  Generic assemblers operate on concatenated token streams. GRPO loss consumes
  separate prompt and completion tensors, so this assembler keeps those fields
  aligned while still implementing the common `BatchAssembler.pack()` contract.
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
      chunk = item_list[start : start + self.batch_size]
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
      c = c_full[: self.max_response_length]
      c_mask_src = c_mask_src[: c.size]

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
        c_mask[: c_mask_src.size] = c_mask_src
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
        prompt_ids=jnp.stack(prompt_ids),
        prompt_mask=jnp.stack(prompt_mask),
        completion_ids=jnp.stack(completion_ids),
        completion_mask=jnp.stack(completion_mask),
        advantages=jnp.stack(advantages),
        ref_per_token_logps=jnp.stack(ref_logps) if has_ref_logps else None,
        old_per_token_logps=jnp.stack(old_logps) if has_old_logps else None,
    )


class PaddedBatchAssembler:
  """Simple 2D rectangular batching into fixed `[B, P + C]` trainer payloads.

  Each output row follows the `TrainerPayload` layout contract: a LEFT-padded
  prompt of width `max_prompt_length` concatenated with a RIGHT-padded
  completion of width `max_response_length`. Because the prompt/completion
  boundary is therefore identical on every row, completion-aligned tensors
  (`advantages`, `ref_per_token_logps`, `old_per_token_logps`, `returns`, ...)
  are emitted in completion space `[B, C]` and stay aligned with
  `completion_ids`.

  Field shapes on the returned payloads:
    token_ids / token_mask / loss_mask / action_mask: `[B, P + C]`
    prompt_ids / prompt_mask:                         `[B, P]`
    completion_ids / completion_mask:                 `[B, C]`
    advantages / *_per_token_logps / returns /
    old_values / sampler_is_weights:                  `[B, C]`

  `token_mask` marks real (non-pad) tokens, whereas `loss_mask` is zero over the
  prompt and equal to the action mask over the completion, so tool-observation
  tokens are attended to but do not contribute to the loss.

  Trailing rows of the final chunk are zero-filled with `token_mask = 0` and
  `loss_mask = 0`; `metadata["num_real_rows"]` records how many rows are real.

  Optional per-token fields (`ref_per_token_logps`, `old_per_token_logps`,
  `returns`, `old_values`, `sampler_is_weights`) are all-or-nothing per
  microbatch: left as `None` when no item carries them, and rejected when only
  some items do. Nothing is materialised for an algorithm that does not use it.

  Memory note: `prompt_ids` / `completion_ids` are views into `token_ids`, and
  `prompt_mask` / `completion_mask` are views into `token_mask` / `loss_mask`.
  Reading them is free; mutating one in place also mutates the other, so treat
  a packed payload as read-only.
  """

  def __init__(
      self,
      *,
      batch_size: int = 4,
      max_prompt_length: int = 512,
      max_response_length: int = 1536,
      pad_id: int = 0,
  ):
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
    self.batch_size = batch_size
    self.max_prompt_length = max_prompt_length
    self.max_response_length = max_response_length
    self.pad_id = pad_id

  @property
  def max_seq_len(self) -> int:
    """Total width of a packed row (`P + C`)."""
    return self.max_prompt_length + self.max_response_length

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

  def _carried_optional_fields(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> tuple[str, ...]:
    """Returns the optional per-token fields carried by every row of a chunk.

    Optional fields are all-or-nothing within a microbatch. A field absent from
    every row stays `None` on the output payload rather than being materialised
    as a dense zero tensor, so a GRPO batch never allocates (or ships to the
    accelerator) `returns` / `old_values` / `old_per_token_logps` buffers it has
    no use for.

    A field present on only *some* rows is rejected instead of zero-filled.
    Zero is not a neutral value for the quantities involved: a zero
    log-probability means `exp(0) == 1`, so a fabricated row would silently
    distort the KL and importance-ratio terms rather than drop out of them.

    Args:
      chunk: The items about to be packed into one microbatch.

    Returns:
      Names of the optional fields to emit, in declaration order.

    Raises:
      ValueError: If an optional field is set on some but not all items.
    """
    carried = []
    for name in _OPTIONAL_PER_TOKEN_FIELDS:
      populated = [getattr(item, name) is not None for item in chunk]
      if all(populated):
        carried.append(name)
      elif any(populated):
        missing = [i for i, has_value in enumerate(populated) if not has_value]
        raise ValueError(
            f"'{name}' is set on some but not all items of a microbatch"
            f" (missing on rows {missing}). Optional per-token fields must be"
            " supplied for every item or for none; zero-filling the gaps would"
            " silently corrupt the loss."
        )
    return tuple(carried)

  def _pack_chunk(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> datatypes.RLTrainerPayload:
    """Pads a single `<= batch_size` chunk into one rectangular payload."""
    rows = self.batch_size
    p_width = self.max_prompt_length
    c_width = self.max_response_length

    # Three `[B, P + C]` buffers back every token-space tensor; `prompt_*` and
    # `completion_*` are returned as VIEWS into them. That is one allocation per
    # token-space tensor instead of one per segment plus a stack and a concat,
    # and it makes the trailing-row padding free: the buffers already hold
    # `pad_id` / 0, so short chunks need no explicit fill pass at all.
    token_ids = np.full((rows, p_width + c_width), self.pad_id, dtype=np.int32)
    token_mask = np.zeros((rows, p_width + c_width), dtype=np.float32)
    loss_mask = np.zeros((rows, p_width + c_width), dtype=np.float32)
    prompt_ids = token_ids[:, :p_width]
    completion_ids = token_ids[:, p_width:]
    prompt_mask = token_mask[:, :p_width]
    completion_valid = token_mask[:, p_width:]
    # The prompt half of `loss_mask` is never written, so it stays zero by
    # construction: exactly the "no loss on the prompt" contract.
    completion_mask = loss_mask[:, p_width:]

    advantages = np.zeros((rows, c_width), dtype=np.float32)
    optional = {
        name: np.zeros((rows, c_width), dtype=np.float32)
        for name in self._carried_optional_fields(chunk)
    }

    truncated_prompts = truncated_completions = 0
    for row, item in enumerate(chunk):
      p_full, c_full = _split_prompt_completion(item)
      truncated_prompts += p_full.size > p_width
      truncated_completions += c_full.size > c_width
      p = p_full[-p_width:]
      c = c_full[:c_width]

      if p.size:
        prompt_ids[row, p_width - p.size :] = p
        prompt_mask[row, p_width - p.size :] = 1.0
      if c.size:
        completion_ids[row, : c.size] = c
        completion_valid[row, : c.size] = 1.0

      # A caller-supplied prompt mask is prompt-aligned, so it must be
      # left-padded exactly like the prompt ids to stay in register. If its
      # length disagrees with the prompt the alignment is undefined, so keep the
      # validity mask derived from the ids themselves.
      if item.prompt_mask is not None:
        src = np.asarray(item.prompt_mask, dtype=np.float32).reshape(-1)
        if src.size == p_full.size:
          src = src[-p_width:]
          prompt_mask[row, :] = 0.0
          if src.size:
            prompt_mask[row, p_width - src.size :] = src

      if not c.size:
        continue

      # Action mask over the completion: prefer an explicit action_mask, fall
      # back to completion_mask, then to "every generated token is an action".
      action_source = (
          item.action_mask
          if item.action_mask is not None
          else item.completion_mask
      )
      if action_source is None:
        completion_mask[row, : c.size] = 1.0
      else:
        completion_mask[row, : c.size] = _completion_values(
            action_source,
            c.size,
            prompt_len=p_full.size,
            full_completion_len=c_full.size,
        )

      advantages[row, : c.size] = _completion_values(
          item.advantages,
          c.size,
          fill_value=0.0,
          prompt_len=p_full.size,
          full_completion_len=c_full.size,
      )
      for name, buffer in optional.items():
        buffer[row, : c.size] = _completion_values(
            getattr(item, name),
            c.size,
            fill_value=0.0,
            prompt_len=p_full.size,
            full_completion_len=c_full.size,
        )

    if truncated_prompts or truncated_completions:
      _logger.warning(
          "PaddedBatchAssembler truncated %d prompt(s) to %d tokens and %d "
          "completion(s) to %d tokens; raise max_prompt_length / "
          "max_response_length to avoid dropping training signal.",
          truncated_prompts,
          p_width,
          truncated_completions,
          c_width,
      )

    return datatypes.RLTrainerPayload(
        token_ids=token_ids,
        token_mask=token_mask,
        loss_mask=loss_mask,
        action_mask=loss_mask,
        advantages=advantages,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        metadata={"num_real_rows": len(chunk)},
        **optional,
    )

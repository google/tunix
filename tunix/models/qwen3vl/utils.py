# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoding utilities for Qwen3-VL: batch and conversation encoders."""

from __future__ import annotations

from typing import Any

from flax import struct
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor
from tunix.models.qwen3vl import model as model_lib
from tunix.models.qwen3vl.model import get_rope_index
from tunix.models.qwen3vl.vision import compute_grid_data
from tunix.models.qwen3vl.vision import VisionGridData

# Special token IDs (constant for all Qwen3-VL checkpoints).
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652


@struct.dataclass
class EncodedBatch:
  """Output of encode_batch / encode_messages.

  B = batch size, L = max sequence length,
  P = total patch tokens across all images in the batch, C = patch channels.

  Attributes:
    input_tokens:    [B, L]    int32      — token ids (right-padded)
    input_mask:      [B, L]    bool       — True at non-padding positions
    completion_mask: [B, L]    bool       — True at tokens to include in loss
    positions:       [3, B, L] int32      — 3-D M-RoPE positions
    pixel_values:    [P, C]    float32    — patch tokens, all images concatenated
                               (None if batch contains no images)
    vision_grid:     VisionGridData       — pre-computed positional data for the
                               visual encoder (None if no images)
  """

  input_tokens: np.ndarray  # [B, L]
  input_mask: np.ndarray  # [B, L]
  completion_mask: np.ndarray  # [B, L]
  positions: np.ndarray  # [3, B, L]
  pixel_values: np.ndarray | None  # [P, C]
  vision_grid: VisionGridData | None


def encode_batch(
    processor: AutoProcessor,
    texts: list[str],
    images: list[list[Any]],
    *,
    vcfg: model_lib.VisionModelConfig,
    max_length: int,
    padding: bool | str = True,
    truncation: bool | str = True,
    pad_to_multiple_of: int | None = None,
) -> EncodedBatch:
  """Encode a batch of pre-formatted texts with corresponding image lists.

  Each ``texts[i]`` is a fully-formatted prompt string (e.g. the output of
  ``processor.apply_chat_template``).  ``images[i]`` is the list of PIL images
  for that item.  If an item has no images pass an empty list.

  Args:
    processor: HuggingFace AutoProcessor for Qwen3-VL.
    texts: List of B formatted prompt strings.
    images: List of B image lists (each inner list may be empty).
    vcfg: VisionModelConfig for the model (used to compute M-RoPE positions and
      VisionGridData — no learned weights needed).
    max_length: Maximum sequence length; longer sequences are truncated.
    padding: Passed to the processor (e.g. True, "max_length").
    truncation: Passed to the processor.
    pad_to_multiple_of: If set, pad sequence length to the next multiple.

  Returns:
    EncodedBatch
  """
  flat_images = [img for imgs in images for img in imgs]
  inputs = processor(
      text=texts,
      images=flat_images if flat_images else None,
      max_length=max_length,
      padding=padding,
      truncation=truncation,
      pad_to_multiple_of=pad_to_multiple_of,
      return_tensors=None,
  )

  input_ids = np.array(inputs['input_ids'], dtype=np.int32)  # [B, L]
  input_mask = np.array(inputs['attention_mask'], dtype=np.int32)  # [B, L]

  if flat_images:
    pixel_values = np.array(inputs['pixel_values'], dtype=np.float32)
    image_grid_thw = np.array(inputs['image_grid_thw'], dtype=np.int32)
    vision_grid = compute_grid_data(image_grid_thw, vcfg)
  else:
    pixel_values = None
    image_grid_thw = None
    vision_grid = None

  positions, _ = get_rope_index(
      input_ids=jnp.array(input_ids),
      image_grid_thw=jnp.array(image_grid_thw)
      if image_grid_thw is not None
      else None,
      video_grid_thw=None,
      attention_mask=jnp.array(input_mask),
      spatial_merge_size=vcfg.spatial_merge_size,
      image_token_id=vcfg.image_pad_id,
      video_token_id=_VIDEO_TOKEN_ID,
      vision_start_token_id=_VISION_START_TOKEN_ID,
  )  # [3, B, L]

  return EncodedBatch(
      input_tokens=input_ids,
      input_mask=input_mask,
      completion_mask=np.zeros(input_ids.shape, dtype=bool),
      positions=np.array(positions),
      pixel_values=pixel_values,
      vision_grid=vision_grid,
  )


def encode_messages(
    processor: AutoProcessor,
    conversations: list[list[dict[str, Any]]],
    loss_roles: set[str],
    *,
    vcfg: model_lib.VisionModelConfig,
    max_seq_len: int,
    padding: bool | str = True,
    truncation: bool | str = True,
    pad_to_multiple_of: int | None = None,
) -> EncodedBatch:
  """Encode OpenAI-format conversations with per-role loss masking.

  Args:
    processor: HuggingFace AutoProcessor for Qwen3-VL.
    conversations: List of B conversations, each a list of message dicts with
      keys ``role`` (str) and ``content`` (str or list of content blocks).
    loss_roles: Set of role names whose tokens are included in the loss (e.g.
      ``{"assistant"}``).  Tokens from all other roles are masked out.
    vcfg: VisionModelConfig for the model.
    max_seq_len: Maximum sequence length; longer sequences are truncated.
    padding: Passed to the processor.
    truncation: Passed to the processor.
    pad_to_multiple_of: If set, pad sequence length to the next multiple.

  Returns:
    EncodedBatch
  """
  comp_masks: list[np.ndarray] = []
  texts: list[str] = []
  all_images: list[list[Any]] = []

  for conv in conversations:
    images: list[Any] = []
    for msg in conv:
      content = msg.get('content', '')
      if isinstance(content, list):
        for block in content:
          if isinstance(block, dict) and block.get('type') == 'image':
            img = block.get('image')
            if img is not None:
              images.append(img)

    full_text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=False
    )
    full_ids = processor.tokenizer.encode(full_text, add_special_tokens=False)

    mask = np.zeros(len(full_ids), dtype=bool)
    for idx, msg in enumerate(conv):
      if msg['role'] not in loss_roles:
        continue
      prefix_ids = processor.tokenizer.encode(
          processor.apply_chat_template(
              conv[:idx], tokenize=False, add_generation_prompt=False
          ),
          add_special_tokens=False,
      )
      prefix_next_ids = processor.tokenizer.encode(
          processor.apply_chat_template(
              conv[: idx + 1], tokenize=False, add_generation_prompt=False
          ),
          add_special_tokens=False,
      )
      mask[len(prefix_ids) : len(prefix_next_ids)] = True

    comp_masks.append(mask[:max_seq_len])
    texts.append(full_text)
    all_images.append(images)

  completion_mask = np.stack(
      [np.pad(m, (0, max(0, max_seq_len - len(m)))) for m in comp_masks]
  )  # [B, max_seq_len]

  batch = encode_batch(
      processor,
      texts,
      all_images,
      vcfg=vcfg,
      max_length=max_seq_len,
      padding=padding,
      truncation=truncation,
      pad_to_multiple_of=pad_to_multiple_of,
  )
  B, L = batch.input_tokens.shape
  out_comp_mask = np.zeros((B, L), dtype=bool)
  clip = min(completion_mask.shape[1], L)
  out_comp_mask[:, :clip] = completion_mask[:, :clip]
  return EncodedBatch(
      input_tokens=batch.input_tokens,
      input_mask=batch.input_mask,
      completion_mask=out_comp_mask,
      positions=batch.positions,
      pixel_values=batch.pixel_values,
      vision_grid=batch.vision_grid,
  )

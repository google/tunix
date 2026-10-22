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

"""Dataset coordinate tagging helpers for prompt items."""

from collections.abc import Mapping
import copy
import dataclasses
from typing import Any
from absl import logging


def prompt_coordinates(prompt_idx: int, full_batch_size: int) -> dict[str, int]:
  """Returns the dataset coordinates of the prompt at `prompt_idx`.

  The coordinates are derived purely from dataset position, so they are stable
  across restarts and identical on retry. `batch_idx` is the prompt batch a
  trajectory belongs to, which is what a batch-ordered queue manager routes and
  orders on; `intra_batch_idx` identifies the prompt inside that batch.

  Args:
    prompt_idx: Position of the prompt in the dataset.
    full_batch_size: Prompts per prompt batch (groups per trainer iteration).
  """
  return {
      "prompt_idx": prompt_idx,
      "batch_idx": prompt_idx // full_batch_size,
      "intra_batch_idx": prompt_idx % full_batch_size,
  }


def tag_prompt(prompt_item: Any, coordinates: Mapping[str, int]) -> Any:
  """Stamps `coordinates` onto a copy of `prompt_item`'s metadata.

  The dataset item itself is never mutated: dicts are copied, and other items
  are shallow-copied before their `metadata` is replaced. Coordinates win over
  any same-named key already present, because dataset position is the only
  authority on them.

  Args:
    prompt_item: A prompt dict, or any object the engine accepts.
    coordinates: The tags to stamp, from `prompt_coordinates`.

  Returns:
    The tagged item, or `prompt_item` unchanged when its metadata cannot be
    replaced (logged, since a batch-ordered queue will later reject it).
  """
  if isinstance(prompt_item, Mapping):
    tagged = dict(prompt_item)
    tagged["metadata"] = {**(tagged.get("metadata") or {}), **coordinates}
    return tagged

  metadata = {**(getattr(prompt_item, "metadata", None) or {}), **coordinates}
  if dataclasses.is_dataclass(prompt_item) and any(
      field.name == "metadata" for field in dataclasses.fields(prompt_item)
  ):
    # Covers frozen dataclasses, whose attributes cannot be assigned.
    return dataclasses.replace(prompt_item, metadata=metadata)

  tagged = copy.copy(prompt_item)
  try:
    tagged.metadata = metadata
  except (AttributeError, TypeError):
    logging.warning(
        "Cannot tag prompt coordinates onto a %s; batch-ordered consumption"
        " requires a `metadata` mapping on every prompt item.",
        type(prompt_item).__name__,
    )
    return prompt_item
  return tagged

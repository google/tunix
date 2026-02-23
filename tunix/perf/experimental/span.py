# Copyright 2026 Google LLC
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

"""Span based data model."""

from __future__ import annotations

import bisect
from typing import Any

from absl import logging


JaxDevice = Any


class Span:
  """Represents a duration of time with a name, beginning, and end.

  Attributes:
    name: The name of the span.
    begin: The start time of the span.
    end: The end time of the span.
    id: The ID of the span.
    parent_id: The ID of the parent span.
    tags: A dictionary of tags associated with the span. Tags are used for post-
      processing and grouping spans. Some well-known tags are defined as
      constants in `tunix.perf.experimental.constants` (e.g.,
      constants.GLOBAL_STEP). Users can also add arbitrary tags.
  """

  id: int
  parent_id: int | None
  name: str
  begin: float
  end: float
  tags: dict[str, Any]

  def __init__(
      self,
      *,
      name: str,
      begin: float,
      id: int,
      parent_id: int | None = None,
      tags: dict[str, Any] | None = None,
  ):
    self.id = id
    self.name = name
    self.begin = begin
    self.parent_id = parent_id
    self.tags = tags or {}
    self.end = float("inf")

  def add_tag(self, key: str, value: Any) -> None:
    """Adds a tag to the span.

    Args:
      key: The tag key.
      value: The tag value.
    """
    if key in self.tags:
      logging.warning(
          "Tag '%s' already exists with value '%s'. Overwriting with '%s'.",
          key,
          self.tags[key],
          value,
      )
    self.tags[key] = value

  def __repr__(self, born_at: float = 0.0) -> str:
    begin = self.begin - born_at
    end = self.end - born_at
    out = f"[{self.id}] {self.name}: {begin:.6f}, {end:.6f}"
    if self.parent_id is not None:
      out += f" (parent={self.parent_id})"
    if self.tags:
      out += f", tags={self.tags}"
    return out

  @property
  def ended(self) -> bool:
    """Returns True if the span has ended."""

    return self.end != float("inf")

  @property
  def duration(self) -> float:
    """Returns the duration of the span."""
    return self.end - self.begin

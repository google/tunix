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

"""Span based data model."""

from __future__ import annotations

from typing import Any

JaxDevice = Any


class Span:
  name: str
  begin: float
  end: float

  def __init__(self, name: str, begin: float):
    self.name = name
    self.begin = begin
    self.end = float("inf")

  def __repr__(self) -> str:
    out = f"{self.name}: {self.begin:.6f}, {self.end:.6f}"
    return out

  @property
  def ended(self) -> bool:
    return self.end != float("inf")

  @property
  def duration(self) -> float:
    return self.end - self.begin


class SpanGroup:
  """Organizes spans into a tree structure."""

  name: str
  begin: float
  end: float
  outer: SpanGroup
  inner: list[Span | SpanGroup]

  def __init__(self, name: str, outer: SpanGroup | None = None):
    self.name = name
    self.begin = 0
    self.end = float("inf")
    self.inner = []
    if outer is not None:
      self.outer = outer
      outer.inner.append(self)
    else:
      self.outer = self

  def __repr__(self) -> str:
    out = f"{self.name}: {self.begin:.6f}, {self.end:.6f}"
    return out

  @property
  def duration(self) -> float:
    return self.end - self.begin

  def find_first_inner_group(self, name: str) -> SpanGroup | None:
    for item in self.inner:
      if isinstance(item, SpanGroup) and item.name == name:
        return item
    return None

  def find_last_inner_group(self, name: str) -> SpanGroup | None:
    for item in reversed(self.inner):
      if isinstance(item, SpanGroup) and item.name == name:
        return item
    return None

  def find_all_inner_groups(self, name: str) -> list[SpanGroup]:
    return [
        item
        for item in self.inner
        if isinstance(item, SpanGroup) and item.name == name
    ]

  def find_last_inner_span(self, name: str) -> Span | None:
    for item in reversed(self.inner):
      if isinstance(item, Span) and item.name == name:
        return item
    return None

  def find_all_inner_spans(self, name: str) -> list[Span]:
    return [
        item
        for item in self.inner
        if isinstance(item, Span) and item.name == name
    ]


def span_group_tostring(group: SpanGroup, born: float = 0.0) -> str:
  """Converts a span group tree to a string."""

  def _tostring_recursive(out: str, group: SpanGroup, indent: int) -> str:
    out += "  " * indent
    out += f"- {group.name} ({group.begin-born:.6f}, {group.end-born:.6f})\n"
    for item in group.inner:
      if isinstance(item, SpanGroup):
        out += _tostring_recursive("", item, indent + 1)
      elif isinstance(item, Span):
        out += "  " * (indent + 1)
        out += f"- {item.name} ({item.begin-born:.6f}, {item.end-born:.6f})\n"
    return out

  return _tostring_recursive("", group, 0)


def span_group_print(group: SpanGroup, born: float = 0.0) -> None:
  """Prints the span group tree."""
  print(span_group_tostring(group, born))


def span_group_stack_clone(stack: list[SpanGroup]) -> list[SpanGroup]:
  outer = SpanGroup(stack[0].name, None)
  outer.begin = stack[0].begin
  outer.end = stack[0].end
  stack_clone = [outer]
  for group in stack[1:]:
    inner = SpanGroup(group.name, outer)
    inner.begin = group.begin
    inner.end = group.end
    stack_clone.append(inner)
    outer = inner
  return stack_clone


# TODO(yangmu) create a new class SpanGroupBatch.


def span_group_batch_query_first(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  out_batch = []
  for group in batch:
    inner = group.find_first_inner_group(name)
    if inner is not None:
      out_batch.append(inner)
  return out_batch


def span_group_batch_query_last(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  out_batch = []
  for group in batch:
    inner = group.find_last_inner_group(name)
    if inner is not None:
      out_batch.append(inner)
  return out_batch


def span_group_batch_query_nth(
    batch: list[SpanGroup], name: str, index: int
) -> list[SpanGroup]:
  out_batch = []
  for group in batch:
    inners = group.find_all_inner_groups(name)
    if 0 <= index < len(inners):
      out_batch.append(inners[index])
  return out_batch


def span_group_batch_query_all(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  out_batch = []
  for group in batch:
    out_batch.extend(group.find_all_inner_groups(name))
  return out_batch

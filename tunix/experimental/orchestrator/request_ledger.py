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

"""The one place a rollout result is admitted, or refused.

While every request is delivered at most once, correlating results by request
id is enough. Retrying changes that: a retry is a second request for the same
logical piece of work, so the original and the retry have different request
ids and are indistinguishable as results. If a straggler arrives after its
group was given up on and retried, both are accepted, the group trains on a
duplicate, and the count it was reshaped against is wrong.

So results are admitted against the slot they fill -- the group, the position
within it, and the lineage it was issued in -- rather than against the request
that happened to produce them. The first result for a slot wins and later ones
are refused, which makes at-least-once delivery safe to build retries on.

Lineage is checked before anything else. A restart that reissues the same
group ids would otherwise let results from the abandoned attempt in through
the front door, since every other identifier matches.

Group metadata lives here rather than on the wire: a worker is told what to
generate, not where its output belongs.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Iterable, Optional

from absl import logging

from tunix.experimental.common import datatypes


class Admission(enum.Enum):
  """What the ledger did with an offered result."""

  ACCEPTED = "accepted"
  DUPLICATE = "duplicate"
  STALE_INCARNATION = "stale_incarnation"
  UNKNOWN = "unknown"


@dataclasses.dataclass(kw_only=True)
class RequestRecord:
  """A dispatched request, plus where its result belongs.

  Attributes:
    request: What was sent to the worker.
    group_id: The group this fills a place in.
    sample_index: Which place, 0..G-1.
    incarnation: The lineage epoch it was dispatched in.
    attempt: Which try this is. A retry shares the slot and takes a new
      request id.
  """

  request: datatypes.RolloutRequest
  group_id: str
  sample_index: int
  incarnation: int = 0
  attempt: int = 0

  @property
  def request_id(self) -> str:
    return self.request.request_id

  @property
  def slot(self) -> tuple[str, int, int]:
    """What this result would fill: group, position, lineage."""
    return (self.group_id, self.sample_index, self.incarnation)


class RequestLedger:
  """Tracks dispatched requests and admits at most one result per slot."""

  def __init__(self, *, incarnation: int = 0, group_size: Optional[int] = None):
    """Initializes the ledger.

    Args:
      incarnation: The lineage epoch results must belong to.
      group_size: Expected members per group, used to answer completeness.
        None means completeness is judged against whatever was registered.
    """
    self._incarnation = incarnation
    self._group_size = group_size
    self._records: dict[str, RequestRecord] = {}
    self._group_slots: dict[str, set[int]] = collections.defaultdict(set)
    self._accepted: dict[
        tuple[str, int, int], datatypes.RolloutResponse
    ] = {}
    self._accepted_by_group: dict[str, set[int]] = collections.defaultdict(set)

  @property
  def incarnation(self) -> int:
    return self._incarnation

  def advance_incarnation(self) -> int:
    """Starts a new lineage; results from the previous one stop being valid.

    Returns:
      The new incarnation.
    """
    self._incarnation += 1
    logging.info("Ledger advanced to incarnation %d", self._incarnation)
    return self._incarnation

  def register(self, records: Iterable[RequestRecord]) -> None:
    """Records requests as dispatched.

    Args:
      records: What was sent, typically a whole group at once.

    Raises:
      ValueError: If a record belongs to another lineage, or reuses a request
        id. Both would make later results unattributable.
    """
    for record in records:
      if record.incarnation != self._incarnation:
        raise ValueError(
            f"Request {record.request_id!r} was built for incarnation"
            f" {record.incarnation}, but the ledger is at"
            f" {self._incarnation}."
        )
      if record.request_id in self._records:
        raise ValueError(
            f"Request id {record.request_id!r} is already dispatched; ids must"
            " be unique or their results cannot be told apart."
        )
      self._records[record.request_id] = record
      self._group_slots[record.group_id].add(record.sample_index)

  def admit(self, response: datatypes.RolloutResponse) -> Admission:
    """Offers a result, applying the lineage gate and then slot dedup.

    Args:
      response: What a worker returned.

    Returns:
      Whether it was accepted, and if not, why.
    """
    record = self._records.get(response.request_id)
    if record is None:
      logging.warning(
          "Refusing result for request %r: nothing by that id was dispatched.",
          response.request_id,
      )
      return Admission.UNKNOWN

    if record.incarnation != self._incarnation:
      # Checked before dedup: a discarded lineage's result must not be able to
      # claim a slot in the current one.
      logging.warning(
          "Refusing result for request %r: it belongs to incarnation %d and"
          " the current lineage is %d.",
          response.request_id,
          record.incarnation,
          self._incarnation,
      )
      return Admission.STALE_INCARNATION

    if record.slot in self._accepted:
      logging.info(
          "Refusing result for request %r: slot %s is already filled, most"
          " likely by a retry or a duplicate delivery.",
          response.request_id,
          record.slot,
      )
      return Admission.DUPLICATE

    self._accepted[record.slot] = response
    self._accepted_by_group[record.group_id].add(record.sample_index)
    return Admission.ACCEPTED

  def record_for(self, request_id: str) -> Optional[RequestRecord]:
    """The record a result should be attributed to, if it is known."""
    return self._records.get(request_id)

  def is_group_complete(self, group_id: str) -> bool:
    """Whether every place in a group has been filled."""
    expected = self._group_size or len(self._group_slots.get(group_id, ()))
    if not expected:
      return False
    return len(self._accepted_by_group.get(group_id, ())) >= expected

  def missing_slots(self, group_id: str) -> list[int]:
    """Places in a group still waiting to be filled."""
    filled = self._accepted_by_group.get(group_id, set())
    return sorted(self._group_slots.get(group_id, set()) - filled)

  def accepted(self, group_id: str) -> list[datatypes.RolloutResponse]:
    """The admitted results for a group, ordered by their place in it."""
    filled = sorted(self._accepted_by_group.get(group_id, ()))
    return [
        self._accepted[(group_id, index, self._incarnation)]
        for index in filled
    ]

  def release_group(self, group_id: str) -> None:
    """Forgets a group that has reached a terminal state.

    Held until asked rather than dropped on completion, because whoever
    assembles the group still needs the requests behind its results.
    """
    for request_id, record in list(self._records.items()):
      if record.group_id == group_id:
        self._accepted.pop(record.slot, None)
        del self._records[request_id]
    self._group_slots.pop(group_id, None)
    self._accepted_by_group.pop(group_id, None)

  def __len__(self) -> int:
    return len(self._records)

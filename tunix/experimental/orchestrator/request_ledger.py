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

"""Request ledger: the orchestrator's book of in-flight rollout requests.

Grouping and lineage metadata (group_id, sample_index, incarnation) live here in
an orchestrator-internal `RequestRecord` rather than on the lean wire DTOs; the
ledger correlates each returned `RolloutResult` back to its record by
`request_id`. It is the single place that enforces two intake invariants:

  * incarnation gate -- results whose record predates the current incarnation are
    rejected before any dedup (so a restart that re-issues identical group_ids
    cannot admit results from a discarded lineage);
  * first-wins dedup on the logical slot `(group_id, sample_index, incarnation)`
    -- at-least-once delivery and retries collapse to one accepted result.

The ledger retains a group's records until the group is explicitly released
(after it reaches a terminal state), which is what lets a later assembly stage
recover the originating request (task payload, prompt) for each result.
"""

import collections
import dataclasses
import enum

from tunix.experimental.common import datatypes


class Admission(enum.Enum):
  """Outcome of offering a result to the ledger."""

  ACCEPTED = "accepted"
  DUPLICATE = "duplicate"
  STALE_INCARNATION = "stale_incarnation"
  UNKNOWN = "unknown"


@dataclasses.dataclass(kw_only=True)
class RequestRecord:
  """An in-flight request plus the orchestrator metadata the wire DTO omits.

  Attributes:
    request: The wire request dispatched to a rollout worker.
    group_id: The group this request belongs to (orchestrator-assigned, unique
      per dispatched group).
    sample_index: Position 0..G-1 of this sample within its group.
    incarnation: Orchestrator lineage epoch at dispatch time.
    attempt: Retry counter; a retry shares (group_id, sample_index) with a new
      request_id.
    mode: "train" or "eval".
  """

  request: datatypes.RolloutRequest
  group_id: str
  sample_index: int
  incarnation: int = 0
  attempt: int = 0
  mode: str = "train"

  @property
  def request_id(self) -> str:
    return self.request.request_id

  @property
  def slot(self) -> tuple[str, int, int]:
    """The logical dedup key: (group_id, sample_index, incarnation)."""
    return (self.group_id, self.sample_index, self.incarnation)


class RequestLedger:
  """Tracks in-flight requests, correlates results, and gates intake."""

  def __init__(self, *, incarnation: int = 0):
    self._incarnation = incarnation
    self._records: dict[str, RequestRecord] = {}
    self._group_samples: dict[str, set[int]] = collections.defaultdict(set)
    self._group_request_ids: dict[str, set[str]] = collections.defaultdict(set)
    self._slot_result: dict[tuple[str, int, int], datatypes.RolloutResult] = {}

  @property
  def incarnation(self) -> int:
    return self._incarnation

  def register(self, records: list[RequestRecord]) -> None:
    """Registers requests as in-flight (typically a full group at once).

    Args:
      records: The request records to track. Every record must belong to the
        ledger's current incarnation.

    Raises:
      ValueError: If a record's incarnation does not match the ledger's, or its
        request_id is already registered.
    """
    for record in records:
      if record.incarnation != self._incarnation:
        raise ValueError(
            f"record {record.request_id!r} has incarnation "
            f"{record.incarnation}, ledger is at {self._incarnation}"
        )
      if record.request_id in self._records:
        raise ValueError(f"duplicate request_id: {record.request_id!r}")
      self._records[record.request_id] = record
      self._group_samples[record.group_id].add(record.sample_index)
      self._group_request_ids[record.group_id].add(record.request_id)

  def admit(self, result: datatypes.RolloutResult) -> Admission:
    """Offers a result to the ledger, applying the incarnation gate and dedup.

    Args:
      result: The rollout result to admit; correlated by `request_id`.

    Returns:
      The admission outcome. Only ACCEPTED results are retained.
    """
    record = self._records.get(result.request_id)
    if record is None:
      return Admission.UNKNOWN
    if record.incarnation != self._incarnation:
      return Admission.STALE_INCARNATION
    if record.slot in self._slot_result:
      return Admission.DUPLICATE
    self._slot_result[record.slot] = result
    return Admission.ACCEPTED

  def group_size(self, group_id: str) -> int:
    """Number of distinct samples expected for a group (its G)."""
    return len(self._group_samples.get(group_id, ()))

  def is_group_complete(self, group_id: str) -> bool:
    """True once every expected sample of the group has an accepted result."""
    samples = self._group_samples.get(group_id)
    if not samples:
      return False
    return all(
        (group_id, sample_index, self._incarnation) in self._slot_result
        for sample_index in samples
    )

  def group_pairs(
      self, group_id: str
  ) -> list[tuple[RequestRecord, datatypes.RolloutResult]]:
    """Returns (record, result) pairs for a complete group, ordered by sample.

    Each result is paired with the record that actually produced it (by
    `request_id`), so a retried sample pairs with its winning attempt.

    Raises:
      KeyError: If the group is not complete.
    """
    if not self.is_group_complete(group_id):
      raise KeyError(f"group {group_id!r} is not complete")
    pairs = []
    for sample_index in sorted(self._group_samples[group_id]):
      result = self._slot_result[(group_id, sample_index, self._incarnation)]
      pairs.append((self._records[result.request_id], result))
    return pairs

  def release_group(self, group_id: str) -> None:
    """Drops all state for a group (call once it reaches a terminal state)."""
    samples = self._group_samples.pop(group_id, set())
    for sample_index in samples:
      self._slot_result.pop((group_id, sample_index, self._incarnation), None)
    for request_id in self._group_request_ids.pop(group_id, set()):
      self._records.pop(request_id, None)

  def group_ids(self) -> list[str]:
    return sorted(self._group_samples)

  def complete_group_ids(self) -> list[str]:
    return [g for g in self.group_ids() if self.is_group_complete(g)]

  def inflight_group_count(self) -> int:
    return len(self._group_samples)

  def advance_incarnation(self) -> int:
    """Bumps the incarnation and abandons all in-flight state.

    Mirrors the rewind protocol: after a restart the ledger rejects any result
    from the prior lineage (its records are gone), before dedup.

    Returns:
      The new incarnation.
    """
    self._incarnation += 1
    self._records.clear()
    self._group_samples.clear()
    self._group_request_ids.clear()
    self._slot_result.clear()
    return self._incarnation

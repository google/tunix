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

"""Group assembler: turns individually-returned results into whole groups.

Grouping is strictly ID-based: a group is keyed by `group_id` (never by array
position), it is emitted only once every one of its G samples has an accepted
result, and a degenerate G=1 group is never emitted. The assembler drives a
`RequestLedger` for intake (incarnation gate + first-wins dedup) and, on drain,
releases each completed group into an ordered `AssembledGroup` of
(request-record, result) pairs and frees it from the ledger.

The staleness gate is a separate, group-atomic step: a group's policy version is
the minimum across its members, so the whole group is kept or dropped together
(`group_policy_version` / `is_group_stale`); the loop applies it to drained
groups.
"""

import dataclasses

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import request_ledger


@dataclasses.dataclass(kw_only=True)
class AssembledGroup:
  """A complete group: ordered (record, result) pairs sharing a group_id."""

  group_id: str
  members: list[tuple[request_ledger.RequestRecord, datatypes.RolloutResult]]

  def __len__(self) -> int:
    return len(self.members)

  def results(self) -> list[datatypes.RolloutResult]:
    return [result for _, result in self.members]

  def policy_version(self) -> int:
    """Group-atomic policy version: the minimum across members (I4)."""
    return min(result.policy_version for _, result in self.members)


def is_group_stale(
    group: AssembledGroup, *, current_version: int, max_staleness: int
) -> bool:
  """Whether a group is too stale to train, atomically for the whole group.

  Args:
    group: The assembled group.
    current_version: The trainer's current weight/policy version.
    max_staleness: Maximum allowed lag; 0 means strictly on-policy.

  Returns:
    True if `current_version - group.policy_version() > max_staleness`.
  """
  return current_version - group.policy_version() > max_staleness


class GroupAssembler:
  """Assembles complete groups from a RequestLedger (ID-based, I1)."""

  def __init__(
      self,
      ledger: request_ledger.RequestLedger | None = None,
      *,
      min_group_size: int = 2,
  ):
    self._ledger = ledger if ledger is not None else request_ledger.RequestLedger()
    self._min_group_size = min_group_size

  @property
  def ledger(self) -> request_ledger.RequestLedger:
    return self._ledger

  def open_group(
      self, records: list[request_ledger.RequestRecord]
  ) -> None:
    """Registers the requests of a newly dispatched group."""
    self._ledger.register(records)

  def admit(self, result: datatypes.RolloutResult) -> request_ledger.Admission:
    """Offers a result to the underlying ledger (incarnation gate + dedup)."""
    return self._ledger.admit(result)

  def drain_ready(self) -> list[AssembledGroup]:
    """Emits and releases every currently-complete group.

    Returns:
      Complete groups, each with all G members ordered by sample_index. Emitted
      groups are released from the ledger (their retention ends here).

    Raises:
      ValueError: If a complete group has fewer than `min_group_size` members
        (I1: never emit a degenerate group).
    """
    ready: list[AssembledGroup] = []
    for group_id in self._ledger.complete_group_ids():
      pairs = self._ledger.group_pairs(group_id)
      if len(pairs) < self._min_group_size:
        raise ValueError(
            f"group {group_id!r} completed with {len(pairs)} member(s); "
            f"minimum is {self._min_group_size} (never emit a G=1 group)"
        )
      ready.append(AssembledGroup(group_id=group_id, members=pairs))
      self._ledger.release_group(group_id)
    return ready

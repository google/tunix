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

"""Admits only whole, healthy groups into training.

Group-relative advantages are computed across the members of a group, so the
math assumes a fixed number of well-formed rows. The pool, by design, reports
failures in band: a request that could not be generated comes back as a
response carrying an error rather than disappearing. Nothing downstream looked
at that, so a failed member would flow into the advantage computation as if it
were a trajectory, and a group missing a member would be reshaped against a
count it no longer had.

This is the seam that stops it. A group is either handed over complete and
entirely successful, or it is dropped whole and reported. A group is never
trimmed to the members that happened to work: training on a partial group is
not a smaller version of the same update, it is a different one, and a
one-member group has no group-relative signal at all.

Grouping is keyed by the group id carried on the request, never by position,
because responses arrive in completion order.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from absl import logging

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rollout_response_adapter

SUCCESS_STATUS = "SUCCEEDED"


class DropReason(enum.Enum):
  """Why a group did not reach training."""

  MEMBER_FAILED = "member_failed"
  INCOMPLETE = "incomplete"
  STALE = "stale"


@dataclasses.dataclass(frozen=True)
class DroppedGroup:
  """A group that will not be trained on, and why.

  Attributes:
    group_id: The group's id.
    reason: What disqualified it.
    detail: Human-readable specifics, for logs and metrics.
  """

  group_id: Any
  reason: DropReason
  detail: str


@dataclasses.dataclass(frozen=True)
class GatedGroups:
  """The outcome of gating one batch of rollouts.

  Attributes:
    complete: Group id to its members, in request order, for groups that are
      whole and successful.
    dropped: Groups that did not qualify.
  """

  complete: Mapping[Any, Sequence[rollout_response_adapter.TrajectoryItem]]
  dropped: Sequence[DroppedGroup]

  @property
  def items(self) -> list[rollout_response_adapter.TrajectoryItem]:
    """Every admitted member, grouped together and in request order."""
    admitted = []
    for members in self.complete.values():
      admitted.extend(members)
    return admitted


def gate_groups(
    requests: Sequence[datatypes.RolloutRequest],
    responses: Sequence[datatypes.RolloutResponse],
    *,
    group_size: int,
    tokenizer: Any = None,
    success_status: str = SUCCESS_STATUS,
    min_policy_version: Optional[int] = None,
) -> GatedGroups:
  """Splits a batch into groups that may train and groups that may not.

  Args:
    requests: The requests that were issued; they carry the group ids and the
      dataset rows the responses do not echo.
    responses: What came back, in any order.
    group_size: How many members a group must have to be complete.
    tokenizer: Used to reconstruct completion text when a worker did not send
      it.
    success_status: The status a member must report to count as healthy.
    min_policy_version: Oldest weights a group may have been generated from.
      A group is judged by its oldest member, not its average: mixing weight
      versions inside one group is what makes the comparison between its
      members meaningless. None disables the check.

  Returns:
    The admitted groups and the dropped ones.

  Raises:
    ValueError: If `group_size` is not positive.
  """
  if group_size < 1:
    raise ValueError(f"group_size must be >= 1, got {group_size}.")

  by_request_id = {request.request_id: request for request in requests}
  expected: dict[Any, list[datatypes.RolloutRequest]] = (
      collections.defaultdict(list)
  )
  for request in requests:
    expected[request.group_id].append(request)

  answered: dict[Any, dict[str, datatypes.RolloutResponse]] = (
      collections.defaultdict(dict)
  )
  for response in responses:
    request = by_request_id.get(response.request_id)
    if request is None:
      logging.warning(
          "Ignoring rollout response %r: no request in this batch claims it.",
          response.request_id,
      )
      continue
    answered[request.group_id][response.request_id] = response

  complete: dict[Any, list[rollout_response_adapter.TrajectoryItem]] = {}
  dropped: list[DroppedGroup] = []
  for group_id, group_requests in expected.items():
    group_responses = answered.get(group_id, {})
    drop = _disqualify(
        group_id,
        group_requests,
        group_responses,
        group_size,
        success_status,
        min_policy_version,
    )
    if drop is not None:
      dropped.append(drop)
      continue
    complete[group_id] = [
        rollout_response_adapter.to_trajectory_item(
            group_responses[request.request_id],
            request,
            tokenizer=tokenizer,
        )
        for request in group_requests
    ]
  return GatedGroups(complete=complete, dropped=dropped)


def _disqualify(
    group_id: Any,
    group_requests: Sequence[datatypes.RolloutRequest],
    group_responses: Mapping[str, datatypes.RolloutResponse],
    group_size: int,
    success_status: str,
    min_policy_version: Optional[int],
) -> Optional[DroppedGroup]:
  """Returns why the group cannot train, or None if it can."""
  if len(group_requests) != group_size:
    return DroppedGroup(
        group_id,
        DropReason.INCOMPLETE,
        f"{len(group_requests)} requests were issued for a group of"
        f" {group_size}.",
    )

  missing = [
      request.request_id
      for request in group_requests
      if request.request_id not in group_responses
  ]
  if missing:
    return DroppedGroup(
        group_id,
        DropReason.INCOMPLETE,
        f"no response for {sorted(missing)}.",
    )

  failed = [
      request_id
      for request_id, response in group_responses.items()
      if response.error is not None or response.status != success_status
  ]
  if failed:
    return DroppedGroup(
        group_id,
        DropReason.MEMBER_FAILED,
        f"{sorted(failed)} did not succeed.",
    )

  if min_policy_version is not None:
    oldest = min(
        response.policy_version for response in group_responses.values()
    )
    if oldest < min_policy_version:
      return DroppedGroup(
          group_id,
          DropReason.STALE,
          f"generated from weight version {oldest}, older than the"
          f" {min_policy_version} required.",
      )
  return None


def log_dropped(dropped: Iterable[DroppedGroup]) -> None:
  """Reports dropped groups, so a silently shrinking batch is visible."""
  for group in dropped:
    logging.warning(
        "Dropping rollout group %r before training: %s (%s)",
        group.group_id,
        group.detail,
        group.reason.value,
    )

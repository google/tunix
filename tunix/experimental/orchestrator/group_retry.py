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

"""Refilling the places in a group that came back empty.

A group has to be complete to train on, and until now a group with a failed
member was simply dropped. That is safe and wasteful: the surviving members
were generated and thrown away, and a fleet losing an occasional worker would
lose whole steps rather than a few trajectories.

Retrying happens here, above the dispatcher, and not inside it. A running
dispatch session cannot take new work -- its completion stream ends when
nothing is left in flight -- so a retry is a fresh batch containing only the
places still unfilled, issued after the previous one has finished.

A retry is a new request for the same place in the group. It gets a new
request id, because the old one may yet come back, and both are then results
for one slot; the ledger is what decides which counts. Everything issued is
recorded there before it is dispatched, so a straggler answering late is
recognized as the duplicate it is rather than as an extra member.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from absl import logging

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import request_ledger as ledger_lib


@dataclasses.dataclass(frozen=True)
class RetryOutcome:
  """What a group-retrying generation achieved.

  Attributes:
    responses: The admitted result for each filled place, grouped by group id
      and ordered within a group.
    complete: Groups that ended up whole.
    incomplete: Groups still missing members when the retry budget ran out.
    attempts: How many dispatch rounds were issued.
  """

  responses: dict[str, list[datatypes.RolloutResponse]]
  complete: list[str]
  incomplete: list[str]
  attempts: int


def _default_retry_request(
    record: ledger_lib.RequestRecord, attempt: int
) -> datatypes.RolloutRequest:
  """Reissues a request for the same place under a new id."""
  return dataclasses.replace(
      record.request,
      request_id=f"{record.request.request_id}:retry{attempt}",
  )


async def generate_groups(
    pool: Any,
    records: Sequence[ledger_lib.RequestRecord],
    ledger: ledger_lib.RequestLedger,
    *,
    group_size: int,
    max_attempts: int = 2,
    success_status: str = "SUCCEEDED",
    retry_request_fn: Callable[
        [ledger_lib.RequestRecord, int], datatypes.RolloutRequest
    ] = _default_retry_request,
) -> RetryOutcome:
  """Generates groups, refilling places that fail, up to a budget.

  Args:
    pool: Something with an async `generate(requests)`.
    records: What to generate, one record per place in each group. Registered
      with the ledger here if they are not already.
    ledger: Decides which results count.
    group_size: Members a group needs to be whole.
    max_attempts: Dispatch rounds allowed, including the first. One means no
      retrying.
    success_status: The status a result must carry to fill its place.
    retry_request_fn: Builds the reissued request for a place. The default
      keeps everything but the id, which must change.

  Returns:
    What was filled and what was not.

  Raises:
    ValueError: If `max_attempts` is not positive.
  """
  if max_attempts < 1:
    raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

  pending = list(records)
  for record in pending:
    if ledger.record_for(record.request_id) is None:
      ledger.register([record])

  attempts = 0
  while pending and attempts < max_attempts:
    attempts += 1
    responses = await pool.generate([record.request for record in pending])
    _admit_all(ledger, responses, success_status)

    unfilled = _unfilled(ledger, pending, group_size)
    if not unfilled:
      pending = []
      break
    if attempts >= max_attempts:
      pending = unfilled
      break

    logging.warning(
        "Retrying %d unfilled group members (round %d of %d).",
        len(unfilled),
        attempts + 1,
        max_attempts,
    )
    pending = _reissue(ledger, unfilled, attempts, retry_request_fn)

  return _summarize(ledger, records, group_size, attempts)


def _admit_all(
    ledger: ledger_lib.RequestLedger,
    responses: Any,
    success_status: str,
) -> None:
  """Offers each successful response to the ledger."""
  if isinstance(responses, datatypes.RolloutResponse):
    responses = [responses]
  for response in responses:
    if response.error is not None or response.status != success_status:
      # A failed result does not fill its place; the place stays open for a
      # retry rather than being filled with a failure.
      continue
    ledger.admit(response)


def _unfilled(
    ledger: ledger_lib.RequestLedger,
    records: Sequence[ledger_lib.RequestRecord],
    group_size: int,
) -> list[ledger_lib.RequestRecord]:
  """The records whose place in their group is still empty."""
  del group_size  # Completeness is per place, not per group, at this point.
  still_open = []
  for record in records:
    if record.sample_index in ledger.missing_slots(record.group_id):
      still_open.append(record)
  return still_open


def _reissue(
    ledger: ledger_lib.RequestLedger,
    records: Sequence[ledger_lib.RequestRecord],
    attempt: int,
    retry_request_fn: Callable[
        [ledger_lib.RequestRecord, int], datatypes.RolloutRequest
    ],
) -> list[ledger_lib.RequestRecord]:
  """Registers a fresh attempt for each still-open place."""
  reissued = []
  for record in records:
    retry = ledger_lib.RequestRecord(
        request=retry_request_fn(record, attempt),
        group_id=record.group_id,
        sample_index=record.sample_index,
        incarnation=record.incarnation,
        attempt=record.attempt + 1,
    )
    ledger.register([retry])
    reissued.append(retry)
  return reissued


def _summarize(
    ledger: ledger_lib.RequestLedger,
    records: Sequence[ledger_lib.RequestRecord],
    group_size: int,
    attempts: int,
) -> RetryOutcome:
  """Reports which groups came out whole."""
  group_ids = []
  for record in records:
    if record.group_id not in group_ids:
      group_ids.append(record.group_id)

  responses: dict[str, list[datatypes.RolloutResponse]] = {}
  complete, incomplete = [], []
  for group_id in group_ids:
    admitted = ledger.accepted(group_id)
    responses[group_id] = admitted
    if len(admitted) >= group_size:
      complete.append(group_id)
    else:
      incomplete.append(group_id)
      logging.error(
          "Group %r is still missing places %s after %d attempts; it will not"
          " be trained on.",
          group_id,
          ledger.missing_slots(group_id),
          attempts,
      )
  return RetryOutcome(
      responses=responses,
      complete=complete,
      incomplete=incomplete,
      attempts=attempts,
  )

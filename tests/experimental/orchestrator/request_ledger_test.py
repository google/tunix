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

"""Retries make delivery at-least-once; the ledger keeps it counting once."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import request_ledger


def _record(
    group_id: str,
    sample_index: int,
    *,
    request_id: str = "",
    incarnation: int = 0,
    attempt: int = 0,
) -> request_ledger.RequestRecord:
  request_id = request_id or f"{group_id}-{sample_index}-a{attempt}"
  return request_ledger.RequestRecord(
      request=datatypes.RolloutRequest(
          request_id=request_id,
          prompt="p",
          prompt_id=group_id,
          group_id=group_id,
          sample_index=sample_index,
          incarnation=incarnation,
      ),
      group_id=group_id,
      sample_index=sample_index,
      incarnation=incarnation,
      attempt=attempt,
  )


def _response(request_id: str) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id, status="SUCCEEDED", policy_version=1
  )


class RequestLedgerTest(absltest.TestCase):

  def test_a_result_for_a_dispatched_request_is_accepted(self):
    ledger = request_ledger.RequestLedger(group_size=2)
    ledger.register([_record("g0", 0), _record("g0", 1)])

    self.assertEqual(
        ledger.admit(_response("g0-0-a0")), request_ledger.Admission.ACCEPTED
    )

  def test_a_result_nobody_asked_for_is_refused(self):
    ledger = request_ledger.RequestLedger()

    self.assertEqual(
        ledger.admit(_response("never-sent")),
        request_ledger.Admission.UNKNOWN,
    )

  def test_a_straggler_cannot_double_count_against_its_retry(self):
    """The case retries exist to survive, and the reason for slot keying."""
    ledger = request_ledger.RequestLedger(group_size=2)
    original = _record("g0", 0, attempt=0)
    retry = _record("g0", 0, attempt=1)
    ledger.register([original, _record("g0", 1)])
    ledger.register([retry])

    # The retry answers first; the original turns up afterwards.
    self.assertEqual(
        ledger.admit(_response(retry.request_id)),
        request_ledger.Admission.ACCEPTED,
    )
    self.assertEqual(
        ledger.admit(_response(original.request_id)),
        request_ledger.Admission.DUPLICATE,
    )
    # One result for that place in the group, not two.
    ledger.admit(_response("g0-1-a0"))
    self.assertLen(ledger.accepted("g0"), 2)

  def test_the_same_result_twice_counts_once(self):
    ledger = request_ledger.RequestLedger(group_size=1)
    ledger.register([_record("g0", 0)])

    first = ledger.admit(_response("g0-0-a0"))
    second = ledger.admit(_response("g0-0-a0"))

    self.assertEqual(first, request_ledger.Admission.ACCEPTED)
    self.assertEqual(second, request_ledger.Admission.DUPLICATE)

  def test_a_result_from_an_abandoned_lineage_is_refused(self):
    """A restart reissues the same group ids; only the epoch tells them apart."""
    ledger = request_ledger.RequestLedger(group_size=1)
    ledger.register([_record("g0", 0, incarnation=0)])
    ledger.advance_incarnation()

    self.assertEqual(
        ledger.admit(_response("g0-0-a0")),
        request_ledger.Admission.STALE_INCARNATION,
    )

  def test_lineage_is_checked_before_duplication(self):
    """Otherwise a discarded result could claim a slot in the new lineage."""
    ledger = request_ledger.RequestLedger(group_size=1)
    old = _record("g0", 0, request_id="old", incarnation=0)
    ledger.register([old])
    ledger.advance_incarnation()
    ledger.register([_record("g0", 0, request_id="new", incarnation=1)])

    self.assertEqual(
        ledger.admit(_response("old")),
        request_ledger.Admission.STALE_INCARNATION,
    )
    self.assertEqual(
        ledger.admit(_response("new")), request_ledger.Admission.ACCEPTED
    )

  def test_registering_a_reused_request_id_is_refused(self):
    ledger = request_ledger.RequestLedger()
    ledger.register([_record("g0", 0)])

    with self.assertRaises(ValueError):
      ledger.register([_record("g0", 0)])

  def test_registering_for_another_lineage_is_refused(self):
    ledger = request_ledger.RequestLedger()

    with self.assertRaises(ValueError):
      ledger.register([_record("g0", 0, incarnation=3)])

  def test_completeness_tracks_filled_places(self):
    ledger = request_ledger.RequestLedger(group_size=2)
    ledger.register([_record("g0", 0), _record("g0", 1)])

    ledger.admit(_response("g0-0-a0"))
    self.assertFalse(ledger.is_group_complete("g0"))
    self.assertEqual(ledger.missing_slots("g0"), [1])

    ledger.admit(_response("g0-1-a0"))
    self.assertTrue(ledger.is_group_complete("g0"))
    self.assertEmpty(ledger.missing_slots("g0"))

  def test_results_come_back_in_group_order(self):
    ledger = request_ledger.RequestLedger(group_size=3)
    ledger.register([_record("g0", i) for i in range(3)])

    for index in (2, 0, 1):
      ledger.admit(_response(f"g0-{index}-a0"))

    self.assertEqual(
        [r.request_id for r in ledger.accepted("g0")],
        ["g0-0-a0", "g0-1-a0", "g0-2-a0"],
    )

  def test_the_originating_request_stays_reachable(self):
    """Assembly needs the prompt behind a result, which the result omits."""
    ledger = request_ledger.RequestLedger(group_size=1)
    ledger.register([_record("g0", 0)])

    record = ledger.record_for("g0-0-a0")

    self.assertIsNotNone(record)
    self.assertEqual(record.group_id, "g0")
    self.assertEqual(record.request.prompt, "p")

  def test_releasing_a_group_forgets_it(self):
    ledger = request_ledger.RequestLedger(group_size=1)
    ledger.register([_record("g0", 0)])
    ledger.admit(_response("g0-0-a0"))

    ledger.release_group("g0")

    self.assertEmpty(ledger)
    self.assertIsNone(ledger.record_for("g0-0-a0"))
    self.assertFalse(ledger.is_group_complete("g0"))

  def test_groups_do_not_interfere(self):
    ledger = request_ledger.RequestLedger(group_size=1)
    ledger.register([_record("g0", 0), _record("g1", 0)])

    ledger.admit(_response("g0-0-a0"))

    self.assertTrue(ledger.is_group_complete("g0"))
    self.assertFalse(ledger.is_group_complete("g1"))


if __name__ == "__main__":
  absltest.main()

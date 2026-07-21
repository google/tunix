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

"""Tests for the RequestLedger (intake gate, dedup, retention)."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import request_ledger


def _request(request_id: str) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=request_id,
      prompt_id="p",
      prompt_text="hi",
      sampling_params=datatypes.SamplingParams(max_tokens=4),
  )


def _record(group_id, sample_index, *, request_id=None, incarnation=0):
  request_id = request_id or f"{group_id}:{sample_index}"
  return request_ledger.RequestRecord(
      request=_request(request_id),
      group_id=group_id,
      sample_index=sample_index,
      incarnation=incarnation,
  )


def _result(request_id, *, policy_version=0, status="COMPLETED"):
  return datatypes.RolloutResult(
      request_id=request_id, prompt_id="p", status=status,
      policy_version=policy_version,
  )


def _group(group_id, size, *, incarnation=0):
  return [_record(group_id, i, incarnation=incarnation) for i in range(size)]


class RequestLedgerTest(absltest.TestCase):

  def test_admit_accepts_and_completes_group(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 3))
    self.assertEqual(ledger.group_size("g0"), 3)
    self.assertFalse(ledger.is_group_complete("g0"))

    self.assertEqual(ledger.admit(_result("g0:0")), request_ledger.Admission.ACCEPTED)
    self.assertEqual(ledger.admit(_result("g0:1")), request_ledger.Admission.ACCEPTED)
    self.assertFalse(ledger.is_group_complete("g0"))
    self.assertEqual(ledger.admit(_result("g0:2")), request_ledger.Admission.ACCEPTED)
    self.assertTrue(ledger.is_group_complete("g0"))

  def test_first_wins_dedup_on_slot(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 2))
    first = _result("g0:0", policy_version=1)
    self.assertEqual(ledger.admit(first), request_ledger.Admission.ACCEPTED)
    # A second delivery for the same slot is dropped; the first result stands.
    second = _result("g0:0", policy_version=99)
    self.assertEqual(ledger.admit(second), request_ledger.Admission.DUPLICATE)
    ledger.admit(_result("g0:1"))
    kept = dict((rec.sample_index, res) for rec, res in ledger.group_pairs("g0"))
    self.assertEqual(kept[0].policy_version, 1)

  def test_unknown_request_rejected(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 2))
    self.assertEqual(
        ledger.admit(_result("does-not-exist")), request_ledger.Admission.UNKNOWN
    )

  def test_incarnation_gate_rejects_prior_lineage(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 2))
    ledger.advance_incarnation()  # e.g. after a rewind.
    self.assertEqual(ledger.incarnation, 1)
    # A result from the discarded lineage is no longer known.
    self.assertEqual(ledger.admit(_result("g0:0")), request_ledger.Admission.UNKNOWN)
    # Fresh requests at the new incarnation admit normally.
    ledger.register(_group("g0", 2, incarnation=1))
    self.assertEqual(ledger.admit(_result("g0:0")), request_ledger.Admission.ACCEPTED)

  def test_group_pairs_ordered_by_sample_index(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 3))
    for i in (2, 0, 1):  # admit out of order
      ledger.admit(_result(f"g0:{i}"))
    pairs = ledger.group_pairs("g0")
    self.assertEqual([rec.sample_index for rec, _ in pairs], [0, 1, 2])

  def test_release_group_frees_state(self):
    ledger = request_ledger.RequestLedger()
    ledger.register(_group("g0", 2))
    ledger.admit(_result("g0:0"))
    ledger.admit(_result("g0:1"))
    ledger.release_group("g0")
    self.assertEqual(ledger.inflight_group_count(), 0)
    self.assertEqual(ledger.group_size("g0"), 0)
    # A late duplicate after release is now unknown (retention ended).
    self.assertEqual(ledger.admit(_result("g0:0")), request_ledger.Admission.UNKNOWN)

  def test_register_rejects_duplicate_request_id(self):
    ledger = request_ledger.RequestLedger()
    ledger.register([_record("g0", 0, request_id="dup")])
    with self.assertRaises(ValueError):
      ledger.register([_record("g0", 1, request_id="dup")])

  def test_register_rejects_wrong_incarnation(self):
    ledger = request_ledger.RequestLedger()
    with self.assertRaises(ValueError):
      ledger.register([_record("g0", 0, incarnation=5)])


if __name__ == "__main__":
  absltest.main()

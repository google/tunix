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

"""Tests for the GroupAssembler (ID-based grouping, I1) and staleness helper."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import group_assembler
from tunix.experimental.orchestrator import request_ledger


def _request(request_id: str) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=request_id,
      prompt_id="p",
      prompt_text="hi",
      sampling_params=datatypes.SamplingParams(max_tokens=4),
  )


def _record(group_id, sample_index):
  return request_ledger.RequestRecord(
      request=_request(f"{group_id}:{sample_index}"),
      group_id=group_id,
      sample_index=sample_index,
  )


def _result(request_id, *, policy_version=0):
  return datatypes.RolloutResult(
      request_id=request_id, prompt_id="p", status="COMPLETED",
      policy_version=policy_version,
  )


def _group(group_id, size):
  return [_record(group_id, i) for i in range(size)]


class GroupAssemblerTest(absltest.TestCase):

  def test_drain_emits_only_complete_groups(self):
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(_group("g0", 2))
    assembler.open_group(_group("g1", 2))
    assembler.admit(_result("g0:0"))
    assembler.admit(_result("g0:1"))
    assembler.admit(_result("g1:0"))  # g1 still incomplete

    ready = assembler.drain_ready()
    self.assertEqual([g.group_id for g in ready], ["g0"])
    self.assertLen(ready[0], 2)
    # g0 was released; g1 remains in flight.
    self.assertEqual(assembler.ledger.inflight_group_count(), 1)

  def test_drain_is_idempotent_after_release(self):
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(_group("g0", 2))
    assembler.admit(_result("g0:0"))
    assembler.admit(_result("g0:1"))
    self.assertLen(assembler.drain_ready(), 1)
    self.assertEmpty(assembler.drain_ready())  # already released

  def test_grouping_is_by_id_not_arrival_order(self):
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(_group("g0", 2))
    assembler.open_group(_group("g1", 2))
    # Interleave arrivals across groups.
    for rid in ("g1:1", "g0:1", "g0:0", "g1:0"):
      assembler.admit(_result(rid))
    ready = {g.group_id: g for g in assembler.drain_ready()}
    self.assertEqual(set(ready), {"g0", "g1"})
    for group_id, group in ready.items():
      for record, _ in group.members:
        self.assertEqual(record.group_id, group_id)

  def test_never_emits_degenerate_group(self):
    assembler = group_assembler.GroupAssembler(min_group_size=2)
    assembler.open_group(_group("solo", 1))
    assembler.admit(_result("solo:0"))
    with self.assertRaises(ValueError):
      assembler.drain_ready()

  def test_policy_version_is_group_minimum(self):
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(_group("g0", 2))
    assembler.admit(_result("g0:0", policy_version=5))
    assembler.admit(_result("g0:1", policy_version=3))
    (group,) = assembler.drain_ready()
    self.assertEqual(group.policy_version(), 3)

  def test_is_group_stale_is_group_atomic(self):
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(_group("g0", 2))
    assembler.admit(_result("g0:0", policy_version=5))
    assembler.admit(_result("g0:1", policy_version=3))
    (group,) = assembler.drain_ready()
    # group version is 3 (the min).
    self.assertTrue(
        group_assembler.is_group_stale(group, current_version=6, max_staleness=2)
    )
    self.assertFalse(
        group_assembler.is_group_stale(group, current_version=4, max_staleness=2)
    )


if __name__ == "__main__":
  absltest.main()

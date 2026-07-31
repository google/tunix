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

"""Fault injection for the seam that guards group-relative training.

Advantages are computed across a group, so a failed, missing, or duplicated
member is not a smaller version of the same update -- it is a different one.
Each case below is injected into a batch and asserted never to reach the
admitted set.
"""

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import group_gate


def _request(index: int, group_id: str) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=f"{group_id}-{index}",
      prompt={"prompts": f"prompt {index}"},
      prompt_id=group_id,
      group_id=group_id,
  )


def _response(
    request: datatypes.RolloutRequest, status: str = "SUCCEEDED"
) -> datatypes.RolloutResponse:
  tokens = np.array([1, 2], dtype=np.int32)
  return datatypes.RolloutResponse(
      request_id=request.request_id,
      status=status,
      prompt_tokens=tokens,
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=tokens,
              loss_mask=np.ones_like(tokens),
          )
      ],
      policy_version=1,
  )


def _failed(request: datatypes.RolloutRequest) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request.request_id,
      status="FAILED",
      error=datatypes.ErrorInfo(error_type="RuntimeError", message="boom"),
  )


def _group(group_id: str, size: int = 2):
  return [_request(i, group_id) for i in range(size)]


class GroupGateTest(absltest.TestCase):

  def test_admits_a_complete_healthy_group(self):
    requests = _group("g0")
    responses = [_response(r) for r in requests]

    gated = group_gate.gate_groups(requests, responses, group_size=2)

    self.assertEmpty(gated.dropped)
    self.assertLen(gated.complete["g0"], 2)
    self.assertEqual(
        [item.group_id for item in gated.items], ["g0", "g0"]
    )

  def test_a_failed_member_drops_its_whole_group(self):
    requests = _group("g0")
    responses = [_response(requests[0]), _failed(requests[1])]

    gated = group_gate.gate_groups(requests, responses, group_size=2)

    self.assertEmpty(gated.complete)
    self.assertLen(gated.dropped, 1)
    self.assertEqual(gated.dropped[0].reason, group_gate.DropReason.MEMBER_FAILED)
    # Crucially, the surviving member does not train on its own.
    self.assertEmpty(gated.items)

  def test_a_missing_member_drops_its_whole_group(self):
    requests = _group("g0")
    responses = [_response(requests[0])]

    gated = group_gate.gate_groups(requests, responses, group_size=2)

    self.assertEmpty(gated.items)
    self.assertEqual(gated.dropped[0].reason, group_gate.DropReason.INCOMPLETE)

  def test_one_bad_group_does_not_take_the_healthy_ones_with_it(self):
    good, bad = _group("good"), _group("bad")
    responses = [
        _response(good[0]),
        _response(good[1]),
        _response(bad[0]),
        _failed(bad[1]),
    ]

    gated = group_gate.gate_groups(good + bad, responses, group_size=2)

    self.assertEqual(list(gated.complete), ["good"])
    self.assertLen(gated.items, 2)
    self.assertLen(gated.dropped, 1)

  def test_groups_are_keyed_by_id_not_arrival_order(self):
    """Responses come back in completion order; grouping must not depend on it."""
    first, second = _group("g0"), _group("g1")
    requests = first + second
    responses = [
        _response(second[1]),
        _response(first[1]),
        _response(second[0]),
        _response(first[0]),
    ]

    gated = group_gate.gate_groups(requests, responses, group_size=2)

    self.assertEmpty(gated.dropped)
    # Members are restored to request order within their group.
    self.assertEqual(
        [item.traj["request_id"] for item in gated.complete["g0"]],
        ["g0-0", "g0-1"],
    )

  def test_a_group_issued_short_never_trains(self):
    """A group of one has no group-relative signal at all."""
    requests = _group("g0", size=1)

    gated = group_gate.gate_groups(
        requests, [_response(requests[0])], group_size=2
    )

    self.assertEmpty(gated.items)
    self.assertEqual(gated.dropped[0].reason, group_gate.DropReason.INCOMPLETE)

  def test_a_duplicate_response_cannot_stand_in_for_a_missing_member(self):
    requests = _group("g0")
    duplicate = _response(requests[0])

    gated = group_gate.gate_groups(
        requests, [_response(requests[0]), duplicate], group_size=2
    )

    self.assertEmpty(gated.items)
    self.assertEqual(gated.dropped[0].reason, group_gate.DropReason.INCOMPLETE)

  def test_a_response_from_outside_the_batch_is_ignored(self):
    requests = _group("g0")
    stranger = datatypes.RolloutResponse(
        request_id="not-ours", status="SUCCEEDED"
    )

    gated = group_gate.gate_groups(
        requests,
        [_response(requests[0]), _response(requests[1]), stranger],
        group_size=2,
    )

    self.assertLen(gated.items, 2)
    self.assertEmpty(gated.dropped)

  def test_a_group_generated_from_old_weights_is_dropped(self):
    requests = _group("g0")
    fresh = _response(requests[0])
    fresh.policy_version = 5
    stale = _response(requests[1])
    stale.policy_version = 2

    gated = group_gate.gate_groups(
        requests, [fresh, stale], group_size=2, min_policy_version=5
    )

    self.assertEmpty(gated.items)
    self.assertEqual(gated.dropped[0].reason, group_gate.DropReason.STALE)

  def test_a_group_is_judged_by_its_oldest_member(self):
    """Mixed versions inside a group make its members incomparable."""
    requests = _group("g0")
    responses = [_response(r) for r in requests]
    for response in responses:
      response.policy_version = 5

    fresh_enough = group_gate.gate_groups(
        requests, responses, group_size=2, min_policy_version=5
    )
    self.assertLen(fresh_enough.items, 2)

    responses[1].policy_version = 4
    now_mixed = group_gate.gate_groups(
        requests, responses, group_size=2, min_policy_version=5
    )
    self.assertEmpty(now_mixed.items)

  def test_staleness_checking_is_optional(self):
    requests = _group("g0")
    responses = [_response(r) for r in requests]

    gated = group_gate.gate_groups(requests, responses, group_size=2)

    self.assertLen(gated.items, 2)

  def test_rejects_a_nonsense_group_size(self):
    with self.assertRaises(ValueError):
      group_gate.gate_groups([], [], group_size=0)


if __name__ == "__main__":
  absltest.main()

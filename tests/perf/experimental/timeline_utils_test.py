# Copyright 2026 Google LLC
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

from __future__ import annotations

import dataclasses
import threading

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.perf.experimental import timeline
from tunix.perf.experimental import timeline_utils


@dataclasses.dataclass
class MockDevice:
  platform: str
  id: int


class TimelineIdGenerationTest(parameterized.TestCase):

  def test_generate_host_timeline_id(self):
    tid = timeline_utils.generate_host_timeline_id()
    self.assertStartsWith(tid, "host-")
    self.assertIn(str(threading.get_ident()), tid)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_timeline",
          spans_to_add=[],
          allowed_types=["rollout"],
          expected=False,
      ),
      dict(
          testcase_name="one_allowed_span",
          spans_to_add=["rollout"],
          allowed_types=["rollout"],
          expected=True,
      ),
      dict(
          testcase_name="multiple_allowed_spans",
          spans_to_add=["rollout", "peft_train"],
          allowed_types=["rollout", "peft_train"],
          expected=True,
      ),
      dict(
          testcase_name="unallowed_span",
          spans_to_add=["rollout", "peft_train"],
          allowed_types=["rollout"],
          expected=False,
      ),
      dict(
          testcase_name="unallowed_overlap",
          spans_to_add=["rollout", "peft_train"],
          allowed_types=["peft_train", "inference"],
          expected=False,
      ),
      dict(
          testcase_name="different_allowed_span",
          spans_to_add=["inference"],
          allowed_types=["inference", "rollout"],
          expected=True,
      ),
  )
  def test_is_timeline_only_of_allowed_type(
      self, spans_to_add, allowed_types, expected
  ):
    tl = timeline.Timeline("test", 0)
    for i, span_name in enumerate(spans_to_add):
      tl.start_span(span_name, float(i))
      tl.stop_span(float(i) + 0.5)

    with self.subTest("without_commit_no_cur_step"):
      # without current step it shouldn't see anything without a commit
      self.assertFalse(
          timeline_utils.is_timeline_only_of_allowed_type(tl, allowed_types)
      )

    with self.subTest("without_commit_with_cur_step"):
      # Using include_current_step should match expected
      self.assertEqual(
          timeline_utils.is_timeline_only_of_allowed_type(
              tl, allowed_types, include_cur_step=True
          ),
          expected,
      )

    tl.commit_step()

    with self.subTest("after_commit_no_cur_step"):
      # After commit, default should match expected with or without
      # include_cur_step
      self.assertEqual(
          timeline_utils.is_timeline_only_of_allowed_type(tl, allowed_types),
          expected,
      )

    with self.subTest("after_commit_with_cur_step"):
      self.assertEqual(
          timeline_utils.is_timeline_only_of_allowed_type(
              tl, allowed_types, include_cur_step=True
          ),
          expected,
      )

    if expected:
      # Add an unallowed span to the current step
      tl.start_span("unallowed_span_xyz", 10.0)
      tl.stop_span(10.5)

      with self.subTest("unallowed_span_in_cur_step_default"):
        # Default (include_cur_step=False) ignores cur_step
        self.assertTrue(
            timeline_utils.is_timeline_only_of_allowed_type(tl, allowed_types)
        )

      with self.subTest("unallowed_span_in_cur_step_included"):
        # With include_cur_step=True, the unallowed span causes a False return
        self.assertFalse(
            timeline_utils.is_timeline_only_of_allowed_type(
                tl, allowed_types, include_cur_step=True
            )
        )

  @parameterized.named_parameters(
      dict(testcase_name="host_with_id", tl_id="host-12345", expected=True),
      dict(testcase_name="host_only", tl_id="host-", expected=True),
      dict(testcase_name="device", tl_id="tpu0", expected=False),
      dict(testcase_name="device_no_number", tl_id="tpu", expected=False),
      dict(testcase_name="empty", tl_id="", expected=False),
  )
  def test_is_host_timeline(self, tl_id, expected):
    self.assertEqual(timeline_utils.is_host_timeline(tl_id), expected)

  @parameterized.named_parameters(
      dict(testcase_name="string", device_id="tpu0", expected_id="tpu0"),
      dict(
          testcase_name="device_object",
          device_id=MockDevice("gpu", 7),
          expected_id="gpu7",
      ),
  )
  def test_generate_device_timeline_id(self, device_id, expected_id):
    self.assertEqual(
        timeline_utils.generate_device_timeline_id(device_id), expected_id
    )

  @dataclasses.dataclass
  class MockDeviceNoId:
    platform: str

  @dataclasses.dataclass
  class MockDeviceNoPlatform:
    id: int

  @parameterized.named_parameters(
      dict(testcase_name="int", invalid_device=123),
      dict(testcase_name="missing_id", invalid_device=MockDeviceNoId("gpu")),
      dict(
          testcase_name="missing_platform",
          invalid_device=MockDeviceNoPlatform(7),
      ),
  )
  def test_generate_device_timeline_id_error(self, invalid_device):
    with self.assertRaisesRegex(ValueError, "Unsupported id type"):
      timeline_utils.generate_device_timeline_id(invalid_device)

  @parameterized.named_parameters(
      dict(testcase_name="none", devices=None, expected_ids=[]),
      dict(testcase_name="empty_list", devices=[], expected_ids=[]),
      dict(
          testcase_name="empty_numpy_array",
          devices=np.array([]),
          expected_ids=[],
      ),
      dict(
          testcase_name="mixed_list",
          devices=["dev1", MockDevice("tpu", 0)],
          expected_ids=["dev1", "tpu0"],
      ),
      dict(
          testcase_name="numpy_array",
          devices=np.array([MockDevice("tpu", 0), MockDevice("tpu", 1)]),
          expected_ids=["tpu0", "tpu1"],
      ),
      dict(
          testcase_name="numpy_array_2d",
          devices=np.array([
              [MockDevice("tpu", 0), MockDevice("tpu", 1)],
              [MockDevice("tpu", 2), MockDevice("tpu", 3)],
          ]),
          expected_ids=["tpu0", "tpu1", "tpu2", "tpu3"],
      ),
  )
  def test_generate_device_timeline_ids(self, devices, expected_ids):
    self.assertEqual(
        timeline_utils.generate_device_timeline_ids(devices), expected_ids
    )

  def test_sequentialize_overlapping_spans_standard(self):
    spans = {
        0: timeline.Span("a", 0.0, id=0),
        1: timeline.Span("b", 1.0, id=1),
    }
    spans[0].end = 2.0
    spans[1].end = 3.0

    active_spans, queue_spans = timeline_utils.sequentialize_overlapping_spans(spans)

    with self.subTest("active_spans"):
      self.assertEqual(active_spans[0].begin, 0.0)
      self.assertEqual(active_spans[0].end, 2.0)
      self.assertEqual(active_spans[1].begin, 2.0)
      self.assertEqual(active_spans[1].end, 3.0)

    with self.subTest("queue_spans"):
      self.assertIn(1, queue_spans)
      self.assertEqual(queue_spans[1].begin, 1.0)
      self.assertEqual(queue_spans[1].end, 2.0)

  def test_sequentialize_overlapping_spans_three_way_contention(self):
    spans = {
        0: timeline.Span("c", 5.0, id=0),
        1: timeline.Span("d", 6.0, id=1),
        2: timeline.Span("e", 6.5, id=2),
    }
    spans[0].end = 8.0
    spans[1].end = 9.0
    spans[2].end = 10.0

    active_spans, queue_spans = timeline_utils.sequentialize_overlapping_spans(spans)

    with self.subTest("active_spans"):
      self.assertEqual(active_spans[0].begin, 5.0)
      self.assertEqual(active_spans[0].end, 8.0)
      self.assertEqual(active_spans[1].begin, 8.0)
      self.assertEqual(active_spans[1].end, 9.0)
      self.assertEqual(active_spans[2].begin, 9.0)
      self.assertEqual(active_spans[2].end, 10.0)

    with self.subTest("queue_spans"):
      self.assertIn(1, queue_spans)
      self.assertEqual(queue_spans[1].begin, 6.0)
      self.assertEqual(queue_spans[1].end, 8.0)

      self.assertIn(2, queue_spans)
      self.assertEqual(queue_spans[2].begin, 8.0)
      self.assertEqual(queue_spans[2].end, 9.0)

  def test_sequentialize_overlapping_spans_subsumed_span(self):
    spans = {
        0: timeline.Span("f", 15.0, id=0),
        1: timeline.Span("g", 15.1, id=1),
    }
    spans[0].end = 18.0
    spans[1].end = 15.5

    active_spans, queue_spans = timeline_utils.sequentialize_overlapping_spans(spans)

    with self.subTest("active_membership"):
      self.assertIn(0, active_spans)
      self.assertNotIn(1, active_spans)

    with self.subTest("queue_membership"):
      self.assertIn(1, queue_spans)

  def test_sequentialize_overlapping_spans_empty(self):
    active_spans, queue_spans = timeline_utils.sequentialize_overlapping_spans({})
    self.assertEmpty(active_spans)
    self.assertEmpty(queue_spans)

  def test_generate_queued_timeline_id(self):
    self.assertEqual(
        timeline_utils.generate_queued_timeline_id("tpu0"), "tpu0_queue"
    )

  def test_is_queued_timeline(self):
    self.assertTrue(timeline_utils.is_queued_timeline("tpu0_queue"))
    self.assertFalse(timeline_utils.is_queued_timeline("tpu0"))


if __name__ == "__main__":
  absltest.main()

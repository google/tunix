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
from tunix.perf.experimental import constants
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

    self.assertEqual(
        timeline_utils.is_timeline_only_of_allowed_type(tl, allowed_types),
        expected,
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


class TimelineUtilsTest(absltest.TestCase):

  def test_deepcopy_timeline(self):
    tl = timeline.Timeline("test_tl", 0.0)
    s1 = tl.start_span("s1", 1.0, tags={"foo": "bar"})
    tl.stop_span(2.0)

    tl.start_span("s2", 3.0)  # Active span

    new_tl = timeline_utils.deepcopy_timeline(tl)

    self.assertIsNot(new_tl, tl)
    self.assertEqual(new_tl.id, tl.id)
    self.assertEqual(new_tl.born, tl.born)

    self.assertLen(new_tl.spans, 2)
    self.assertIsNot(new_tl.spans[0], tl.spans[0])
    self.assertEqual(new_tl.spans[0].name, "s1")
    self.assertEqual(new_tl.spans[0].end, 2.0)
    self.assertEqual(new_tl.spans[0].tags, {"foo": "bar"})

    self.assertIsNot(new_tl.spans[1], tl.spans[1])
    self.assertEqual(new_tl.spans[1].name, "s2")
    self.assertEqual(new_tl.spans[1].end, float("inf"))

    # Check that tags dict was deepcopied
    self.assertIsNot(new_tl.spans[0].tags, tl.spans[0].tags)

    # _active_spans should be copied correctly
    self.assertEqual(new_tl._active_spans, tl._active_spans)
    self.assertIsNot(new_tl._active_spans, tl._active_spans)

  def test_add_idle_spans_empty(self):
    tl = timeline.Timeline("test_tl", 0.0)
    new_tl = timeline_utils.add_idle_spans(tl)
    self.assertEmpty(new_tl.spans)

  def test_add_idle_spans_no_gaps(self):
    tl = timeline.Timeline("test_tl", 0.0)
    tl.start_span("s1", 0.0)
    tl.stop_span(1.0)
    tl.start_span("s2", 1.0)
    tl.stop_span(2.0)

    new_tl = timeline_utils.add_idle_spans(tl)
    self.assertLen(new_tl.spans, 2)
    spans = sorted(new_tl.spans.values(), key=lambda s: s.begin)
    self.assertEqual(spans[0].name, "s1")
    self.assertEqual(spans[1].name, "s2")

  def test_add_idle_spans_with_gaps(self):
    tl = timeline.Timeline("test_tl", 0.0)
    tl.start_span("s1", 1.0)
    tl.stop_span(2.0)
    tl.start_span("s2", 3.0)
    tl.stop_span(4.0)

    new_tl = timeline_utils.add_idle_spans(tl)
    self.assertLen(new_tl.spans, 4)
    spans = sorted(new_tl.spans.values(), key=lambda s: s.begin)

    self.assertEqual(spans[0].name, constants.IDLE)
    self.assertEqual(spans[0].begin, 0.0)
    self.assertEqual(spans[0].end, 1.0)

    self.assertEqual(spans[1].name, "s1")
    self.assertEqual(spans[1].begin, 1.0)
    self.assertEqual(spans[1].end, 2.0)

    self.assertEqual(spans[2].name, constants.IDLE)
    self.assertEqual(spans[2].begin, 2.0)
    self.assertEqual(spans[2].end, 3.0)

    self.assertEqual(spans[3].name, "s2")
    self.assertEqual(spans[3].begin, 3.0)
    self.assertEqual(spans[3].end, 4.0)

  def test_add_idle_spans_overlapping(self):
    tl = timeline.Timeline("test_tl", 0.0)
    # s1 from 1.0 to 3.0
    s1 = tl.start_span("s1", 1.0)
    # s2 from 2.0 to 4.0
    s2 = tl.start_span("s2", 2.0)
    tl.stop_span(4.0)  # stops s2
    tl.stop_span(3.0)  # stops s1 (wait, Timeline stack is LIFO!)
    # Actually Timeline requires stopping the top of the stack. So s2 must be stopped before s1.
    # The above is valid.

    # Let's manually add spans to test overlapping without stack restrictions
    tl2 = timeline.Timeline("test_tl2", 0.0)
    span1 = timeline.Span("s1", 1.0, 0)
    span1.end = 3.0
    tl2.spans[0] = span1

    span2 = timeline.Span("s2", 2.0, 1)
    span2.end = 4.0
    tl2.spans[1] = span2

    new_tl = timeline_utils.add_idle_spans(tl2)

    self.assertLen(new_tl.spans, 3)
    spans = sorted(new_tl.spans.values(), key=lambda s: s.begin)

    self.assertEqual(spans[0].name, constants.IDLE)
    self.assertEqual(spans[0].begin, 0.0)
    self.assertEqual(spans[0].end, 1.0)

    self.assertEqual(spans[1].name, "s1")
    self.assertEqual(spans[2].name, "s2")

  def test_add_idle_spans_ignore_active(self):
    tl = timeline.Timeline("test_tl", 0.0)
    # create an active span manually (since snapshot removes active tracked spans)
    span1 = timeline.Span("s1", 1.0, 0)
    span1.end = float("inf")
    tl.spans[0] = span1

    span2 = timeline.Span("s2", 2.0, 1)
    span2.end = 4.0
    tl.spans[1] = span2

    tl._last_span_id = 1

    # ignore_active_spans=True is the default
    new_tl = timeline_utils.add_idle_spans(tl, ignore_active_spans=True)

    # The active span s1 is ignored for gap calculation.
    # So gaps are 0.0 to 2.0 (before s2).
    self.assertLen(new_tl.spans, 3)  # s1, s2, idle
    spans = sorted(new_tl.spans.values(), key=lambda s: s.begin)

    self.assertEqual(spans[0].name, constants.IDLE)
    self.assertEqual(spans[0].begin, 0.0)
    self.assertEqual(spans[0].end, 2.0)

    self.assertEqual(spans[1].name, "s1")
    self.assertEqual(spans[1].end, float("inf"))

    self.assertEqual(spans[2].name, "s2")

  def test_add_idle_spans_do_not_ignore_active(self):
    tl = timeline.Timeline("test_tl", 0.0)
    span1 = timeline.Span("s1", 1.0, 0)
    span1.end = float("inf")
    tl.spans[0] = span1

    tl._last_span_id = 0

    new_tl = timeline_utils.add_idle_spans(tl, ignore_active_spans=False)

    # Gap from 0 to 1, then s1 spans to inf.
    self.assertLen(new_tl.spans, 2)
    spans = sorted(new_tl.spans.values(), key=lambda s: s.begin)

    self.assertEqual(spans[0].name, constants.IDLE)
    self.assertEqual(spans[0].begin, 0.0)
    self.assertEqual(spans[0].end, 1.0)
    self.assertEqual(spans[1].name, "s1")

  def test_merge_overlapping_spans(self):
    tl2 = timeline.Timeline("test_tl2", 0.0)

    span1 = timeline.Span("s1", 1.0, 10, parent_id=5, tags={"a": 1, "b": 2})
    span1.end = 3.0
    tl2.spans[10] = span1

    span2 = timeline.Span("s2", 2.0, 11, tags={"b": 3, "c": 4})
    span2.end = 4.0
    tl2.spans[11] = span2

    span3 = timeline.Span("s3", 5.0, 12, tags={"d": 5})
    span3.end = 6.0
    tl2.spans[12] = span3

    tl2._last_span_id = 12

    merged_tl = timeline_utils.merge_overlapping_spans(tl2)

    self.assertLen(merged_tl.spans, 2)
    self.assertIn(10, merged_tl.spans)
    self.assertIn(12, merged_tl.spans)

    merged_s1 = merged_tl.spans[10]
    self.assertEqual(merged_s1.name, "s1,s2")
    self.assertEqual(merged_s1.begin, 1.0)
    self.assertEqual(merged_s1.end, 4.0)
    self.assertEqual(merged_s1.id, 10)
    self.assertEqual(merged_s1.parent_id, 5)
    self.assertEqual(merged_s1.tags, {"a": 1, "b": 2, "c": 4})

    merged_s3 = merged_tl.spans[12]
    self.assertEqual(merged_s3.name, "s3")
    self.assertEqual(merged_s3.begin, 5.0)
    self.assertEqual(merged_s3.end, 6.0)
    self.assertEqual(merged_s3.id, 12)
    self.assertIsNone(merged_s3.parent_id)
    self.assertEqual(merged_s3.tags, {"d": 5})

  def test_flatten_overlapping_spans(self):
    tl = timeline.Timeline("test_tl", 0.0)

    # Span 1: 1.0 to 5.0
    span1 = timeline.Span("s1", 1.0, 10)
    span1.end = 5.0
    tl.spans[10] = span1

    # Span 2: 2.0 to 6.0
    span2 = timeline.Span("s2", 2.0, 11)
    span2.end = 6.0
    tl.spans[11] = span2

    # Span 3: 3.0 to 7.0
    span3 = timeline.Span("s3", 3.0, 12)
    span3.end = 7.0
    tl.spans[12] = span3

    tl._last_span_id = 12

    exec_tl, queue_tl = timeline_utils.flatten_overlapping_spans(tl)

    # Check exec_tl
    self.assertEqual(exec_tl.id, "test_tl_exec")
    self.assertLen(exec_tl.spans, 3)

    e1 = exec_tl.spans[10]
    self.assertEqual(e1.name, "s1")
    self.assertEqual(e1.begin, 1.0)
    self.assertEqual(e1.end, 5.0)

    e2 = exec_tl.spans[11]
    self.assertEqual(e2.name, "s2")
    self.assertEqual(e2.begin, 5.0)
    self.assertEqual(e2.end, 6.0)

    e3 = exec_tl.spans[12]
    self.assertEqual(e3.name, "s3")
    self.assertEqual(e3.begin, 6.0)
    self.assertEqual(e3.end, 7.0)

    # Check queue_tl
    self.assertEqual(queue_tl.id, "test_tl_queue")
    self.assertLen(queue_tl.spans, 2)

    queue_spans = sorted(queue_tl.spans.values(), key=lambda s: s.begin)

    q1 = queue_spans[0]
    self.assertEqual(q1.name, f"{constants.QUEUE}")
    self.assertEqual(q1.begin, 2.0)
    self.assertEqual(q1.end, 5.0)
    self.assertEqual(q1.tags[constants.QUEUED_SPAN], "s2")

    q2 = queue_spans[1]
    self.assertEqual(q2.name, f"{constants.QUEUE}")
    self.assertEqual(q2.begin, 5.0)
    self.assertEqual(q2.end, 6.0)
    self.assertEqual(q2.tags[constants.QUEUED_SPAN], "s3")


if __name__ == "__main__":
  absltest.main()

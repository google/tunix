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

from absl.testing import absltest
from tunix.perf.experimental import constants
from tunix.perf.experimental import span


class TimelineTest(absltest.TestCase):

  def test_span(self):
    s = span.Span(name="test", begin=1.0, id=0)
    self.assertEqual(s.name, "test")
    self.assertEqual(s.begin, 1.0)
    self.assertEqual(s.end, float("inf"))
    self.assertEqual(s.ended, False)
    self.assertEqual(s.duration, float("inf"))

  def test_span_with_tags(self):
    tags_dict = {constants.GLOBAL_STEP: 1, "custom_tag": "value"}
    s = span.Span(name="test_tags", begin=1.0, id=0, tags=tags_dict)
    self.assertEqual(s.tags, tags_dict)
    self.assertIn("tags=", repr(s))
    self.assertIn("global_step", repr(s))

  def test_add_tag(self):
    s = span.Span(name="test_add_tag", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    self.assertEqual(s.tags, {"foo": "bar"})
    s.add_tag(constants.GLOBAL_STEP, 100)
    self.assertEqual(s.tags, {"foo": "bar", "global_step": 100})

  def test_add_tag_overwrite_warning(self):
    s = span.Span(name="test_add_tag_overwrite", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    with self.assertLogs(level="WARNING") as cm:
      s.add_tag("foo", "baz")
    self.assertEqual(s.tags, {"foo": "baz"})
    self.assertTrue(
        any(
            "Tag 'foo' already exists with value 'bar'. Overwriting with 'baz'."
            in o
            for o in cm.output
        )
    )

  def test_repr_with_born_at(self):
    born_at = 100.0
    s = span.Span(name="test_born_at", begin=101.0, id=0)
    s.end = 105.0

    # Check default repr (born_at=0.0)
    expected_default = "[0] test_born_at: 101.000000, 105.000000"
    self.assertEqual(repr(s), expected_default)

    # Check repr with explicit born_at
    expected_adjusted = "[0] test_born_at: 1.000000, 5.000000"
    self.assertEqual(s.__repr__(born_at=born_at), expected_adjusted)


if __name__ == "__main__":
  absltest.main()

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

from typing import List, Optional
from absl.testing import absltest
from tunix.perf import span


def create_span(name: str, begin: float, end: float) -> span.Span:
  s = span.Span(name, begin)
  s.end = end
  return s


def create_group(
    name: str,
    begin: float,
    end: float,
    children: Optional[List[span.Span]] = None,
) -> span.SpanGroup:
  g = span.SpanGroup(name, None)
  g.begin = begin
  g.end = end
  if children:
    for child in children:
      g.inner.append(child)
      if isinstance(child, span.SpanGroup):
        child.outer = g
  return g


class SpanTest(absltest.TestCase):

  def test_clone_span(self):
    s = create_span("s1", 1.0, 2.0)
    clone = span.clone_span_or_group(s)
    self.assertIsNot(s, clone)
    self.assertEqual(s.name, clone.name)
    self.assertEqual(s.begin, clone.begin)
    self.assertEqual(s.end, clone.end)
    self.assertIsInstance(clone, span.Span)

  def test_clone_span_group(self):
    s1 = create_span("s1", 1.0, 2.0)
    g1 = create_group("g1", 0.0, 3.0, [s1])

    clone = span.clone_span_or_group(g1)

    with self.subTest("CloneGroupProperties"):
      self.assertIsNot(g1, clone)
      self.assertEqual(g1.name, clone.name)
      self.assertEqual(g1.begin, clone.begin)
      self.assertEqual(g1.end, clone.end)
      self.assertIsInstance(clone, span.SpanGroup)
      self.assertLen(clone.inner, 1)

    with self.subTest("CloneInnerSpan"):
      cloned_s1 = clone.inner[0]
      self.assertIsNot(s1, cloned_s1)
      self.assertEqual(s1.name, cloned_s1.name)
      self.assertEqual(s1.begin, cloned_s1.begin)
      self.assertEqual(s1.end, cloned_s1.end)
      self.assertIsInstance(cloned_s1, span.Span)

  def test_clone_span_group_parent_pointers(self):
    # Check that outer/parent pointers are correctly set after cloning.
    g2 = create_group("g2", 1.5, 2.5)
    g_parent = create_group("parent", 0.0, 10.0, [g2])
    clone_parent = span.clone_span_or_group(g_parent)
    self.assertLen(clone_parent.inner, 1)
    clone_g2 = clone_parent.inner[0]
    self.assertIsInstance(clone_g2, span.SpanGroup)
    self.assertIs(clone_g2.outer, clone_parent)

  def test_merge_identical_spans(self):
    s1 = create_span("s1", 1.0, 2.0)
    s2 = create_span("s1", 1.0, 2.0)

    merged = span.merge_span_group_trees(s1, s2)
    self.assertEqual(merged.name, "s1")
    self.assertEqual(merged.begin, 1.0)
    self.assertEqual(merged.end, 2.0)
    self.assertIsNot(merged, s1)
    self.assertIsNot(merged, s2)

  def test_merge_identical_span_groups(self):
    s1 = create_span("s1", 1.0, 2.0)
    g1 = create_group("g1", 0.0, 3.0, [s1])

    s2 = create_span("s1", 1.0, 2.0)
    g2 = create_group("g1", 0.0, 3.0, [s2])

    merged = span.merge_span_group_trees(g1, g2)

    self.assertEqual(merged.name, "g1")
    self.assertLen(merged.inner, 1)
    self.assertEqual(merged.inner[0].name, "s1")

  def test_merge_disjoint_children(self):
    s1 = create_span("s1", 1.0, 2.0)
    g1 = create_group("root", 0.0, 10.0, [s1])

    s2 = create_span("s2", 3.0, 4.0)
    g2 = create_group("root", 0.0, 10.0, [s2])

    merged = span.merge_span_group_trees(g1, g2)

    self.assertEqual(merged.name, "root")
    self.assertLen(merged.inner, 2)
    self.assertEqual(merged.inner[0].name, "s1")
    self.assertEqual(merged.inner[1].name, "s2")

  def test_merge_overlap_identical_children(self):
    # Tree 1: Root -> [A (1-2), B (3-4)]
    # Tree 2: Root -> [B (3-4), C (5-6)]
    # Result: Root -> [A, B, C]
    sA = create_span("A", 1.0, 2.0)
    sB1 = create_span("B", 3.0, 4.0)
    g1 = create_group("root", 0.0, 10.0, [sA, sB1])

    sB2 = create_span("B", 3.0, 4.0)
    sC = create_span("C", 5.0, 6.0)
    g2 = create_group("root", 0.0, 10.0, [sB2, sC])

    merged = span.merge_span_group_trees(g1, g2)

    self.assertLen(merged.inner, 3)
    self.assertEqual(merged.inner[0].name, "A")
    self.assertEqual(merged.inner[1].name, "B")
    self.assertEqual(merged.inner[2].name, "C")

  def test_merge_nested_recursion(self):
    # Tree 1: Root -> GroupA (1-5) -> [Span1 (2-3)]
    # Tree 2: Root -> GroupA (1-5) -> [Span2 (3-4)]
    # Result: Root -> GroupA (1-5) -> [Span1, Span2]
    span1 = create_span("s1", 2.0, 3.0)
    ga1 = create_group("GroupA", 1.0, 5.0, [span1])
    root1 = create_group("Root", 0.0, 10.0, [ga1])

    span2 = create_span("s2", 3.0, 4.0)
    ga2 = create_group("GroupA", 1.0, 5.0, [span2])
    root2 = create_group("Root", 0.0, 10.0, [ga2])

    merged = span.merge_span_group_trees(root1, root2)

    self.assertLen(merged.inner, 1)
    merged_ga = merged.inner[0]
    self.assertEqual(merged_ga.name, "GroupA")
    self.assertLen(merged_ga.inner, 2)
    self.assertEqual(merged_ga.inner[0].name, "s1")
    self.assertEqual(merged_ga.inner[1].name, "s2")

  def test_merge_conflict_overlap_spans(self):
    # Overlapping spans with different names or times
    s1 = create_span("s1", 1.0, 3.0)
    g1 = create_group("root", 0.0, 10.0, [s1])

    s2 = create_span("s2", 2.0, 4.0)  # Overlaps s1
    g2 = create_group("root", 0.0, 10.0, [s2])

    with self.assertRaisesRegex(ValueError, "Overlap detected"):
      span.merge_span_group_trees(g1, g2)

  def test_merge_conflict_same_name_diff_time(self):
    # Same name but different time -> treated as different, but if overlap -> error
    s1 = create_span("s1", 1.0, 3.0)
    g1 = create_group("root", 0.0, 10.0, [s1])

    s2 = create_span("s1", 2.0, 4.0)  # Overlaps s1, different time
    g2 = create_group("root", 0.0, 10.0, [s2])

    with self.assertRaisesRegex(ValueError, "Overlap detected"):
      span.merge_span_group_trees(g1, g2)

  def test_merge_root_mismatch(self):
    g1 = create_group("root1", 0.0, 10.0)
    g2 = create_group("root2", 0.0, 10.0)

    with self.assertRaisesRegex(ValueError, "Roots are not identical"):
      span.merge_span_group_trees(g1, g2)

  def test_merge_type_mismatch(self):
    # Span vs SpanGroup with same name and time
    s1 = create_span("node", 1.0, 2.0)
    g1 = create_group("node", 1.0, 2.0)

    with self.assertRaisesRegex(ValueError, "Roots are not identical"):
      span.merge_span_group_trees(s1, g1)

  def test_merge_empty_groups(self):
    g1 = create_group("root", 0.0, 10.0)
    g2 = create_group("root", 0.0, 10.0)
    merged = span.merge_span_group_trees(g1, g2)
    self.assertEmpty(merged.inner)

  def test_merge_multiple_overlaps(self):
    # [A(1-2), B(3-4), C(5-6)] merged with [A(1-2), C(5-6)]
    # Should result in [A, B, C]

    sA = create_span("A", 1.0, 2.0)
    sB = create_span("B", 3.0, 4.0)
    sC = create_span("C", 5.0, 6.0)

    g1 = create_group("root", 0.0, 10.0, [sA, sB, sC])
    g2 = create_group("root", 0.0, 10.0, [sA, sC])

    merged = span.merge_span_group_trees(g1, g2)
    self.assertLen(merged.inner, 3)

  def test_merge_identical_spans_with_jitter(self):
    s1 = create_span("s1", 1.0, 2.0)
    s2 = create_span("s1", 1.0 + 1e-10, 2.0 - 1e-10)

    self.assertTrue(span._are_nodes_shallowly_identical(s1, s2))
    merged = span.merge_span_group_trees(s1, s2)
    self.assertEqual(merged.name, "s1")
    self.assertEqual(merged.begin, 1.0)
    self.assertEqual(merged.end, 2.0)


if __name__ == "__main__":
  absltest.main()

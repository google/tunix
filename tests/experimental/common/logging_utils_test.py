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

"""Tests for logging_utils."""

from absl.testing import absltest
from tunix.experimental.common import logging_utils


class LoggingUtilsTest(absltest.TestCase):

  def test_summarize_list_empty(self):
    self.assertEqual(logging_utils.summarize_list([]), "[]")
    self.assertEqual(logging_utils.summarize_list(()), "[]")
    self.assertEqual(logging_utils.summarize_list([None]), "[]")
    self.assertEqual(logging_utils.summarize_list([""]), "[]")
    self.assertEqual(logging_utils.summarize_list([None, "", None]), "[]")

  def test_summarize_list_single(self):
    self.assertEqual(logging_utils.summarize_list(["prompt_3"]), "[prompt_3]")
    self.assertEqual(logging_utils.summarize_list([42]), "[42]")
    self.assertEqual(logging_utils.summarize_list([0]), "[0]")

  def test_summarize_list_default_max_length(self):
    self.assertEqual(
        logging_utils.summarize_list(["p0", "p1", "p2"]),
        "[p0, p1, p2]",
    )
    self.assertEqual(
        logging_utils.summarize_list(["p0", "p1", "p2", "p3"]),
        "[p0, p1, p2, p3]",
    )
    self.assertEqual(
        logging_utils.summarize_list(["p0", "p1", "p2", "p3", "p4"]),
        "[p0, p1, ..., p3, p4]",
    )
    self.assertEqual(
        logging_utils.summarize_list(["p0", "p1", "p2", "p3", "p4", "p5"]),
        "[p0, p1, ..., p4, p5]",
    )

  def test_summarize_list_even_max_length(self):
    ids_2 = ["p0", "p1"]
    ids_3 = ["p0", "p1", "p2"]
    self.assertEqual(
        logging_utils.summarize_list(ids_2, max_length=2), "[p0, p1]"
    )
    self.assertEqual(
        logging_utils.summarize_list(ids_3, max_length=2),
        "[p0, ..., p2]",
    )

    ids_6 = ["p0", "p1", "p2", "p3", "p4", "p5"]
    ids_7 = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]
    self.assertEqual(
        logging_utils.summarize_list(ids_6, max_length=6),
        "[p0, p1, p2, p3, p4, p5]",
    )
    self.assertEqual(
        logging_utils.summarize_list(ids_7, max_length=6),
        "[p0, p1, p2, ..., p4, p5, p6]",
    )

  def test_summarize_list_odd_max_length(self):
    ids_3 = ["p0", "p1", "p2"]
    ids_4 = ["p0", "p1", "p2", "p3"]
    ids_5 = ["p0", "p1", "p2", "p3", "p4"]
    self.assertEqual(
        logging_utils.summarize_list(ids_3, max_length=3),
        "[p0, p1, p2]",
    )
    self.assertEqual(
        logging_utils.summarize_list(ids_4, max_length=3),
        "[p0, p1, ..., p3]",
    )
    self.assertEqual(
        logging_utils.summarize_list(ids_5, max_length=3),
        "[p0, p1, ..., p4]",
    )

    ids_6 = ["p0", "p1", "p2", "p3", "p4", "p5"]
    self.assertEqual(
        logging_utils.summarize_list(ids_5, max_length=5),
        "[p0, p1, p2, p3, p4]",
    )
    self.assertEqual(
        logging_utils.summarize_list(ids_6, max_length=5),
        "[p0, p1, p2, ..., p4, p5]",
    )

  def test_summarize_list_filters_none_and_empty(self):
    ids = ["p0", None, "", "p1", "p2", None, "p3", ""]
    self.assertEqual(logging_utils.summarize_list(ids), "[p0, p1, p2, p3]")
    ids_with_extra = ["p0", None, "", "p1", "p2", "p3", "p4"]
    self.assertEqual(
        logging_utils.summarize_list(ids_with_extra),
        "[p0, p1, ..., p3, p4]",
    )

  def test_summarize_list_non_string_types(self):
    int_ids = [0, 1, 2, 3, 4, 5]
    self.assertEqual(
        logging_utils.summarize_list(int_ids),
        "[0, 1, ..., 4, 5]",
    )
    tuple_ids = ("a", "b", "c", "d", "e")
    self.assertEqual(
        logging_utils.summarize_list(tuple_ids),
        "[a, b, ..., d, e]",
    )

  def test_summarize_ids_alias(self):
    ids = ["p0", "p1", "p2", "p3", "p4", "p5"]
    self.assertEqual(
        logging_utils.summarize_ids(ids),
        "[p0, p1, ..., p4, p5]",
    )

  def test_invalid_max_length(self):
    with self.assertRaises(ValueError):
      logging_utils.summarize_list(["p0", "p1"], max_length=1)
    with self.assertRaises(ValueError):
      logging_utils.summarize_list(["p0", "p1"], max_length=0)
    with self.assertRaises(ValueError):
      logging_utils.summarize_list(["p0", "p1"], max_length=-5)


if __name__ == "__main__":
  absltest.main()

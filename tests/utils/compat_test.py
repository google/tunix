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

"""Tests for compatibility utilities."""

from typing import Any
from absl.testing import absltest
from tunix.utils import compat


class CompatTest(absltest.TestCase):

  def test_alias_init_param(self):
    class DummyClass:

      @compat.alias_init_param("old_arg", "new_arg")
      def __init__(self, new_arg: str = "", **kwargs: Any):
        self.new_arg = new_arg

    obj1 = DummyClass(new_arg="value")
    self.assertEqual(obj1.new_arg, "value")
    obj2 = DummyClass(old_arg="value")
    self.assertEqual(obj2.new_arg, "value")
    with self.assertRaisesRegex(ValueError, "Cannot specify both"):
      DummyClass(new_arg="value", old_arg="value")


if __name__ == "__main__":
  absltest.main()

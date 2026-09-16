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

"""Tests for import_utils."""

import os
from absl.testing import absltest
from tunix.experimental.common import import_utils


class ImportUtilsTest(absltest.TestCase):

  def test_import_symbol_success(self):
    symbol = import_utils.import_symbol("os.path.join")
    self.assertIs(symbol, os.path.join)

  def test_import_symbol_invalid_fqn_raises_value_error(self):
    with self.assertRaises(ValueError):
      import_utils.import_symbol("invalid_no_dot")

  def test_import_symbol_missing_module_raises_module_not_found_error(self):
    with self.assertRaises(ModuleNotFoundError):
      import_utils.import_symbol("non_existent_module_xyz.some_symbol")

  def test_import_symbol_missing_attribute_raises_attribute_error(self):
    with self.assertRaises(AttributeError):
      import_utils.import_symbol("os.non_existent_function_xyz")


if __name__ == "__main__":
  absltest.main()

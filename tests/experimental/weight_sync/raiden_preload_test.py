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

"""Tests for the Raiden native-module preload."""

from unittest import mock

from absl.testing import absltest
from tunix.experimental.weight_sync import raiden_preload


class ImportRaidenTest(absltest.TestCase):

  def test_loads_every_module_from_the_tpu_sync_package(self):
    importer = mock.Mock()
    with mock.patch.object(raiden_preload.importlib, "import_module", importer):
      loaded = raiden_preload.import_raiden()

    self.assertEqual(loaded, raiden_preload.RAIDEN_MODULES)
    self.assertEqual(
        [c.args[0] for c in importer.call_args_list],
        [f"tpu_sync.frameworks.jax.{n}" for n in raiden_preload.RAIDEN_MODULES],
    )

  def test_absent_wheel_is_not_an_error(self):
    """Every run_*_node.py preloads, including on hosts with no Raiden."""
    with mock.patch.object(
        raiden_preload.importlib, "import_module", side_effect=ImportError
    ):
      self.assertEqual(raiden_preload.import_raiden(), ())

  def test_targets_native_extensions_not_python_wrappers(self):
    for name in raiden_preload.RAIDEN_MODULES:
      self.assertStartsWith(name, "_")


if __name__ == "__main__":
  absltest.main()

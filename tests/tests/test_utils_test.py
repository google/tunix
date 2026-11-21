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

"""Tests for test_utils."""

import importlib.machinery
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from tests import test_utils


class TunixPackageDirTest(absltest.TestCase):

  def _make_package(self, pkg_dir: Path) -> Path:
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    return pkg_dir

  def _patch_test_utils_file(self, path: Path) -> None:
    self.enter_context(mock.patch.object(test_utils, "__file__", str(path)))

  def _patch_find_spec(self, spec) -> mock.MagicMock:
    return self.enter_context(
        mock.patch.object(
            test_utils.importlib.util, "find_spec", return_value=spec
        )
    )

  def test_oss_checkout_layout(self):
    repo = Path(self.create_tempdir().full_path)
    pkg = self._make_package(repo / "tunix")
    self._patch_test_utils_file(repo / "tests" / "test_utils.py")
    find_spec = self._patch_find_spec(None)
    self.assertEqual(test_utils.tunix_package_dir(), pkg.resolve())
    find_spec.assert_not_called()

  def test_in_package_layout(self):
    root = Path(self.create_tempdir().full_path)
    pkg = self._make_package(root / "tunix")
    self._patch_test_utils_file(pkg / "tests" / "test_utils.py")
    find_spec = self._patch_find_spec(None)
    self.assertEqual(test_utils.tunix_package_dir(), pkg.resolve())
    find_spec.assert_not_called()

  def test_falls_back_to_installed_package_without_checkout(self):
    repo = Path(self.create_tempdir("repo").full_path)
    self._patch_test_utils_file(repo / "tests" / "test_utils.py")
    site_packages = Path(self.create_tempdir("site_packages").full_path)
    installed = self._make_package(site_packages / "tunix")
    spec = importlib.machinery.ModuleSpec(
        "tunix", loader=None, origin=str(installed / "__init__.py")
    )
    self._patch_find_spec(spec)
    self.assertEqual(test_utils.tunix_package_dir(), installed.resolve())

  def test_raises_when_tunix_is_missing(self):
    repo = Path(self.create_tempdir().full_path)
    self._patch_test_utils_file(repo / "tests" / "test_utils.py")
    self._patch_find_spec(None)
    with self.assertRaisesRegex(ImportError, "neither checked out nor"):
      test_utils.tunix_package_dir()


class LoadModuleFromPathTest(absltest.TestCase):

  def test_loads_module_with_default_name(self):
    path = Path(self.create_tempdir().full_path) / "sample_mod.py"
    path.write_text("VALUE = 42\n")
    module = test_utils.load_module_from_path(path)
    self.assertEqual(module.__name__, "sample_mod")
    self.assertEqual(module.VALUE, 42)

  def test_loads_module_with_explicit_name(self):
    path = Path(self.create_tempdir().full_path) / "sample_mod.py"
    path.write_text("VALUE = 7\n")
    module = test_utils.load_module_from_path(path, "custom_name")
    self.assertEqual(module.__name__, "custom_name")
    self.assertEqual(module.VALUE, 7)

  def test_raises_for_non_python_file(self):
    path = Path(self.create_tempdir().full_path) / "data.txt"
    path.write_text("not python")
    with self.assertRaisesRegex(ImportError, "Cannot load module"):
      test_utils.load_module_from_path(path)


if __name__ == "__main__":
  absltest.main()

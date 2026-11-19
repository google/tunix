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

"""Unit tests for `tunix.utils.env_utils`."""

import ast
import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.utils import env_utils

_FORBIDDEN_ML_ROOTS = frozenset({
    "_tpu_raiden_jax",
    "flax",
    "jax",
    "jaxtyping",
    "optax",
    "pathwaysutils",
    "torch",
    "tpu_raiden",
    "tpu_sync",
})


class EnvUtilsTest(parameterized.TestCase):

  def test_is_internal_env_reflects_concrete_module_spec(self):
    if env_utils._INTERNAL_PACKAGE_ROOT.startswith("GOOGLE_INTERNAL_"):
      self.assertFalse(env_utils.is_internal_env())
    else:
      self.assertTrue(env_utils.is_internal_env())
    with mock.patch.object(env_utils, "_INTERNAL_PACKAGE_ROOT", "os"):
      self.assertTrue(env_utils.is_internal_env())
    with mock.patch.object(
        env_utils, "_INTERNAL_PACKAGE_ROOT", "_missing_internal_pkg_"
    ):
      self.assertFalse(env_utils.is_internal_env())
    namespace_spec = importlib.machinery.ModuleSpec("_ns_pkg", loader=None)
    with mock.patch.object(
        importlib.util, "find_spec", autospec=True, return_value=namespace_spec
    ):
      self.assertFalse(env_utils.is_internal_env())
    with mock.patch.object(
        importlib.util,
        "find_spec",
        autospec=True,
        side_effect=ValueError("__spec__ is None"),
    ):
      self.assertFalse(env_utils.is_internal_env())

  @parameterized.named_parameters(
      ("unset_false", {}, False, False),
      ("unset_true", {}, True, True),
      ("blank_uses_default_false", {"FLAG": "  "}, False, False),
      ("blank_uses_default_true", {"FLAG": "  "}, True, True),
      ("truthy_one", {"FLAG": "1"}, False, True),
      ("truthy_true_padded", {"FLAG": " TRUE "}, False, True),
      ("truthy_yes", {"FLAG": "YeS"}, False, True),
      ("truthy_on", {"FLAG": "ON"}, False, True),
      ("falsy_zero", {"FLAG": "0"}, True, False),
      ("falsy_false_padded", {"FLAG": " False "}, True, False),
      ("falsy_no", {"FLAG": "no"}, True, False),
      ("falsy_off", {"FLAG": "Off"}, True, False),
  )
  def test_parse_env_bool(
      self,
      env: dict[str, str],
      default: bool,
      expected: bool,
  ):
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(
          env_utils.parse_env_bool("FLAG", default=default),
          expected,
      )

  def test_parse_env_bool_rejects_invalid_tokens(self):
    with mock.patch.dict(os.environ, {"FLAG": "maybe"}, clear=True):
      with self.assertRaisesRegex(ValueError, "Unsupported FLAG='maybe'"):
        env_utils.parse_env_bool("FLAG", default=False)
      with self.assertRaisesRegex(ValueError, "Unsupported FLAG='maybe'"):
        env_utils.parse_env_bool("FLAG", default=True)

  def test_env_utils_ast_has_only_stdlib_imports(self):
    self.assertIsNotNone(env_utils.__file__)
    source = pathlib.Path(env_utils.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=env_utils.__file__)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          imported_roots.add(alias.name.split(".", 1)[0])
      elif isinstance(node, ast.ImportFrom) and node.module:
        imported_roots.add(node.module.split(".", 1)[0])
    self.assertLessEqual(imported_roots, sys.stdlib_module_names)
    self.assertEmpty(imported_roots & _FORBIDDEN_ML_ROOTS)

  def test_env_utils_is_ml_and_clif_free_at_import(self):
    self.assertIsNotNone(env_utils.__file__)
    expected_internal = env_utils.is_internal_env()
    if sys.executable:
      forbidden = sorted(_FORBIDDEN_ML_ROOTS)
      subprocess.run(
          [
              sys.executable,
              "-c",
              (
                  "import importlib.util, sys; "
                  f"sys.path = {sys.path!r}; "
                  "spec = importlib.util.spec_from_file_location("
                  f"'_isolated_env_utils', {env_utils.__file__!r}); "
                  "assert spec is not None and spec.loader is not None; "
                  "mod = importlib.util.module_from_spec(spec); "
                  "sys.modules[spec.name] = mod; "
                  "spec.loader.exec_module(mod); "
                  "assert mod.parse_env_bool("
                  "'UNSET_FLAG', default=True) is True; "
                  f"assert mod.is_internal_env() is {expected_internal!r}; "
                  f"loaded = [m for m in {forbidden!r} if m in sys.modules]; "
                  "assert not loaded, f'Unexpected ML imports: {loaded}'; "
                  "gfile_mods = [m for m in sys.modules "
                  "if m == 'gfile' or m.endswith('.gfile')]; "
                  "assert not gfile_mods, f'Unexpected gfile: {gfile_mods}'; "
                  "assert 'absl.flags' not in sys.modules"
              ),
          ],
          check=True,
          timeout=30,
      )
    else:
      self.assertTrue(env_utils.parse_env_bool("UNSET_FLAG", default=True))
      self.assertIsInstance(expected_internal, bool)
      self.assertEmpty(sys.modules.keys() & _FORBIDDEN_ML_ROOTS)
      self.assertEmpty(
          {m for m in sys.modules if m == "gfile" or m.endswith(".gfile")}
      )


if __name__ == "__main__":
  absltest.main()

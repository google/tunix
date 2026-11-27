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

"""Standard-library-only helpers for tests that load Tunix source files.

This module deliberately imports nothing beyond the Python standard library, so
tests using it run without JAX, absl, or pytest installed.
"""

import importlib.util
from pathlib import Path
import types


def tunix_package_dir() -> Path:
  """Returns the `tunix` package directory that tests should read files from.

  Prefers a source checkout: the OSS layout (`<repo>/tests/test_utils.py` next
  to `<repo>/tunix/`) or the in-package layout (`tunix/tests/test_utils.py`).
  Falls back to the importable `tunix` package when no checkout is present,
  e.g. when testing an installed wheel after `./tunix` was removed.

  Raises:
    ImportError: If `tunix` is neither checked out nor installed.
  """
  parent = Path(__file__).resolve().parents[1]
  for checkout_pkg in (parent / "tunix", parent):
    if checkout_pkg.name == "tunix" and (checkout_pkg / "__init__.py").is_file():
      return checkout_pkg
  spec = importlib.util.find_spec("tunix")
  if spec is None or spec.origin is None:
    raise ImportError("tunix package is neither checked out nor installed.")
  return Path(spec.origin).resolve().parent


def load_module_from_path(
    path: Path, module_name: str | None = None
) -> types.ModuleType:
  """Loads a Python source file as a module without importing its package.

  Args:
    path: Path to the `.py` file.
    module_name: Module name to use; defaults to the file stem.

  Returns:
    The executed module.
  """
  spec = importlib.util.spec_from_file_location(module_name or path.stem, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

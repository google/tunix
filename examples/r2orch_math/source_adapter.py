# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit, revision-pinned access to the official R²Orch implementation.

Source code is loaded from committed Git objects, never silently from a dirty
working tree. No source is vendored or redistributed. Callers opt into executing
the original package and must install its required dependencies separately.
"""

import functools
import importlib
import importlib.util
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

from examples.r2orch_math.prepare_data import SOURCE_COMMIT


@functools.lru_cache(maxsize=None)
def _source_package(source: str):
  source = str(Path(source).resolve())
  revision = subprocess.check_output(
      ["git", "-C", source, "rev-parse", "HEAD"], text=True
  ).strip()
  if revision != SOURCE_COMMIT:
    raise ValueError(
        f"Expected R²Orch source revision {SOURCE_COMMIT}, got {revision}."
    )
  raw = subprocess.check_output([
      "git",
      "-C",
      source,
      "archive",
      "--format=tar",
      SOURCE_COMMIT,
      "src/r2orch",
      "configs",
  ])
  directory = tempfile.TemporaryDirectory(prefix="tunix-r2orch-source-")
  root = Path(directory.name).resolve()
  with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
    # Reject links/devices and escaping names before extracting trusted pinned
    # source. This also makes the boundary safe against malformed Git archives.
    for member in archive.getmembers():
      target = (root / member.name).resolve()
      if not target.is_relative_to(root) or not (
          member.isfile() or member.isdir()
      ):
        directory.cleanup()
        raise ValueError("Source archive contains an unsafe member.")
    archive.extractall(root, filter="data")
  name = f"_tunix_r2orch_{SOURCE_COMMIT}"
  if name not in sys.modules:
    package_path = root / "src/r2orch"
    spec = importlib.util.spec_from_file_location(
        name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    try:
      spec.loader.exec_module(package)
    except BaseException:
      sys.modules.pop(name, None)
      directory.cleanup()
      raise
  # Keep the TemporaryDirectory alive as long as its imported modules can be
  # used, including deferred relative imports after this function returns.
  return name, directory


def load_source_module(source, module: str):
  """Import an official module from the exact pinned source revision.

  Example: load_source_module(checkout, "curriculum").ADSBoundaryScheduler.
  The returned classes retain original behavior; this function does not patch
  tools, replace checkpoints or weaken the original admission checks.
  """
  if not re.fullmatch(
      r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*", module
  ):
    raise ValueError("Expected a relative Python module name.")
  name, _ = _source_package(str(Path(source).resolve()))
  return importlib.import_module(f"{name}.{module}")

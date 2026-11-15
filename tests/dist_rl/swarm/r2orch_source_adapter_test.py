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

"""Pinned source loading ignores local edits and preserves relative imports."""

from pathlib import Path
import subprocess

import pytest
from examples.r2orch_math import source_adapter


def git(root, *args):
  return subprocess.check_output(
      ["git", "-C", str(root), *args], text=True
  ).strip()


def source_tree(tmp_path, monkeypatch):
  root = tmp_path / "reference"
  (root / "src/r2orch").mkdir(parents=True)
  (root / "configs").mkdir()
  (root / "src/r2orch/__init__.py").write_text(
      '"""Reference test package."""\n'
  )
  (root / "src/r2orch/state.py").write_text("VALUE = 'committed'\n")
  (root / "src/r2orch/consumer.py").write_text("from .state import VALUE\n")
  (root / "configs/test.toml").write_text("pinned = true\n")
  git(root, "init", "-q")
  git(root, "add", ".")
  git(
      root,
      "-c",
      "user.name=Fixture",
      "-c",
      "user.email=fixture@example.invalid",
      "commit",
      "-qm",
      "Pin source fixture",
  )
  monkeypatch.setattr(
      source_adapter, "SOURCE_COMMIT", git(root, "rev-parse", "HEAD")
  )
  return root


def test_loader_uses_committed_bytes_and_resolves_relative_imports(
    tmp_path, monkeypatch
):
  source = source_tree(tmp_path, monkeypatch)
  (source / "src/r2orch/state.py").write_text(
      "raise RuntimeError('dirty working tree was executed')\n"
  )
  loaded = source_adapter.load_source_module(source, "consumer")
  assert loaded.VALUE == "committed"
  assert not Path(loaded.__file__).is_relative_to(source)
  assert loaded.__name__.startswith("_tunix_r2orch_")


def test_loader_rejects_unpinned_revision_and_path_names(tmp_path, monkeypatch):
  source = source_tree(tmp_path, monkeypatch)
  monkeypatch.setattr(source_adapter, "SOURCE_COMMIT", "0" * 40)
  with pytest.raises(ValueError, match="revision"):
    source_adapter.load_source_module(source, "state")
  for module in ("../state", "state/other", "", ".state"):
    with pytest.raises(ValueError, match="module name"):
      source_adapter.load_source_module(source, module)

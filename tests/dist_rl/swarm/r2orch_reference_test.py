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

"""Optional executable checks against the pinned original scorer and datasets."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from examples.r2orch_math import grading
from examples.r2orch_math import prepare_data


@pytest.fixture
def source():
  value = os.environ.get("TUNIX_R2ORCH_SOURCE")
  if not value:
    pytest.skip("Set TUNIX_R2ORCH_SOURCE to the pinned original checkout.")
  return Path(value)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", True),
        ("002", True),
        ("2.0", True),
        (r"\boxed{2}", True),
        ("The final answer is 2.", True),
        ('{"answer":"2"}', True),
        ('{"final_answer":"2"}', True),
        (r"\boxed{2} or \boxed{3}", False),
        (r"\boxed{3}", False),
        ("import os; print(2)", False),
        ("", False),
    ],
)
def test_original_grader_contract(source, text, expected):
  assert grading.load_original_aime_grader(source)(text, "2") is expected


def test_original_split_counts_hashes_and_immutable_source(source, tmp_path):
  manifest = prepare_data.prepare(source, tmp_path / "data")
  assert {k: v["count"] for k, v in manifest["splits"].items()} == {
      "train": 947,
      "dev": 30,
      "test": 30,
      "ood": 128,
  }
  assert manifest["commit"] == prepare_data.SOURCE_COMMIT
  for name, split in manifest["splits"].items():
    path = tmp_path / "data" / f"{name}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == split["sha256"]
    original = subprocess.check_output([
        "git",
        "-C",
        str(source),
        "show",
        f"{prepare_data.SOURCE_COMMIT}:{split['source_path']}",
    ])
    assert hashlib.sha256(original).hexdigest() == split["source_sha256"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row["problem"] == row["source"]["prompt"] for row in records)
  with pytest.raises(ValueError, match="empty"):
    prepare_data.prepare(source, tmp_path / "data")


def test_wrong_source_revision_is_rejected_before_reading_data(
    monkeypatch, tmp_path
):
  monkeypatch.setattr(
      subprocess, "check_output", lambda *a, **k: "not-the-pinned-revision\n"
  )
  with pytest.raises(ValueError, match="revision"):
    prepare_data.prepare(tmp_path / "source", tmp_path / "data")
  grading.load_original_aime_grader.cache_clear()
  with pytest.raises(ValueError, match="revision"):
    grading.load_original_aime_grader(tmp_path / "source")

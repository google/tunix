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

"""Public task DSL, grading isolation and atomic round-state contracts."""

import json

import pytest
from examples.swarm_repair import environment
from tunix.dist_rl.swarm import checkpoint


def test_splits_are_deterministic_family_disjoint_and_no_hidden_tests_in_prompt():
  splits = {
      name: environment.tasks(name, count=4)
      for name in ("train", "dev", "test")
  }
  assert splits["train"] == environment.tasks("train", count=4)
  families = [{t.family for t in tasks} for tasks in splits.values()]
  assert not (
      families[0] & families[1]
      or families[0] & families[2]
      or families[1] & families[2]
  )
  for tasks in splits.values():
    for task in tasks:
      public = task.public()
      assert "hidden_inputs" not in json.dumps(public)
      assert all(isinstance(v, str) for v in public["requirements"].values())
      assert not set(task.hidden_inputs) & {
          e["input"] for e in public["public_examples"]
      }
      assert task.grade(json.dumps(task.oracle_modules())) == 1
      wrong = {p: {"op": "identity", "value": 0} for p in environment.FILES}
      assert task.grade(json.dumps(wrong)) == 0


@pytest.mark.parametrize(
    "module",
    [
        {"op": "__import__('os').system('echo unsafe')", "value": 0},
        {"op": "scale", "value": True},
        {"op": "shift", "value": 101},
        {"op": "scale", "value": 2, "code": "print(1)"},
    ],
)
def test_dsl_rejects_arbitrary_execution_and_invalid_parameters(module):
  with pytest.raises(ValueError):
    environment.parse_module(module)


def test_manifest_integrity_and_failed_replace_preserve_previous_state(
    tmp_path, monkeypatch
):
  path = tmp_path / "state.json"
  checkpoint.write_manifest(path, {"round": 1})
  assert checkpoint.read_manifest(path) == {"round": 1}
  original_replace = checkpoint.os.replace

  def fail(*args):
    raise OSError("simulated commit failure")

  monkeypatch.setattr(checkpoint.os, "replace", fail)
  with pytest.raises(OSError):
    checkpoint.write_manifest(path, {"round": 2})
  assert checkpoint.read_manifest(path) == {"round": 1}
  monkeypatch.setattr(checkpoint.os, "replace", original_replace)
  raw = json.loads(path.read_text())
  raw["state"]["round"] = 9
  path.write_text(json.dumps(raw))
  with pytest.raises(ValueError, match="integrity"):
    checkpoint.read_manifest(path)

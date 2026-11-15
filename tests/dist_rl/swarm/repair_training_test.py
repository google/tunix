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

"""Native CISPO updates, exact restart and durable zero-update cursors."""

import json
import numpy as np
import pytest
from examples.swarm_repair import environment
from examples.swarm_repair import train
from tunix.dist_rl.swarm import checkpoint


def config():
  return json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())


def test_two_updates_resume_and_all_baseline_metrics(tmp_path):
  cfg = config()
  full = train.run("cpu", cfg, tmp_path / "full")
  train.run("cpu", {**cfg, "num_steps": 1}, tmp_path / "resumed")
  resumed = train.run("cpu", cfg, tmp_path / "resumed", resume=True)
  assert full["optimizer_step"] == resumed["optimizer_step"] == 2
  assert full["parameter_delta_l2"] > 0
  assert full["history"] == resumed["history"]
  np.testing.assert_array_equal(
      np.load(tmp_path / "full/parameters.npz")["final"],
      np.load(tmp_path / "resumed/parameters.npz")["final"],
  )
  assert set(full["initial_untrained_eval"]) == {
      "single",
      "independent",
      "fixed",
      "swarm",
  }
  assert full["final_eval"]["fixed"]["mean_peak_concurrent_calls"] == 2
  tested = train.run(
      "cpu",
      cfg,
      tmp_path / "resumed",
      resume=True,
      mode="eval",
      eval_split="test",
  )
  assert tested["parameter_delta_l2"] == 0 and tested["optimizer_step"] == 2
  assert tested["eval_manifest"]["families"] == ["absolute_scale", "quadratic"]


def test_zero_signal_round_cursor_advances_without_optimizer(
    tmp_path, monkeypatch
):
  monkeypatch.setattr(environment.RepairTask, "grade", lambda self, text: 0.0)
  cfg = config()
  first = train.run("cpu", {**cfg, "num_steps": 1}, tmp_path / "zero")
  resumed = train.run("cpu", cfg, tmp_path / "zero", resume=True)
  assert first["optimizer_step"] == resumed["optimizer_step"] == 0
  assert len(resumed["history"]) == 2
  assert resumed["parameter_delta_l2"] == 0
  state = checkpoint.read_manifest(tmp_path / "zero/round_state.json")
  assert state["repair_round"] == 2 and state["optimizer_step"] == 0


def test_durable_boundary_rolls_forward_after_manifest_write_failure(
    tmp_path, monkeypatch
):
  cfg = config()
  expected = train.run("cpu", cfg, tmp_path / "expected")
  original = checkpoint.write_manifest

  def fail_after_durable_boundary(path, state):
    if state["repair_round"] == 1:
      raise OSError("simulated pointer interruption")
    return original(path, state)

  monkeypatch.setattr(checkpoint, "write_manifest", fail_after_durable_boundary)
  with pytest.raises(OSError, match="interruption"):
    train.run("cpu", cfg, tmp_path / "crashed")
  monkeypatch.setattr(checkpoint, "write_manifest", original)
  actual = train.run("cpu", cfg, tmp_path / "crashed", resume=True)
  assert actual["history"] == expected["history"]
  np.testing.assert_array_equal(
      np.load(tmp_path / "expected/parameters.npz")["final"],
      np.load(tmp_path / "crashed/parameters.npz")["final"],
  )

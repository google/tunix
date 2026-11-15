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

"""Native optimizer updates, immutable worker and deterministic checkpoint replay."""

import dataclasses
import json

import numpy as np
import pytest
from examples.r2orch_math import train
from tunix.dist_rl.swarm import checkpoint


def inputs():
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  return (
      config,
      train.load_tasks(train.ROOT / "data/cpu_train.jsonl"),
      train.load_tasks(train.ROOT / "data/cpu_eval.jsonl"),
  )


def test_native_updates_freeze_workers_and_resume_matches_uninterrupted(
    tmp_path,
):
  config, tasks, evaluation = inputs()
  full = train.run("cpu", config, tasks, evaluation, tmp_path / "full")
  first = train.run(
      "cpu", {**config, "num_steps": 1}, tasks, evaluation, tmp_path / "resumed"
  )
  resumed = train.run(
      "cpu", config, tasks, evaluation, tmp_path / "resumed", resume=True
  )
  assert first["optimizer_step"] == 2
  assert full["optimizer_step"] == resumed["optimizer_step"] == 4
  assert full["director_parameter_delta_l2"] > 0
  assert (
      full["frozen_parameters_unchanged"]
      and resumed["frozen_parameters_unchanged"]
  )
  a, b = (
      np.load(tmp_path / name / "parameters.npz")
      for name in ("full", "resumed")
  )
  np.testing.assert_array_equal(a["director"], b["director"])
  np.testing.assert_array_equal(a["frozen"], b["frozen"])
  assert full["history"] == resumed["history"]
  assert all(h["optimizer_steps"] == 2 for h in full["history"])
  assert all(set(h["role_advantages"]) == {"director"} for h in full["history"])
  traces = [
      json.loads(row)
      for row in (tmp_path / "full/train_traces.jsonl").read_text().splitlines()
  ]
  assert {
      call["policy_id"] for row in traces for call in row["episode"]["calls"]
  } == {"director", "worker"}
  with pytest.raises(ValueError, match="identity"):
    train.run(
        "cpu",
        {**config, "learning_rate": 0.2},
        tasks,
        evaluation,
        tmp_path / "resumed",
        resume=True,
    )
  changed_eval = [
      dataclasses.replace(evaluation[0], problem="Compute 8 minus 6.")
  ]
  with pytest.raises(ValueError, match="identity"):
    train.run(
        "cpu", config, tasks, changed_eval, tmp_path / "resumed", resume=True
    )
  test_report = train.run(
      "cpu",
      config,
      tasks,
      changed_eval,
      tmp_path / "resumed",
      resume=True,
      mode="eval",
  )
  assert test_report["optimizer_step"] == 4
  assert test_report["director_parameter_delta_l2"] == 0
  assert test_report["eval_data_sha256"] != full["eval_data_sha256"]


def test_bad_sampling_and_data_contracts_fail_before_loading_models(tmp_path):
  config, tasks, evaluation = inputs()
  with pytest.raises(ValueError, match="temperature"):
    train.run(
        "cpu",
        {**config, "train_temperature": 0.7},
        tasks,
        evaluation,
        tmp_path / "bad",
    )
  with pytest.raises(ValueError, match="disjoint"):
    train.run("cpu", config, tasks, tasks, tmp_path / "bad")


@pytest.mark.parametrize(
    "key",
    [
        "worker_temperature",
        "learning_rate",
        "eval_temperature",
        "kl_beta",
        "cost_lambda",
        "max_logprob_error",
    ],
)
def test_nonfinite_hyperparameters_rejected(key):
  config, _, _ = inputs()
  with pytest.raises(ValueError, match="finite"):
    train.validate_config({**config, key: float("nan")})


def test_zero_update_rounds_preserve_collection_cursor(tmp_path):
  config, tasks, evaluation = inputs()
  config = {**config, "kl_beta": 0.0, "cost_lambda": 0.0}
  tasks = [dataclasses.replace(t, answer="999") for t in tasks]
  first = train.run(
      "cpu", {**config, "num_steps": 1}, tasks, evaluation, tmp_path / "run"
  )
  resumed = train.run(
      "cpu", config, tasks, evaluation, tmp_path / "run", resume=True
  )
  assert first["optimizer_step"] == resumed["optimizer_step"] == 0
  assert resumed["director_parameter_delta_l2"] == 0
  assert [h["step"] for h in resumed["history"]] == [1, 2]
  state = checkpoint.read_manifest(
      tmp_path / "run/checkpoints/round_state.json"
  )
  assert state["r2orch_step"] == 2 and state["optimizer_step"] == 0


def test_durable_snapshot_recovers_manifest_publication_failure(
    tmp_path, monkeypatch
):
  config, tasks, evaluation = inputs()
  original = checkpoint.write_manifest

  def fail_after_snapshot(path, state):
    if state["r2orch_step"] == 1:
      raise OSError("injected manifest publication failure")
    return original(path, state)

  monkeypatch.setattr(checkpoint, "write_manifest", fail_after_snapshot)
  with pytest.raises(OSError, match="injected"):
    train.run("cpu", config, tasks, evaluation, tmp_path / "failed")
  monkeypatch.setattr(checkpoint, "write_manifest", original)
  resumed = train.run(
      "cpu", config, tasks, evaluation, tmp_path / "failed", resume=True
  )
  full = train.run("cpu", config, tasks, evaluation, tmp_path / "full")
  assert resumed["history"] == full["history"]
  np.testing.assert_array_equal(
      np.load(tmp_path / "failed/parameters.npz")["director"],
      np.load(tmp_path / "full/parameters.npz")["director"],
  )

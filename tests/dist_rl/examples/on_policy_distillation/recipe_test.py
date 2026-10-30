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

"""Full recipe provenance, configuration and launch lifecycle checks."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tunix.dist_rl.examples.on_policy_distillation import models
from tunix.dist_rl.examples.on_policy_distillation import recipe
from tunix.dist_rl.examples.on_policy_distillation import serve_teacher


def _prepared():
  return {
      "schema_version": 1,
      "tokenizer_fingerprint": "f" * 64,
      "student": {
          "repo_id": models.STUDENT_MODEL_ID,
          "revision": models.STUDENT_REVISION,
          "snapshot_path": "/prepared student/snapshot",
          "tokenizer_path": "/prepared student/snapshot",
          "declared_revision": (
              f"hf:{models.STUDENT_MODEL_ID}@{models.STUDENT_REVISION}"
          ),
      },
      "teacher": {
          "repo_id": models.TEACHER_MODEL_ID,
          "revision": models.TEACHER_REVISION,
          "snapshot_path": "/prepared teacher/snapshot",
          "tokenizer_path": "/prepared teacher/snapshot",
          "declared_revision": (
              f"hf:{models.TEACHER_MODEL_ID}@{models.TEACHER_REVISION}"
          ),
      },
  }


class RecipeTest(unittest.TestCase):

  def setUp(self):
    self.config = recipe.load_config()
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.run_dir = Path(self.directory.name) / "run with spaces"
    self.manifest = Path(self.directory.name) / "models.json"

  def _args(
      self,
      *,
      command="train",
      resume=False,
      dry_run=False,
      teacher_url="http://teacher:30001",
  ):
    return SimpleNamespace(
        command=command,
        resume=resume,
        dry_run=dry_run,
        models_manifest=str(self.manifest),
        run_dir=str(self.run_dir),
        data_dir=None,
        teacher_url=teacher_url,
    )

  def _launch(self, args=None, config=None):
    with mock.patch.object(
        models, "load_manifest", return_value=_prepared()
    ), mock.patch.object(
        serve_teacher,
        "verify_teacher_service",
        new_callable=mock.AsyncMock,
        return_value={"served_model_name": "fixed-teacher"},
    ) as verify, mock.patch.object(
        recipe.subprocess, "run", return_value=SimpleNamespace(returncode=0)
    ) as process:
      result = recipe._launch(args or self._args(), config or self.config)
    return result, verify, process

  def test_profile_pins_models_and_disjoint_official_splits(self):
    self.assertEqual(
        self.config["student"]["revision"], models.STUDENT_REVISION
    )
    self.assertEqual(
        self.config["teacher"]["revision"], models.TEACHER_REVISION
    )
    self.assertEqual(self.config["training"]["tfds_split"], "train")
    self.assertEqual(self.config["evaluation"]["split"], "test")
    self.assertEqual(self.config["evaluation"]["max_examples"], 0)
    self.assertEqual(
        self.config["training"]["max_steps"]
        * self.config["training"]["batch_size"],
        4800,
    )
    self.assertLessEqual(4800, 7473)

  def test_teacher_command_consumes_profile_without_launching_in_dry_run(self):
    with mock.patch(
        "sys.stdout", new_callable=io.StringIO
    ) as output, mock.patch.object(serve_teacher, "main") as launch:
      result = recipe.main([
          "teacher",
          "--models-manifest",
          str(self.manifest),
          "--cuda-visible-devices",
          "0,1",
          "--dry-run",
      ])
    self.assertEqual(result, 0)
    command = json.loads(output.getvalue())
    for field, value in self.config["teacher_service"].items():
      index = command.index("--" + field.replace("_", "-"))
      self.assertEqual(command[index + 1], str(value))
    self.assertEqual(command[-2:], ["--cuda-visible-devices", "0,1"])
    launch.assert_not_called()

  def test_invalid_profile_changes_fail_before_launch(self):
    invalid = (
        ("student", "revision", "main"),
        ("teacher", "repo_id", "other/teacher"),
        ("training", "tfds_split", "test"),
        ("training", "tfds_dataset", "gsm8k"),
        ("training", "top_p", 0.9),
        ("training", "temperature", 0.8),
        ("training", "num_generations", 1),
        ("training", "mini_batch_size", 8),
        ("training", "max_steps", 1000),
        ("training", "checkpoint_save_interval_steps", 7),
        ("training", "train_micro_batch_size", 64),
        ("training", "opd_coefficient", float("nan")),
        ("training", "beta", 0.1),
        ("evaluation", "split", "train[:20]"),
        ("evaluation", "max_examples", 10),
        ("evaluation", "max_response_length", 8192),
        ("hardware", "rollout_tpu_chips", "0,1,2,3"),
        ("hardware", "trainer_tpu_chips", "00,1,2,3"),
        ("hardware", "tpu_chips_per_host_bounds", "1,1,1"),
        ("hardware", "tpu_chips_per_host_bounds", "2,2,0"),
        ("hardware", "tpu_host_bounds", "1,2,1"),
        ("teacher_service", "context_length", 1024),
    )
    path = Path(self.directory.name) / "profile.json"
    for section, key, value in invalid:
      with self.subTest(section=section, key=key, value=value):
        config = copy.deepcopy(self.config)
        config[section][key] = value
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
          recipe.load_config(path)

  def test_explicit_environment_preserves_paths_and_overrides_smoke_defaults(
      self,
  ):
    result, verify, process = self._launch()
    self.assertEqual(result, 0)
    verify.assert_awaited_once()
    self.assertEqual(verify.await_args.kwargs["min_context_length"], 5120)
    environment = process.call_args.kwargs["env"]
    for key, expected in {
        "MODEL_ID": models.STUDENT_MODEL_ID,
        "MODEL_REVISION": models.STUDENT_REVISION,
        "MODEL_DIR": "/prepared student/snapshot",
        "TOKENIZER_PATH": "/prepared student/snapshot",
        "OPD_TEACHER_REVISION": _prepared()["teacher"]["declared_revision"],
        "MAX_STEPS": "300",
        "BATCH_SIZE": "16",
        "NUM_GENERATIONS": "4",
        "ALLOW_MODEL_DOWNLOAD": "false",
        "SCHEDULE_TYPE": "",
        "OPD_TASK_REWARD_COEFFICIENT": "0.0",
        "BETA": "0.0",
        "MODEL_DTYPE": "bfloat16",
        "REMAT_CONFIG": "DECODER",
        "TFDS_DATASET": "gsm8k:1.0.0",
        "GENERATION_SEED": "42",
        "MAX_STALENESS": "0",
        "OPD_REQUIRE_RESUME": "false",
        "MAX_SEQ_TOKEN_PER_TPU": "",
    }.items():
      self.assertEqual(environment[key], expected, key)
    self.assertEqual(
        json.loads((self.run_dir / "evaluation.json").read_text()),
        self.config["evaluation"],
    )
    persisted = json.loads((self.run_dir / "run_manifest.json").read_text())
    self.assertEqual(persisted["profile"], self.config)
    self.assertEqual(
        persisted["config_sha256"], recipe.config_digest(self.config)
    )
    self.assertEqual(persisted["models"], _prepared())
    self.assertNotIn("WANDB_API_KEY", persisted["environment"])

  def test_dry_run_has_no_network_or_filesystem_launch_effects(self):
    with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
      result, verify, process = self._launch(self._args(dry_run=True))
    self.assertEqual(result, 0)
    self.assertEqual(json.loads(output.getvalue())["models"], _prepared())
    verify.assert_not_awaited()
    process.assert_not_called()
    self.assertFalse(self.run_dir.exists())

  def test_run_directory_cannot_be_reused_silently(self):
    self._launch()
    with self.assertRaisesRegex(ValueError, "empty directory"):
      self._launch()
    with self.assertRaisesRegex(ValueError, "numeric checkpoint"):
      self._launch(self._args(resume=True))

  def test_resume_keeps_frozen_experiment_and_requires_a_checkpoint(self):
    self._launch()
    (self.run_dir / "checkpoints" / "25").mkdir(parents=True)
    original = (self.run_dir / "run_manifest.json").read_text()
    _, verify, process = self._launch(self._args(resume=True))
    verify.assert_awaited_once()
    self.assertEqual(
        process.call_args.kwargs["env"]["OPD_REQUIRE_RESUME"], "true"
    )
    self.assertEqual((self.run_dir / "run_manifest.json").read_text(), original)
    changed = copy.deepcopy(self.config)
    changed["training"]["learning_rate"] *= 2
    with self.assertRaisesRegex(
        ValueError, "configuration/model manifest changed"
    ):
      self._launch(self._args(resume=True), changed)

  def test_checkpoint_evaluation_uses_recorded_identity_without_teacher_network(
      self,
  ):
    self._launch()
    (self.run_dir / "checkpoints" / "300").mkdir(parents=True)
    before = (self.run_dir / "teacher_preflight.json").read_text()
    _, verify, process = self._launch(
        self._args(command="eval", teacher_url=None)
    )
    verify.assert_not_awaited()
    self.assertEqual(process.call_args.kwargs["env"]["OPD_EVAL_ONLY"], "true")
    self.assertEqual(
        process.call_args.kwargs["env"]["OPD_REQUIRE_RESUME"], "true"
    )
    self.assertEqual(
        process.call_args.kwargs["env"]["OPD_TEACHER_URL"],
        "http://teacher:30001",
    )
    self.assertEqual(
        (self.run_dir / "teacher_preflight.json").read_text(), before
    )

  def test_bad_teacher_url_is_not_logged_or_launched(self):
    for address in (
        "http://user:secret@teacher",
        "http://teacher?token=secret",
        "file:///teacher",
    ):
      with self.subTest(address=address):
        with self.assertRaisesRegex(ValueError, "base URL"):
          self._launch(self._args(teacher_url=address))
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
  unittest.main()

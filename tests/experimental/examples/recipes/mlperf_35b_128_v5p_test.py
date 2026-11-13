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

"""Dry-run tests for `experimental/examples/recipes/mlperf_35b_128_v5p.sh`."""

from __future__ import annotations

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
import yaml

_RECIPE = (
    pathlib.Path(resolve_config.__file__).parents[3]
    / "experimental/examples/recipes/mlperf_35b_128_v5p.sh"
)


class Mlperf35bRecipeTest(launcher_testing.LauncherTestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {"MLPERF_SKIP_INTERNAL_ARGS": "1"})
    )

  def test_oss_cli_flags_resolve_and_render_18_jobsets(self):
    run = self.dry_run(
        str(_RECIPE),
        "--image=test-image",
        "--bucket=gs://test-bucket",
        "--maxtext-ckpt=gs://test-bucket/ckpt",
        "--dataset-path=gs://test-bucket/data",
    )
    hp = run.resolve()
    cfg = hp.config

    self.assertEqual(cfg["model_config"]["model_name"], "Qwen3.5-35B-A3B")
    self.assertEqual(cfg["model_config"]["model_source"], "maxtext")
    self.assertEqual(cfg["model_config"]["remat_config"], "none")
    self.assertFalse(cfg["model_config"]["use_flash_attention"])
    self.assertEqual(
        cfg["rl_training_config"]["compute_logps_micro_batch_size"], 1
    )
    self.assertEqual(
        cfg["actor_model_config"]["model_path"], "gs://test-bucket/ckpt"
    )
    self.assertEqual(
        cfg["data_config"]["dataset_path"], "gs://test-bucket/data"
    )
    self.assertEqual(
        cfg["rl_training_config"]["checkpoint_root_directory"],
        "gs://test-bucket/checkpoints/testuser",
    )
    self.assertEqual(
        cfg["prompt_items_wrapper"],
        "tunix.experimental.examples.deepswe_dist.prewarm.prewarm_sandboxes",
    )
    self.assertIn("ORCHESTRATOR_ID=testuser-orch", run.stdout)

    jobsets = []
    for section in run.stdout.split("\n---\n")[1:]:
      for doc in yaml.safe_load_all(section):
        if doc and doc.get("kind") == "JobSet":
          jobsets.append(doc["metadata"]["name"])
    self.assertLen(jobsets, 18)
    self.assertContainsSubset(
        {"testuser-orch", "testuser-train", "testuser-roll-15"}, jobsets
    )

  def test_cli_validation_and_stop(self):
    run = self.run_launcher(str(_RECIPE))
    self.assertNotEqual(run.returncode, 0)
    for flag in ("--image", "--bucket", "--maxtext-ckpt", "--dataset-path"):
      self.assertIn(flag, run.stderr)
    self.assertNotIn("internal_args", run.stderr)

    stop_run = self.dry_run(str(_RECIPE), "stop")
    self.assertIn("kubectl delete jobset testuser-orch", stop_run.stdout)


if __name__ == "__main__":
  absltest.main()

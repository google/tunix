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

import pathlib

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
import yaml

_RECIPE = (
    pathlib.Path(resolve_config.__file__).parents[3]
    / "experimental/examples/recipes/mlperf_35b_128_v5p.sh"
)


class Mlperf35bRecipeTest(launcher_testing.LauncherTestCase):

  def test_resolves_and_renders_18_jobsets(self):
    run = self.dry_run(
        str(_RECIPE),
        "cluster_config.k8s.tunix_image=test-image",
        "actor_model_config.model_path=gs://test-bucket/ckpt",
        "data_config.dataset_path=gs://test-bucket/data",
    )
    cfg = run.resolve().config

    self.assertEqual(cfg["model_config"]["model_name"], "Qwen3.5-35B-A3B")
    self.assertEqual(cfg["model_config"]["model_source"], "maxtext")
    self.assertEqual(
        cfg["actor_model_config"]["model_path"], "gs://test-bucket/ckpt"
    )
    self.assertEqual(
        cfg["data_config"]["dataset_path"], "gs://test-bucket/data"
    )
    self.assertEqual(
        cfg["data_module"], "tunix.experimental.examples.deepswe_dist.prewarm"
    )
    self.assertIn(
        "--xla_tpu_scoped_vmem_limit_kib=65536",
        cfg["cluster_config"]["roles"]["trainer"]["env"]["LIBTPU_INIT_ARGS"],
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

  def test_stop(self):
    run = self.dry_run(
        str(_RECIPE),
        "stop",
        "actor_model_config.model_path=gs://test-bucket/ckpt",
    )
    self.assertIn("kubectl delete jobset testuser-orch", run.stdout)


if __name__ == "__main__":
  absltest.main()

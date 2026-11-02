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

"""Tests for `launcher_testing`."""

from __future__ import annotations

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config

# No example uses the shared launcher in this package, so these tests dry-run
# `launch.sh` itself.
_LAUNCH_SH = pathlib.Path(resolve_config.__file__).parent / "launch.sh"


class DryRunTest(absltest.TestCase):

  def _dry_run(
      self, script: pathlib.Path, *args: str
  ) -> launcher_testing.DryRun:
    return launcher_testing.dry_run(
        script, args, tmp_dir=pathlib.Path(self.create_tempdir().full_path)
    )

  def test_resolves_in_the_dry_run_environment(self):
    # A host environment that would change the config or leak a token.
    host_env = {
        "USER": "someone",
        "HF_TOKEN": "real-secret",
        "T_BATCH_SIZE": "7",
    }
    with mock.patch.dict(os.environ, host_env):
      run = self._dry_run(
          _LAUNCH_SH,
          "cluster_config.target=k8s",
          "cluster_config.k8s.tunix_image=test-image",
      )
      hp = run.resolve()

    self.assertEqual(run.returncode, 0, msg=run.stderr)
    self.assertEqual(
        run.recorded_config_args(),
        (
            "cluster_config.dry_run=true",
            f"cluster_config.run_id={launcher_testing.RUN_ID}",
            "cluster_config.target=k8s",
            "cluster_config.k8s.tunix_image=test-image",
        ),
    )
    # The resolved config matches the rendered manifests.
    self.assertEqual(
        hp.config["cluster_config"]["k8s"]["orchestrator_id"], "testuser-orch"
    )
    self.assertIn("kubectl delete jobset testuser-orch", run.stdout)
    # `T_BATCH_SIZE` does not override the base config.
    self.assertEqual(hp.config["batch_size"], 4)
    self.assertNotIn("real-secret", run.stdout)

  def test_exit_before_resolve_config_keeps_the_output(self):
    run = self._dry_run(_LAUNCH_SH, "--help")
    self.assertEqual(run.returncode, 0, msg=run.stderr)
    self.assertIn("Usage:", run.stdout)
    self.assertIsNone(run.config_args)
    with self.assertRaisesRegex(ValueError, "before running resolve_config"):
      run.resolve()

  def test_missing_script_reports_the_shell_error(self):
    script = pathlib.Path(self.create_tempdir().full_path) / "missing.sh"
    run = self._dry_run(script)
    self.assertNotEqual(run.returncode, 0)
    self.assertIn("missing.sh", run.stderr)
    self.assertIsNone(run.config_args)


if __name__ == "__main__":
  absltest.main()

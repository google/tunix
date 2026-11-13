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

"""Tests for PEP 562 lazy exports on `tunix.dist_rl.rollout`."""

import os
import subprocess
import sys

if os.environ.get("_ROLLOUT_LAZY_SUBPROCESS") == "1":
  import tunix.dist_rl.rollout  # pylint: disable=g-import-not-at-top,unused-import

  bad = [
      m
      for m in sys.modules
      if m == "jax"
      or m.startswith("jax.")
      or m == "flax"
      or m.startswith("flax.")
  ]
  if bad:
    raise AssertionError(f"Unexpected eager imports: {bad}")
  sys.exit(0)

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from tunix.dist_rl import rollout  # pylint: disable=g-import-not-at-top


class RolloutLazyImportTest(absltest.TestCase):

  def test_import_rollout_does_not_pull_jax_or_flax(self):
    if sys.executable:
      code = (
          "import sys; "
          f"sys.path[:] = {sys.path!r}; "
          "import tunix.dist_rl.rollout; "
          "bad = [m for m in sys.modules if m == 'jax' or m.startswith('jax.') "
          "or m == 'flax' or m.startswith('flax.')]; "
          "assert not bad, f'unexpected eager imports: {bad}'"
      )
      cmd = [sys.executable, "-c", code]
      env = os.environ
    else:
      cmd = [sys.argv[0]]
      env = {**os.environ, "_ROLLOUT_LAZY_SUBPROCESS": "1"}
    proc = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.returncode,
        0,
        msg=f"stdout={proc.stdout}\nstderr={proc.stderr}",
    )

  def test_all_exports_resolve_and_unknown_attr_raises(self):
    self.assertNotEmpty(rollout.__all__)
    directory = dir(rollout)
    for name in rollout.__all__:
      self.assertIn(name, directory)
      val = getattr(rollout, name)
      self.assertIsNotNone(val)
      self.assertIn(name, vars(rollout))

    with self.assertRaisesRegex(AttributeError, "has no attribute"):
      _ = getattr(rollout, "_definitely_missing_rollout_attr")


if __name__ == "__main__":
  absltest.main()

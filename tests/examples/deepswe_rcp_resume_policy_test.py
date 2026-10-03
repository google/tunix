# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the RCP checkpoint-restore policy in run_deepswe_dist."""

from __future__ import annotations

import types
from unittest import mock

from absl.testing import absltest
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist
from tunix.utils import mllog_utils


def _args(**overrides) -> types.SimpleNamespace:
  base = dict(rcp_logging=True, allow_checkpoint_resume=False)
  base.update(overrides)
  return types.SimpleNamespace(**base)


class ResumePolicyFlagTest(absltest.TestCase):

  def test_allow_checkpoint_resume_defaults_off(self):
    args = run_deepswe_dist._parse_args(["--rcp_logging"])
    self.assertFalse(args.allow_checkpoint_resume)
    self.assertTrue(run_deepswe_dist._disallow_checkpoint_resume(args))

  def test_opt_out_re_enables_resume_for_rcp_runs(self):
    args = run_deepswe_dist._parse_args(
        ["--rcp_logging", "--allow_checkpoint_resume"]
    )
    self.assertTrue(args.allow_checkpoint_resume)
    self.assertFalse(run_deepswe_dist._disallow_checkpoint_resume(args))

  def test_non_rcp_runs_are_never_restricted(self):
    args = run_deepswe_dist._parse_args([])
    self.assertFalse(args.rcp_logging)
    self.assertFalse(run_deepswe_dist._disallow_checkpoint_resume(args))


class EmitRcpRestoreEventsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.init_print = self.enter_context(
        mock.patch.object(mllog_utils, "init_print", autospec=True)
    )
    self.train_start = self.enter_context(
        mock.patch.object(mllog_utils, "train_start", autospec=True)
    )
    self.dataset = ["task-0", "task-1"]

  def test_no_events_without_rcp_logging(self):
    run_deepswe_dist._emit_rcp_restore_events(
        _args(rcp_logging=False), self.dataset, 18
    )
    self.init_print.assert_not_called()
    self.train_start.assert_not_called()

  def test_fresh_run_logs_step_zero_and_opens_the_run(self):
    args = _args()
    run_deepswe_dist._emit_rcp_restore_events(args, self.dataset, 0)
    self.init_print.assert_called_once_with(
        args, train_dataset=self.dataset, init_checkpoint_step=0
    )
    self.train_start.assert_called_once_with(args, step=0)

  def test_disallowed_resume_logs_true_step_but_never_opens_the_run(self):
    args = _args()
    with self.assertLogs(level="ERROR") as logs:
      run_deepswe_dist._emit_rcp_restore_events(args, self.dataset, 18)
    self.init_print.assert_called_once_with(
        args, train_dataset=self.dataset, init_checkpoint_step=18
    )
    self.train_start.assert_not_called()
    self.assertRegex(
        "\n".join(logs.output),
        r"restored checkpoint step 18.*ALLOW_CHECKPOINT_RESUME=true",
    )

  def test_opted_out_resume_logs_true_step_and_warns(self):
    args = _args(allow_checkpoint_resume=True)
    with self.assertLogs(level="WARNING") as logs:
      run_deepswe_dist._emit_rcp_restore_events(args, self.dataset, 18)
    self.init_print.assert_called_once_with(
        args, train_dataset=self.dataset, init_checkpoint_step=18
    )
    self.train_start.assert_called_once_with(args, step=18)
    self.assertRegex(
        "\n".join(logs.output),
        r"init_checkpoint_step=18.*NOT submission-compliant",
    )


if __name__ == "__main__":
  absltest.main()

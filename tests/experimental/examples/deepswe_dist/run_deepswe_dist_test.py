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

"""Unit tests for experimental DeepSWE orchestrator fault-tolerance and startup wiring."""

from __future__ import annotations

import os
import pathlib
import shlex
import signal
import stat
import subprocess
import unittest
from unittest import mock

from absl.testing import absltest

# `r2egym` is an optional dependency in OSS (`swe_agent` requires it at import
# time); match `deepswe_test.py` so OSS CPU CI skips cleanly when absent.
try:
  import r2egym  # pylint: disable=unused-import,g-import-not-at-top
except ImportError as exc:
  raise unittest.SkipTest("r2egym is not installed") from exc

# pylint: disable=g-import-not-at-top
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import fault_tolerance_manager
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist

# pylint: enable=g-import-not-at-top

_FT_ENV_KEYS = (
    "MIN_ROLLOUT_REPLICAS",
    "MIN_ROLLOUT_WORKERS",
    "ROLLOUT_FAULT_TOLERANCE",
    "MAX_TASK_RETRIES",
    "POLL_RETRY_BUDGET_S",
    "MIN_ACTIVE_ROLLOUT_RATIO",
    "MIN_ACTIVE_ROLLOUT_GRACE_S",
)


def _clean_env(**overrides: str) -> dict[str, str]:
  env = {k: v for k, v in os.environ.items() if k not in _FT_ENV_KEYS}
  env.update(overrides)
  return env


class RunDeepsweDistTest(absltest.TestCase):

  def test_defaults_match_fault_tolerance_config_and_rollout_replicas(self):
    with mock.patch.dict(
        os.environ, _clean_env(ROLLOUT_FAULT_TOLERANCE=""), clear=True
    ):
      args = run_deepswe_dist._parse_args(["--rollout_replicas=4"])

    self.assertEqual(
        run_deepswe_dist._build_fault_tolerance_config(args),
        datatypes.FaultToleranceConfig(),
    )
    self.assertEqual(
        run_deepswe_dist._required_workers(args),
        {
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 4,
            datatypes.Role.REFERENCE: 0,
        },
    )

  def test_cli_and_env_overrides_build_ft_config_and_min_rollouts(self):
    env = _clean_env(
        MIN_ROLLOUT_REPLICAS="2",
        ROLLOUT_FAULT_TOLERANCE="false",
        MAX_TASK_RETRIES="3",
        POLL_RETRY_BUDGET_S="45.5",
        MIN_ACTIVE_ROLLOUT_RATIO="null",
        MIN_ACTIVE_ROLLOUT_GRACE_S="1200",
    )
    with mock.patch.dict(os.environ, env, clear=True):
      args = run_deepswe_dist._parse_args(
          ["--rollout_replicas=4", "--beta=0.04"]
      )

    self.assertEqual(
        run_deepswe_dist._build_fault_tolerance_config(args),
        datatypes.FaultToleranceConfig(
            enabled=False,
            max_task_retries=3,
            poll_retry_budget_s=45.5,
            min_active_rollout_ratio=None,
            min_active_rollout_grace_s=1200.0,
        ),
    )
    self.assertEqual(
        run_deepswe_dist._required_workers(args),
        {
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 2,
            datatypes.Role.REFERENCE: 1,
        },
    )

    with mock.patch.dict(os.environ, env, clear=True):
      cli_args = run_deepswe_dist._parse_args([
          "--rollout_replicas=6",
          "--min_rollout_replicas=3",
          "--rollout_fault_tolerance=true",
          "--max_task_retries=0",
          "--poll_retry_budget_s=null",
          "--min_active_rollout_ratio=0.75",
          "--min_active_rollout_grace_s=0",
      ])
    self.assertEqual(
        run_deepswe_dist._build_fault_tolerance_config(cli_args),
        datatypes.FaultToleranceConfig(
            enabled=True,
            max_task_retries=0,
            poll_retry_budget_s=None,
            min_active_rollout_ratio=0.75,
            min_active_rollout_grace_s=0.0,
        ),
    )
    self.assertEqual(
        run_deepswe_dist._required_workers(cli_args)[datatypes.Role.ROLLOUT], 3
    )

  def test_invalid_ft_and_replica_settings_fail_loud(self):
    bad_ft_cases = (
        ["--max_task_retries=-1"],
        ["--poll_retry_budget_s=-0.1"],
        ["--poll_retry_budget_s=inf"],
        ["--poll_retry_budget_s=nan"],
        ["--min_active_rollout_ratio=0.0"],
        ["--min_active_rollout_ratio=1.2"],
        ["--min_active_rollout_ratio=nan"],
        ["--min_active_rollout_grace_s=-1.0"],
        ["--min_active_rollout_grace_s=inf"],
        ["--min_active_rollout_grace_s=nan"],
    )
    with mock.patch.dict(os.environ, _clean_env(), clear=True):
      for argv in bad_ft_cases:
        with self.subTest(argv=argv):
          args = run_deepswe_dist._parse_args(argv)
          with self.assertRaises(ValueError):
            run_deepswe_dist._build_fault_tolerance_config(args)

      for argv in (
          ["--rollout_replicas=0"],
          ["--rollout_replicas=2", "--min_rollout_replicas=0"],
          ["--rollout_replicas=2", "--min_rollout_replicas=3"],
      ):
        with self.subTest(argv=argv):
          args = run_deepswe_dist._parse_args(argv)
          with self.assertRaises(ValueError):
            run_deepswe_dist._required_workers(args)

  def test_k8s_launcher_dry_run_forwards_ft_and_min_rollout_flags(self):
    launcher = pathlib.Path(run_deepswe_dist.__file__).with_name(
        "k8s_launcher.sh"
    )
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    record_file = tmp_dir / "yaml_gen_args"
    python_recorder = tmp_dir / "python_recorder.sh"
    python_recorder.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "-c" ]]; then cat; exit 0; fi\n'
        f"printf '%s\\0' \"$@\" > {shlex.quote(str(record_file))}\n",
        encoding="utf-8",
    )
    python_recorder.chmod(python_recorder.stat().st_mode | stat.S_IXUSR)

    env = _clean_env(
        USER="johndoe",
        TUNIX_IMAGE="test-image",
        PYTHON=str(python_recorder),
        DISABLE_JAX_CACHE="1",
        ROLLOUT_REPLICAS="4",
        MIN_ROLLOUT_REPLICAS="2",
        ROLLOUT_FAULT_TOLERANCE="false",
        MAX_TASK_RETRIES="3",
        POLL_RETRY_BUDGET_S="45.5",
        MIN_ACTIVE_ROLLOUT_RATIO="null",
        MIN_ACTIVE_ROLLOUT_GRACE_S="1200",
    )
    subprocess.run(
        ["bash", str(launcher), "--command=orchestrator", "--dry-run"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    legacy = subprocess.run(
        ["bash", str(launcher), "--command=orchestrator", "--dry-run"],
        env={**env, "MIN_ROLLOUT_WORKERS": "2"},
        check=False,
        capture_output=True,
        text=True,
    )
    self.assertNotEqual(legacy.returncode, 0)
    self.assertIn("MIN_ROLLOUT_REPLICAS", legacy.stderr)
    yaml_gen_args = [
        arg.decode("utf-8")
        for arg in record_file.read_bytes().split(b"\0")
        if arg
    ]
    startup_cmd = next(
        arg.split("=", 1)[1]
        for arg in yaml_gen_args
        if arg.startswith("--worker_startup_command=")
    )
    tokens = shlex.split(startup_cmd)
    proc_main_idx = next(
        i for i, tok in enumerate(tokens) if tok.startswith("--process_main=")
    )
    orch_argv = tokens[proc_main_idx + 1 :]

    with mock.patch.dict(os.environ, _clean_env(), clear=True):
      parsed = run_deepswe_dist._parse_args(orch_argv)
    self.assertEqual(
        run_deepswe_dist._build_fault_tolerance_config(parsed),
        datatypes.FaultToleranceConfig(
            enabled=False,
            max_task_retries=3,
            poll_retry_budget_s=45.5,
            min_active_rollout_ratio=None,
            min_active_rollout_grace_s=1200.0,
        ),
    )
    self.assertEqual(
        run_deepswe_dist._required_workers(parsed)[datatypes.Role.ROLLOUT], 2
    )

  def test_main_aborts_cluster_and_preserves_sigterm_on_run_failure(self):
    fake_cluster = mock.MagicMock()
    fake_cluster.trajectory_store = None
    fake_cluster.registry.infos.return_value = []
    fake_cluster.run.side_effect = fault_tolerance_manager.RolloutCapacityError(
        "below floor"
    )

    fake_fleet = mock.MagicMock()
    fake_prewarm = mock.MagicMock()
    fake_program = mock.MagicMock()
    observed_sigterm_after_program_init: list[object] = []

    def _init_fleet(*unused_args, **unused_kwargs):
      # Simulate a sandbox helper attempting to overwrite SIGTERM.
      signal.signal(signal.SIGTERM, signal.SIG_IGN)
      return fake_fleet

    def _make_program(*unused_args, **unused_kwargs):
      # Simulate `AsyncTrajectoryLogger.__init__` clobbering SIGTERM.
      signal.signal(signal.SIGTERM, signal.SIG_DFL)
      return fake_program

    def _bring_up(dummy_data=None):
      del dummy_data
      observed_sigterm_after_program_init.append(
          signal.getsignal(signal.SIGTERM)
      )

    fake_cluster.bring_up_workers.side_effect = _bring_up

    fake_tokenizer = mock.MagicMock(pad_token_id=0, eos_token_id=1)
    fake_tokenizer.__len__.return_value = 100
    fake_ctx = mock.MagicMock()
    log_dir = self.create_tempdir().full_path

    with (
        mock.patch.dict(
            os.environ,
            _clean_env(ORCHESTRATOR_ID="johndoe-orch"),
            clear=True,
        ),
        mock.patch.object(
            run_deepswe_dist.AutoTokenizer,
            "from_pretrained",
            return_value=fake_tokenizer,
        ),
        mock.patch.object(
            run_deepswe_dist.deepswe,
            "load_deepswe_dataset",
            return_value=[{"problem_statement": "fix bug", "instance_id": "1"}],
            autospec=True,
        ),
        mock.patch.object(
            run_deepswe_dist.sandbox_utils,
            "init_global_fleet",
            side_effect=_init_fleet,
            autospec=True,
        ) as init_fleet_mock,
        mock.patch.object(
            run_deepswe_dist.sandbox_utils,
            "PrewarmDatasetIterator",
            return_value=fake_prewarm,
            autospec=True,
        ),
        mock.patch.object(
            run_deepswe_dist.sandbox_utils,
            "teardown_global_fleet",
            autospec=True,
        ) as teardown_fleet_mock,
        mock.patch.object(
            run_deepswe_dist.orchestrator,
            "ClusterOrchestrator",
            return_value=fake_cluster,
            autospec=True,
        ) as cluster_cls,
        mock.patch.object(
            run_deepswe_dist.rl_program,
            "StandardRLProgram",
            side_effect=_make_program,
            autospec=True,
        ),
    ):
      with self.assertRaisesRegex(
          fault_tolerance_manager.RolloutCapacityError, "below floor"
      ):
        run_deepswe_dist.main(
            [
                "--rollout_replicas=4",
                "--min_rollout_replicas=2",
                "--max_task_retries=3",
                "--use_agent_sandbox",
                f"--log_dir={log_dir}",
                "--stop_workers_on_exit",
            ],
            context=fake_ctx,
        )

    cluster_cls.assert_called_once()
    self.assertEqual(
        cluster_cls.call_args.kwargs["fault_tolerance_config"],
        datatypes.FaultToleranceConfig(max_task_retries=3),
    )
    init_fleet_mock.assert_called_once()
    self.assertFalse(init_fleet_mock.call_args.kwargs["teardown_on_exit"])
    fake_cluster.registry.wait_for_workers.assert_called_once_with(
        min_workers={
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 2,
            datatypes.Role.REFERENCE: 0,
        },
        timeout=None,
    )
    self.assertLen(observed_sigterm_after_program_init, 1)
    self.assertNotIn(
        observed_sigterm_after_program_init[0],
        (signal.SIG_DFL, signal.SIG_IGN),
    )
    fake_prewarm.close.assert_called_once()
    teardown_fleet_mock.assert_called_once()
    fake_program.close.assert_called_once()
    fake_cluster.shutdown.assert_called_once_with(abort=True)


if __name__ == "__main__":
  absltest.main()

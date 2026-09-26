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

"""Unit tests for the Trajectory Explorer CLI entry point."""

import contextlib
import io
import json
from absl.testing import absltest
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory.explorer import main
from tunix.experimental.trajectory.explorer.commands import ping


def _parse_cmd(cmd: str = "") -> main.ExplorerConfig:
  """Parses a command string into an ExplorerConfig instance.

  Args:
    cmd: Optional CLI arguments string (e.g. '--json ping').

  Returns:
    Parsed ExplorerConfig dataclass instance.
  """
  argv = cmd.split(" ") if cmd else []
  return main.parse_args(argv)


class ParseArgsTest(absltest.TestCase):

  def test_defaults_to_an_empty_in_memory_store(self) -> None:
    config = _parse_cmd()
    self.assertEqual(config.backend, "memory")
    self.assertEmpty(config.root_dir)
    self.assertEmpty(config.run_id)
    self.assertFalse(config.json)
    self.assertIsInstance(config.cmd, ping.PingCommand)

  def test_json_flag(self) -> None:
    config = _parse_cmd("--json")
    self.assertTrue(config.json)

  def test_file_backend_flags(self) -> None:
    config = _parse_cmd(
        "--backend file --root_dir /tmp/trajectories --run_id r"
    )
    self.assertEqual(config.backend, "file")
    self.assertEqual(config.root_dir, "/tmp/trajectories")
    self.assertEqual(config.run_id, "r")

  def test_subcommand_ping(self) -> None:
    config = _parse_cmd("ping")
    self.assertIsInstance(config.cmd, ping.PingCommand)

  def test_global_flags_compose_with_a_subcommand(self) -> None:
    config = _parse_cmd("--backend file --root_dir /tmp/t --run_id r ping")
    self.assertEqual(config.backend, "file")
    self.assertEqual(config.run_id, "r")
    self.assertIsInstance(config.cmd, ping.PingCommand)


class GetReaderTest(absltest.TestCase):

  def test_memory_backend_starts_empty(self) -> None:
    reader = main.get_reader(_parse_cmd("--backend memory"))

    self.assertIsInstance(reader, in_memory_store.InMemoryTrajectoryStore)
    self.assertEmpty(reader.get_trajectories_metadata())

  def test_file_backend_opens_the_named_run(self) -> None:
    root_dir = self.create_tempdir().full_path

    reader = main.get_reader(
        _parse_cmd(f"--backend file --root_dir {root_dir} --run_id my_run")
    )
    self.addCleanup(reader.close)

    self.assertIsInstance(reader, file_store.FileTrajectoryStore)

  def test_file_backend_requires_a_root_dir(self) -> None:
    with self.assertRaisesRegex(ValueError, "root_dir"):
      main.get_reader(_parse_cmd("--backend file --run_id my_run"))

  def test_file_backend_requires_a_run_id(self) -> None:
    with self.assertRaisesRegex(ValueError, "run_id"):
      main.get_reader(_parse_cmd("--backend file --root_dir /tmp/trajectories"))


class RunCliTest(absltest.TestCase):

  def test_ping_succeeds_with_json_output(self) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      exit_code = main.run_cli(["--json", "ping"])

    self.assertEqual(exit_code, 0)
    self.assertEqual(
        json.loads(stdout.getvalue()),
        {"status": "ok", "trajectories_count": 0},
    )

  def test_missing_backend_flag_exits_as_a_usage_error(self) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
      exit_code = main.run_cli(["--backend", "file", "--json", "ping"])

    self.assertEqual(exit_code, 2)
    self.assertEmpty(stdout.getvalue())
    self.assertIn("trajectory-explore: error:", stderr.getvalue())


if __name__ == "__main__":
  absltest.main()
